// embed: h[t][:] = table[tokens[t]][:] for T tokens — one SIMD-group per token, lanes copy 16-byte words.
//
// EMBED_PACKED 0: a row-major BF16 table [vocab][K].
// EMBED_PACKED 1: the table is the BF16 block-lane-major slab a tied lm_head streams (design §5.5): row v lives in
//                 block v / R at row v % R, lane ℓ's stripe (columns [ℓ·K/32, (ℓ+1)·K/32)) is UNIT_WORDS 16-byte
//                 words at unit_word(ℓ, r, j) (macros R, UNIT_WORDS, LANE_ORDER as for gemv_T; needs K % 256 == 0).
// Tokens outside [0, vocab) read row 0 (never out of bounds; the sampler guarantees valid ids).
// EMBED_DEQUANT 1: the packed slab is a quantized format (its decode snippet is pasted ahead of this file): every
//                 element is decoded (code · scale [+ bias] of its group) and rounded to BF16 — the row the reference
//                 model's dequantized embedding holds. Needs a per-tensor scale of 1 (int4_affine, int8).
// EMBED_IDS 1: a draft block (design §5.8) — row 0 reads tokens[0] (the anchor), rows ≥ 1 take the mask id from params.
// With STEP_STATE, T_SRC selects the row count: 0 = t_this_step, 1 = n_inject, 2 = the static T_STATIC_ROWS.
#ifndef EMBED_PACKED
#define EMBED_PACKED 0
#endif
#ifndef EMBED_IDS
#define EMBED_IDS 0
#endif
#ifndef EMBED_DEQUANT
#define EMBED_DEQUANT 0
#endif
#ifndef SCALE_BIAS
#define SCALE_BIAS 0
#endif
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef T_SRC
#define T_SRC 0
#endif
#ifndef T_STATIC_ROWS
#define T_STATIC_ROWS 1u
#endif
#ifndef SCALE_W0
#define SCALE_W0 PAYLOAD_WORDS       // the scale bytes start on the word after the payload, at its first uint (gemv_T's
#define SCALE_UOFF 0u                // SCALE_W0 / SCALE_UOFF: a ragged stripe keeps them in the partial tail word)
#endif

struct EmbedParams { uint k; uint t_active; uint vocab; uint mask_id; };

#if EMBED_PACKED
static inline uint unit_word(uint lane, uint r, uint j) {
#if LANE_ORDER == 0
  return (lane * R + r) * UNIT_WORDS + j;
#else
  return (r * UNIT_WORDS + j) * 32u + lane;
#endif
}
#ifdef LANES_PER_WORD
#error "sub-word units are the shader GEMV's: the tile and the gather read whole-word units"
#endif
#ifndef SCALE_PLACEMENT
#define SCALE_PLACEMENT 0            // 1: the block's scales in their own region after its payload words (blm.py, #101):
#endif                               //    lane ln's row r scales start (ln * SCALE_RUN) % 16 bytes into word SCALE_WORD(ln, r, 0)
#if SCALE_PLACEMENT
#define SCALE_BASE (R * 32u * PAYLOAD_WORDS)
#define SCALE_WORD(ln, r, s) (SCALE_BASE + ((r) * 32u * SCALE_RUN + (ln) * SCALE_RUN) / 16u + (s))
#define SCALE_SOFF(ln) ((((ln) * SCALE_RUN) % 16u) / SCALE_UNIT_BYTES)
#else
#define SCALE_WORD(ln, r, s) unit_word((ln), (r), SCALE_W0 + (s))
#define SCALE_SOFF(ln) 0u
#endif
#if SCALE_PLACEMENT
#define BLOCK_WORDS (R * 32u * UNIT_WORDS + SCALE_REGION_WORDS)       // a block: its payload words then its scale region
#else
#define BLOCK_WORDS (R * 32u * UNIT_WORDS)
#endif
#endif

kernel void embed(device const int* tokens [[buffer(0)]], device const uint4* table [[buffer(1)]], device uint4* h [[buffer(2)]],
                  constant EmbedParams& p [[buffer(3)]],
#if STEP_STATE
                  device const StepState* st [[buffer(15)]],
#endif
                  uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint t = gid / sw;
#if STEP_STATE
  if (st->done || t >= ((T_SRC == 1) ? st->n_inject : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step))) return;
#else
  if (t >= p.t_active) return;
#endif
#if EMBED_IDS
  uint tok = (t == 0u) ? uint(tokens[0]) : p.mask_id;
#else
  uint tok = uint(tokens[t]);
#endif
  if (tok >= p.vocab) tok = 0u;
  device uint4* out = h + (ulong)t * (p.k / 8u);
#if EMBED_PACKED
  const uint b = tok / R, r = tok % R;
  device const uint4* blk = table + (ulong)b * BLOCK_WORDS;
#if EMBED_DEQUANT
  // the lane's stripe: PAYLOAD_WORDS words of codes, SCALE_WORDS words of group scales; out columns [lane·KL, +KL)
  uint scw[SCALE_WORDS * 4];
  for (uint s = 0; s < SCALE_WORDS; s++) { uint4 q = blk[SCALE_WORD(lane, r, s)]; scw[4 * s] = q.x; scw[4 * s + 1] = q.y; scw[4 * s + 2] = q.z; scw[4 * s + 3] = q.w; }
  const uint kl = K / 32u, lane_off = (lane * kl) % SCALE_GROUP;     // a stripe may start inside a group (ragged K)
  device ushort* orow = (device ushort*)out + lane * kl;
  for (uint j = 0; j < PAYLOAD_WORDS; j++) {
    float wv[WEIGHTS_PER_WORD];
    decode_word(blk[unit_word(lane, r, j)], wv);
    for (uint e = 0; e < WEIGHTS_PER_WORD; e++) {
      const uint c = j * WEIGHTS_PER_WORD + e;
      if (c >= kl) break;                                              // the padding of a partial last word
      const uint g = (lane_off + c) / SCALE_GROUP;
      float v = wv[e] * decode_scale(scw + SCALE_UOFF, SCALE_SOFF(lane) + g);
#if SCALE_BIAS
      v += decode_bias(scw + SCALE_UOFF, SCALE_SOFF(lane) + g);
#endif
      uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u);
      orow[c] = ushort(u >> 16);
    }
  }
#else
  for (uint j = 0; j < UNIT_WORDS; j++) out[lane * UNIT_WORDS + j] = blk[unit_word(lane, r, j)];
#endif
#else
  device const uint4* row = table + (ulong)tok * (p.k / 8u);
  for (uint j = lane; j < p.k / 8u; j += 32u) out[j] = row[j];
#endif
}
