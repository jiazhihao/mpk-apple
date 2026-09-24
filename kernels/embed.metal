// embed: h[t][:] = table[tokens[t]][:] for T tokens — one SIMD-group per token, lanes copy 16-byte words.
//
// EMBED_PACKED 0: a row-major BF16 table [vocab][K].
// EMBED_PACKED 1: the table is the BF16 block-lane-major slab a tied lm_head streams (design §5.5): row v lives in
//                 block v / R at row v % R, lane ℓ's stripe (columns [ℓ·K/32, (ℓ+1)·K/32)) is UNIT_WORDS 16-byte
//                 words at unit_word(ℓ, r, j) (macros R, UNIT_WORDS, LANE_ORDER as for gemv_T; needs K % 256 == 0).
// Tokens outside [0, vocab) read row 0 (never out of bounds; the sampler guarantees valid ids).
#ifndef EMBED_PACKED
#define EMBED_PACKED 0
#endif

struct EmbedParams { uint k; uint t_active; uint vocab; uint pad; };

#if EMBED_PACKED
static inline uint unit_word(uint lane, uint r, uint j) {
#if LANE_ORDER == 0
  return (lane * R + r) * UNIT_WORDS + j;
#else
  return (r * UNIT_WORDS + j) * 32u + lane;
#endif
}
#endif

kernel void embed(device const int* tokens [[buffer(0)]], device const uint4* table [[buffer(1)]], device uint4* h [[buffer(2)]],
                  constant EmbedParams& p [[buffer(3)]],
                  uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint t = gid / sw;
  if (t >= p.t_active) return;
  uint tok = uint(tokens[t]);
  if (tok >= p.vocab) tok = 0u;
  device uint4* out = h + (ulong)t * (p.k / 8u);
#if EMBED_PACKED
  const uint b = tok / R, r = tok % R;
  device const uint4* blk = table + (ulong)b * (R * 32u * UNIT_WORDS);
  for (uint j = 0; j < UNIT_WORDS; j++) out[lane * UNIT_WORDS + j] = blk[unit_word(lane, r, j)];
#else
  device const uint4* row = table + (ulong)tok * (p.k / 8u);
  for (uint j = lane; j < p.k / 8u; j += 32u) out[j] = row[j];
#endif
}
