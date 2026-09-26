// gemm_tile: y[t][row] = sum_k dequant(W[row][k]) * x[t][k] for T <= TM tokens through the M5 neural accelerators
// (mpp::tensor_ops::matmul2d — design §5.6 / §5.12, plan M9, #50), reading the same block-lane-major pack as gemv_T.
//
// The Python side prepends `#include <metal_stdlib>`, the MPP header and a FORMAT SNIPPET (WEIGHTS_PER_WORD,
// SCALE_GROUP, decode_word, decode_scale[, decode_bias]) and sets the macros below. One SIMD-group multiplies one
// [TN rows x TK columns] weight tile (4096 weights: 16 x 256 up to 16 tokens, 32 x 128 at 32 — measured,
// docs/research/decode-kernels.md §6) at a time into a cooperative destination tensor; the right operand is a
// *cooperative* tensor filled straight from the pack words — no threadgroup staging, no barrier, no exchange:
//
//   * the accelerator's register layout (measured; tests/kernels/test_gemm_tile.py checks it against
//     get_multidimensional_index) gives thread `lane` the reduction slots 4*(bit0 + 2*bit3) + 16*jump + q for the rows
//     (bits 1,2,4) + 8*slot, i.e. a quarter of the tile's columns, in runs of 4, for TN/8 rows;
//   * the reduction index is free, so the tile's columns are permuted: quad member (bit0, bit3) owns TK/4 *consecutive*
//     pack columns of each of its rows — one contiguous half-word (4-bit formats at TK = 64), whole 16-byte words
//     otherwise — which it loads and decodes alone with the format snippet, every weight decoded once, one block
//     scale per 16-column chunk, and lands in its operand registers as BF16 (the reference's dequantized dtype);
//   * a tile's row piece is TK/WPW adjacent lanes' word j of the pack — adjacent 16-byte words in the interleaved
//     layout, a whole 128-byte cache line at TK = 256 (then lane group q runs outer, word j inner, and a thread's
//     block-scale words stay in registers across the lane group's words: Q_OUTER, SCALE_CACHE);
//   * the activation rows are read through a device tensor in the same order (x' from x_permute: pack order —
//     word j of lane l -> columns [j*32*WPW + l*WPW, +WPW) — then the slot order inside each tile), so the left
//     operand of a tile is one contiguous slice of x'.
//
// Geometry: the crew (n_sg SIMD-groups take static slices of the row tiles; S SIMD-groups per threadgroup share
// nothing; more SIMD-groups per core hide more latency), or with KSPLIT > 1 one row tile per threadgroup of KSPLIT
// SIMD-groups, each streaming a contiguous K / KSPLIT slice of the tile and the partial tiles reduced through
// threadgroup memory (the register layout is the same in every SIMD-group; SIMD-group 0 adds and runs the
// epilogue) — the remedy for the shapes whose row tiles cannot occupy the crew (256 tiles of a 4096-row projection
// against 480 SIMD-groups: decode-kernels.md §6). Macros: K, R (rows per pack block, divides TN), TM (token
// rows: 8 leaves half of the accelerator's 16-row minimum unused, so 16 costs the same), TN, TK, LANE_ORDER,
// UNIT_WORDS, PAYLOAD_WORDS, SCALE_W0, SCALE_UOFF, SCALE_WORDS (kernels.unit_geometry), Q_OUTER, SCALE_CACHE, OUT_BF16,
// plus the snippet's. Requires K % (32*WPW) == 0 and K % TK == 0, T_act <= TM (rows beyond t_active are zero in x'
// and are not written). Measured on the M5 Pro at 17408 x 5120: NVFP4 177 GB/s, FP8 253, INT4 204 at 8 or 16 tokens
// — 0.9-1.1x a T = 1 GEMV pass — against p14's staged tile (119 / 186); at 32 tokens the un-overlapped fill and
// the activation traffic leave it below the staged tile (a multi-SIMD-group staged variant is the follow-up).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef OUT_BF16
#define OUT_BF16 0
#endif
#ifndef STEP_STATE
#define STEP_STATE 0                 // 1: the row count comes from StepState (buffer 15) and the dispatch is a per-T variant
#endif
#ifndef T_SRC
#define T_SRC 0                      // with STEP_STATE: 0 = t_this_step, 1 = n_inject, 2 = the T_STATIC_ROWS macro
#endif
#ifndef T_STATIC_ROWS
#define T_STATIC_ROWS 1u
#endif
#ifndef T_LO
#define T_LO 0                       // with T_HI: the variant runs only when T_LO < T_act <= T_HI (design §5.7)
#endif
#ifndef EPILOGUE
#define EPILOGUE 0                   // 1: y = bf16(v + residual) (EPILOGUE_ROUND: v rounded to BF16 first); 2: silu(gate)·up over
#endif                               //    chunk-interleaved rows (block b = gate rows [0, CHUNK) | up rows [CHUNK, R)), N/2 outputs
#ifndef EPILOGUE_ROUND
#define EPILOGUE_ROUND 0
#endif
#ifndef STAT_OUT
#define STAT_OUT 0                   // 1: per-block partial sums of squares of the BF16-rounded outputs, stat_out[t * n_blocks + block]
#endif
#if R != 16 && R != 8
#error "gemm_tile: the epilogues index pack blocks of 8 or 16 rows"
#endif
#define CHUNK (R / 2u)
#define PAIR_XOR ((R == 16u) ? 8u : 1u)                    // the lane holding the partner row of a silu_mul pair
#ifndef Q_OUTER
#define Q_OUTER 0                    // 1: lane group outer, word inner (a tile's row piece is a whole cache line)
#endif
#ifndef SCALE_CACHE
#define SCALE_CACHE 0                // 1: keep the scale words of a thread's rows in registers across a lane group's words (needs Q_OUTER)
#endif
#if SCALE_CACHE && !Q_OUTER
#error "gemm_tile: SCALE_CACHE needs Q_OUTER"
#endif
#ifndef EXP_MODE
#define EXP_MODE 0                   // bench experiments (tools/bench/gemm_bench.py --macro EXP_MODE=n): 2 = the matmul with one
                                     // fill (no loads, no decode); 5 = the fill from synthetic words (no loads); 6 = the
                                     // activation slice pinned to the first tile (A traffic served from L1)
#endif
#ifndef SCALE_BIAS
#define SCALE_BIAS 0
#endif
#ifndef SCALE_W0
#define SCALE_W0 PAYLOAD_WORDS
#define SCALE_UOFF 0u
#endif
#ifndef KSPLIT
#define KSPLIT 1u                                         // SIMD-groups per row tile, each a contiguous K slice (1 … 16)
#endif
#ifndef TN
#define TN 64u                                            // rows per tile (16, 32 or 64)
#endif
#ifndef TK
#define TK 64u                                            // columns per tile (64, 128 or 256): TN * TK = 4096
#endif
#define KL (K / 32u)
#define WPW WEIGHTS_PER_WORD
#define LPT (TK / WPW)                                    // lanes (pack words) per K tile
#define KT (K / TK)                                       // K tiles per row
#define NB (TN / R)                                       // pack blocks per row tile
#define CT (TK / 4u)                                      // a thread's consecutive columns per row (its quad's quarter)
#define NW ((CT >= WPW) ? (CT / WPW) : 1u)                // words holding them (one half-word when CT < WPW)
#define NCH (CT / 16u)                                    // 16-column chunks of them (one block scale each)
#if SCALE_GROUP > 0 && (SCALE_GROUP % 16) != 0
#error "gemm_tile: the block scale group must be a multiple of 16 columns"
#endif
#define NS_B (TN / 8u)                                    // row slots per thread in the right operand
#define NB_C ((TM > 16u) ? (TM / 16u) : 1u)               // 16-row blocks of the destination (its element order is
                                                          // q, slot (2), jump (TN/16), block — the right operand's is q, slot (8), jump)
#define C_CAP (NB_C * TN / 2u)                            // destination elements per thread (16 · NB_C rows × TN over 32 lanes)
#define KT_S (KT / KSPLIT)                                // K tiles per slice
#if (KT % KSPLIT) != 0
#error "gemm_tile: KSPLIT must divide the K tiles"
#endif
#if SCALE_CACHE && (KT_S % PAYLOAD_WORDS) != 0
#error "gemm_tile: with the scale cache a K slice must be whole lane groups (KSPLIT divides 32 / LPT)"
#endif
#if SCALE_GROUP > 0
#if (KL % SCALE_GROUP) == 0
#define LANE_OFF 0u
#else
#define LANE_OFF ((ln * KL) % SCALE_GROUP)
#endif
#endif

struct GemmParams { uint n_rows; uint n_tiles; uint n_sg; uint t_active; float out_scale; uint tile0; uint n_blocks; uint pad; };
// n_rows / n_tiles / n_blocks / tile0 describe a row range of the slab (a whole slab: N / all tiles / all blocks / 0);
// outputs and stat partials are range-relative, the weights and row scales are addressed by the slab tile.

static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline float silu_f(float g) { return g / (1.0f + exp(-g)); }

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

constexpr constant auto desc = matmul2d_descriptor(int(TM), int(TN), int(TK), false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
using tA_t = tensor<device bfloat, dextents<int, 2>, tensor_inline>;

kernel void gemm_tile(device const uint4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
                      device bfloat* xp [[buffer(2)]],
#if OUT_BF16
                      device ushort* y [[buffer(3)]],
#else
                      device float* y [[buffer(3)]],
#endif
                      constant GemmParams& p [[buffer(4)]],
#if EPILOGUE == 1
                      device const ushort* residual [[buffer(7)]],
#endif
#if STAT_OUT
                      device float* stat_out [[buffer(8)]],
#endif
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
#if KSPLIT > 1
  const uint slice = sg % KSPLIT, sg_tile = sg / KSPLIT, n_tg = p.n_sg / KSPLIT;   // a threadgroup is the KSPLIT SIMD-groups of one tile
  threadgroup float part[KSPLIT - 1][32][C_CAP];                            // the partial tiles of slices 1 … KSPLIT-1
#else
  const uint sg_tile = sg, n_tg = p.n_sg;
#endif
#if STEP_STATE
  if (st->done) return;                                                     // uniform over the threadgroup: no barrier is skipped
  const uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step);
#ifdef T_HI
  if (T_act > T_HI || T_act <= T_LO) return;
#endif
#else
  const uint T_act = p.t_active;
#endif
  matmul2d<desc, execution_simdgroup> op;
  tA_t tA(xp, dextents<int, 2>(int(K), int(TM)));
  const uint c0b = 4u * ((lane & 1u) + 2u * ((lane >> 3) & 1u));      // this thread's column-run base
  const uint c1b = ((lane >> 1) & 3u) + 4u * ((lane >> 4) & 1u);      // this thread's row-slot base
  const uint mq = (lane & 1u) | (((lane >> 3) & 1u) << 1);            // its member id in the quad sharing those rows
  for (uint tile = p.tile0 + sg_tile; tile < p.tile0 + p.n_tiles; tile += n_tg) {
    auto bT = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
    auto cT = op.get_destination_cooperative_tensor<tA_t, decltype(bT), float>();
    for (uint16_t i = 0; i < cT.get_capacity(); i++) cT[i] = 0.0f;
#if SCALE_CACHE
    uint scc[NS_B][NW][SCALE_WORDS * 4];                                // the scale words of this thread's rows and lanes,
#endif                                                                  // kept across the PAYLOAD_WORDS tiles of a lane group
#if KSPLIT > 1
    for (uint kt = slice * KT_S; kt < (slice + 1u) * KT_S; kt++) {      // this SIMD-group's K slice (whole lane groups under Q_OUTER)
#else
    for (uint kt = 0; kt < KT; kt++) {
#endif
      // a tile's row piece is LPT adjacent lanes' word j, LPT*16 bytes of a 128-byte line: when that is the whole line
      // (Q_OUTER) lane group q runs outer and word j inner, so the scale words can be kept across j; otherwise j runs
      // outer so the lane groups sharing a line are consecutive tiles
#if Q_OUTER
      const uint q = kt / PAYLOAD_WORDS, j = kt % PAYLOAD_WORDS;        // pack word j of lanes [q*LPT, q*LPT + LPT)
#else
      const uint j = kt / (32u / LPT), q = kt % (32u / LPT);
#endif
#if EXP_MODE == 2
      if (kt == 0) {
#endif
      // the reduction index is permuted so that this thread's 16 slots of each row are 16 *consecutive* pack columns,
      // [16*mq, 16*mq + 16) of the tile (x_permute orders x' the same way): one contiguous half-word (4-bit formats),
      // a whole word (8-bit) or two words (bf16) per row, loaded by this thread alone — no exchange, no redundancy
      // beyond the other half of a 4-bit word, one block scale per row
#pragma clang loop unroll(full)
      for (uint s = 0; s < NS_B; s++) {
        const uint n = c1b + 8u * s;                                     // row inside the tile
        const uint b = tile * NB + n / R, r = n % R;                     // its pack block and row
        device const uint4* wb = w + (ulong)b * BLOCK_WORDS;
        const uint c0 = CT * mq;                                         // this thread's first tile column
        const uint lw0 = c0 / WPW, e0 = c0 % WPW;                        // its first word inside the tile and code offset
        const uint ln0 = q * LPT + lw0;                                  // the lane holding that word
        uint4 words[NW];
#pragma clang loop unroll(full)
        for (uint i = 0; i < NW; i++) {
#if EXP_MODE == 5
          words[i] = uint4(lane * 2654435761u + kt * 40503u + i, n * 97u + kt, i * 7u + lane, kt);
#else
          words[i] = wb[unit_word(ln0 + i, r, j)];
#endif
        }
#if CT < WPW
        words[0] = (e0 != 0u) ? uint4(words[0].z, words[0].w, 0u, 0u) : words[0];   // the half this thread owns
#endif
        float wv[NW * WPW];
#pragma clang loop unroll(full)
        for (uint i = 0; i < NW; i++) decode_word(words[i], wv + i * WPW);
#if SCALE_GROUP > 0
        float scv[NCH];                                                  // one block scale per 16-column chunk
#if SCALE_BIAS
        float bv[NCH];
#endif
#pragma clang loop unroll(full)
        for (uint i = 0; i < NW; i++) {
#if SCALE_CACHE
          if (j == 0u) {
#pragma clang loop unroll(full)
            for (uint sc = 0; sc < SCALE_WORDS; sc++) {
#if EXP_MODE == 5
              const uint4 v4 = uint4(0x38383838u + lane, 0x38383838u, 0x38383838u + kt, 0x38383838u);
#else
              const uint4 v4 = wb[SCALE_WORD(ln0 + i, r, sc)];
#endif
              scc[s][i][4 * sc] = v4.x; scc[s][i][4 * sc + 1] = v4.y; scc[s][i][4 * sc + 2] = v4.z; scc[s][i][4 * sc + 3] = v4.w;
            }
          }
          thread const uint* scw = scc[s][i];
#else
          uint scw[SCALE_WORDS * 4];
#pragma clang loop unroll(full)
          for (uint sc = 0; sc < SCALE_WORDS; sc++) {
#if EXP_MODE == 5
            const uint4 v4 = uint4(0x38383838u + lane, 0x38383838u, 0x38383838u + kt, 0x38383838u);
#else
            const uint4 v4 = wb[SCALE_WORD(ln0 + i, r, sc)];
#endif
            scw[4 * sc] = v4.x; scw[4 * sc + 1] = v4.y; scw[4 * sc + 2] = v4.z; scw[4 * sc + 3] = v4.w;
          }
#endif
          const uint ln = ln0 + i;                                       // (LANE_OFF is a function of ln)
#pragma clang loop unroll(full)
          for (uint ch = 0; ch < ((CT < WPW) ? 1u : (WPW / 16u)); ch++) {
            const uint g = (LANE_OFF + j * WPW + e0 + 16u * ch) / SCALE_GROUP;
            scv[i * (WPW / 16u) + ch] = decode_scale(scw + SCALE_UOFF, SCALE_SOFF(ln) + g);
#if SCALE_BIAS
            bv[i * (WPW / 16u) + ch] = decode_bias(scw + SCALE_UOFF, SCALE_SOFF(ln) + g);
#endif
          }
        }
#endif
#pragma clang loop unroll(full)
        for (uint jump = 0; jump < TK / 16u; jump++)
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) {
            float v = wv[4u * jump + qq];
#if SCALE_GROUP > 0
            v *= scv[jump / 4u];
#if SCALE_BIAS
            v += bv[jump / 4u];
#endif
#endif
            bT[uint16_t(((jump * NS_B + s) << 2) | qq)] = bfloat(v);
          }
      }
#if EXP_MODE == 2
      }
#endif
      // the activation slice of this tile: x' is in pack order (word j major, lane minor), whatever the loop order
      const uint kp = j * (32u / LPT) + q;
#if EXP_MODE == 6
      auto sA = tA.slice<int(TK), int(TM)>(0, 0);                         // a pinned slice: the A traffic without its bytes
#else
      auto sA = tA.slice<int(TK), int(TM)>(int(kp * TK), 0);
#endif
#if EXP_MODE != 1
      op.run(sA, bT, cT);
#else
      if (kt == KT - 1u) op.run(sA, bT, cT);                          // fill-only timing: one run so the fill is not dead
#endif
    }
#if KSPLIT > 1
    // the slices' partial tiles meet in threadgroup memory: slices 1 … KSPLIT-1 write theirs, slice 0 adds them into
    // its own (the same element order in every SIMD-group) and runs the epilogue alone; the second barrier keeps the
    // next tile's writes behind this tile's reads. Every thread of the threadgroup reaches both barriers.
    if (slice != 0u) {
#pragma clang loop unroll(full)
      for (uint16_t i = 0; i < C_CAP; i++) part[slice - 1u][lane][i] = cT[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice == 0u) {
#pragma clang loop unroll(full)
      for (uint s2 = 1; s2 < KSPLIT; s2++)
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < C_CAP; i++) cT[i] += part[s2 - 1u][lane][i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (slice != 0u) continue;
#endif
    // epilogue over the destination: element ((blk*(TN/16) + jump)*2 + s2) << 2 | q holds row n = c0b + 16*jump + q
    // and token m = 16*blk + c1b + 8*s2; a lane's 4 rows lie in one pack block, the block's other rows in the lanes
    // differing in bit 0 (and bit 3 when R = 16); rows beyond t_active and the range are not written
#pragma clang loop unroll(full)
    for (uint jump = 0; jump < TN / 16u; jump++) {
      const uint n = c0b + 16u * jump;                                   // this lane's 4 consecutive rows of the tile
      const uint row = tile * TN + n;                                    // the slab row (weights, row scales)
      const uint rrow = (tile - p.tile0) * TN + n;                       // the range-relative row (outputs)
      const uint bb = (tile - p.tile0) * (TN / R) + (n / R);             // the range-relative pack block
      float rs[4];
#pragma clang loop unroll(full)
      for (uint qq = 0; qq < 4u; qq++) rs[qq] = (rrow + qq < p.n_rows) ? row_scale[row + qq] * p.out_scale : 0.0f;
#pragma clang loop unroll(full)
      for (uint blk = 0; blk < NB_C; blk++)
#pragma clang loop unroll(full)
        for (uint s2 = 0; s2 < 2u; s2++) {
          const uint m = 16u * blk + c1b + 8u * s2;
          float v[4];
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) v[qq] = cT[uint16_t((((blk * (TN / 16u) + jump) * 2u + s2) << 2) | qq)] * rs[qq];
#if EPILOGUE == 2
          float pv[4];                                                   // the partner rows: up for a gate lane, gate for an up lane
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) pv[qq] = simd_shuffle_xor(v[qq], ushort(PAIR_XOR));
          const bool gate_lane = (lane & PAIR_XOR) == 0u;               // rows [0, CHUNK) of the block
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) v[qq] = silu_f(v[qq]) * pv[qq];
          const uint orow0 = bb * CHUNK + (n % R), n_out = p.n_rows / 2u;
          const bool writer = gate_lane && m < T_act;
#else
          const uint orow0 = rrow, n_out = p.n_rows;
          const bool writer = m < T_act;
#endif
          float ssq = 0.0f;
#pragma clang loop unroll(full)
          for (uint qq = 0; qq < 4u; qq++) {
            if (!writer || rrow + qq >= p.n_rows) continue;
            float vv = v[qq];
#if EPILOGUE == 1
#if EPILOGUE_ROUND
            vv = round_bf16(vv);
#endif
            vv += as_type<float>(uint(residual[(ulong)m * n_out + orow0 + qq]) << 16);
#endif
            const float vr = round_bf16(vv);
            ssq = fma(vr, vr, ssq);
#if PERM_OUT
            y[(ulong)m * PERM_K + perm_dest(orow0 + qq)] = ushort(as_type<uint>(vr) >> 16);   // the consumer tile's x' (its K = n_out)
#elif OUT_BF16
            y[(ulong)m * n_out + orow0 + qq] = ushort(as_type<uint>(vr) >> 16);
#else
            y[(ulong)m * n_out + orow0 + qq] = vv;
#endif
          }
#if STAT_OUT
          // the block's sum over its rows: the lanes differing in bit 0 (and bit 3 when a block is 16 rows)
          ssq += simd_shuffle_xor(ssq, ushort(1));
#if R == 16
          ssq += simd_shuffle_xor(ssq, ushort(8));
#endif
          if (m < T_act && (lane & ((R == 16u) ? 9u : 1u)) == 0u && rrow < p.n_rows) stat_out[m * p.n_blocks + bb] = ssq;
#endif
        }
    }
  }
}

// x_permute: x [T, K] BF16 (rows t >= T_act ignored) -> x' [TM, K] in gemm_tile's reduction order, zero beyond
// T_act; with PERM_NORM the RMSNorm scaling is applied on the way (x'[t][i] = bf16(h[t][c(i)] · r[t] · norm_w[c(i)]),
// r[t] = rsqrt(Σ stat[t·stat_parts ..] / K + eps) — the normalized activation the reference's norm produces).
// Pack order first (word j of lane l -> columns [j*32*WPW + l*WPW, +WPW)), then inside each TK-column tile the
// accelerator's slot order: slot 4*(bit0 + 2*bit3) + 16*jump + q of quad member (bit0, bit3) holds tile column
// (TK/4)*(bit0 + 2*bit3) + 4*jump + q, so a thread's TK/4 slots are TK/4 consecutive pack columns. PERM_SG
// SIMD-groups per output row, each a K/PERM_SG slice, the strides compile-time (K, TK, WPW) and the gathers issued
// PERM_UNROLL at a time — a step runs one of these per GEMV input, so its latency is the whole cost.
// With STEP_STATE the same per-T predicate as the tile it feeds.
#ifndef PERM_NORM
#define PERM_NORM 0
#endif
#ifndef PERM_SG
#define PERM_SG 4u
#endif
#define PERM_UNROLL 4u
struct XPermParams { uint k; uint t_active; uint tm; uint wpw; uint tk; uint stat_parts; float eps; uint pad; };

static inline uint perm_source(uint i) {                                  // the original column of slot i
  const uint kt = i / TK, slot = i % TK;
  const uint mq = ((slot >> 2) & 1u) + 2u * ((slot >> 3) & 1u), jump = slot >> 4, qq = slot & 3u;
  const uint c = kt * TK + (TK / 4u) * mq + 4u * jump + qq;               // the pack-order column
  const uint span = 32u * WPW;
  const uint j = c / span, rem = c % span, l = rem / WPW, e = rem % WPW;
  return l * KL + j * WPW + e;
}

kernel void x_permute(device const ushort* x [[buffer(0)]],
#if PERM_NORM
                      device const float* stat [[buffer(1)]], device const float* norm_w [[buffer(2)]],
#endif
                      device ushort* xp [[buffer(3)]], constant XPermParams& p [[buffer(4)]],
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / PERM_SG, slice = sg % PERM_SG;
  if (t >= p.tm) return;
#if STEP_STATE
  if (st->done) return;
  const uint T_act = (T_SRC == 1) ? st->n_inject : ((T_SRC == 2) ? T_STATIC_ROWS : st->t_this_step);
#ifdef T_HI
  if (T_act > T_HI || T_act <= T_LO) return;
#endif
#else
  const uint T_act = p.t_active;
#endif
  const uint k0 = slice * (K / PERM_SG), k1 = k0 + K / PERM_SG;          // this SIMD-group's slice of the row
  device ushort* out = xp + (ulong)t * K;
  if (t >= T_act) {
    for (uint i = k0 + lane; i < k1; i += 32u) out[i] = 0;
    return;
  }
#if PERM_NORM
  float ssq = 0.0f;
  for (uint i = lane; i < p.stat_parts; i += 32u) ssq += stat[t * p.stat_parts + i];
  ssq = simd_sum(ssq);
  const float r = rsqrt(ssq / float(K) + p.eps);
#endif
  device const ushort* row = x + (ulong)t * K;
  for (uint i = k0 + lane; i < k1; i += 32u * PERM_UNROLL) {
    uint src[PERM_UNROLL];
    ushort v[PERM_UNROLL];
#pragma clang loop unroll(full)
    for (uint u = 0; u < PERM_UNROLL; u++) {                                // independent gathers (a slice may be shorter than the unroll)
      src[u] = (i + 32u * u < k1) ? perm_source(i + 32u * u) : 0u;
      v[u] = (i + 32u * u < k1) ? row[src[u]] : ushort(0);
    }
#pragma clang loop unroll(full)
    for (uint u = 0; u < PERM_UNROLL; u++) {
      if (i + 32u * u >= k1) continue;
#if PERM_NORM
      const float f = round_bf16(as_type<float>(uint(v[u]) << 16) * r * norm_w[src[u]]);
      v[u] = ushort(as_type<uint>(f) >> 16);
#endif
      out[i + 32u * u] = v[u];
    }
  }
}

// coop_layout: the register layout the fill assumes, read back from the API for the test (one SIMD-group).
kernel void coop_layout(device int* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  matmul2d<desc, execution_simdgroup> op;
  auto bT = op.get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto cT = op.get_destination_cooperative_tensor<tA_t, decltype(bT), float>();
  device int* o = out + lane * (4 + 3 * 1024);                       // [cap_b, cap_c, -, -, then (valid, c0, c1) per element]
  o[0] = int(bT.get_capacity()); o[1] = int(cT.get_capacity());
  int q = 4;
  for (uint16_t i = 0; i < bT.get_capacity() && i < 512; i++) { auto c = bT.get_multidimensional_index(i); o[q++] = bT.is_valid_element(i); o[q++] = c[0]; o[q++] = c[1]; }
  for (uint16_t i = 0; i < cT.get_capacity() && i < 512; i++) { auto c = cT.get_multidimensional_index(i); o[q++] = cT.is_valid_element(i); o[q++] = c[0]; o[q++] = c[1]; }
}
