// gemv_T: y[t][row] = sum_k dequant(W[row][k]) * x[t][k] over a block-lane-major pack (design D8, §5.6).
//
// The Python side prepends `#include <metal_stdlib>` and a FORMAT SNIPPET (monolith.formats.<fmt>.msl_decode) that
// defines WEIGHTS_PER_WORD, SCALE_GROUP, decode_word(uint4, thread float*) and decode_scale(thread const uint*, uint),
// then sets the macros below. One kernel = one op of the step program: `n_sg` SIMD-groups take static slices of the
// `n_blocks` row blocks (block b -> rows [b*R, b*R+R)); the 32 lanes own 32 column stripes of K/32 columns; every
// weight load is one 16-byte word of the lane-row unit; activations are BF16 and are re-read per RG-row group from
// cached device memory; accumulation is FP32; one simd_sum per (row, token); the per-row scale (the source matrix's
// per-tensor scale, from the pack's row-scale table) is applied once per output.
//
// Macros: K (columns), R (rows per block), T (tokens), RG (rows per activation reuse group, divides R),
//         LANE_ORDER (0 contiguous, 1 interleaved16), UNIT_WORDS (16-byte words per lane-row unit),
//         PAYLOAD_WORDS (weight words per lane-row), SCALE_WORDS (uint4 words holding this row's scale bytes, 0 = none),
//         OUT_BF16 (1: write bf16 outputs, 0: float), X_PRECONVERT (1: convert the activation chunk to float once
//         per word and reuse it across the row group; 0: keep it as bf16 words and convert per row — for large T*WPW)
//
// Fusions (design §5.1, §5.6), each a macro so a pipeline variant is selected by (format, T, fusions):
//   NORM=1      the input is the raw residual stream h: x[t][k] = bf16(h[t][k] · r[t] · norm_w[k]) with
//               r[t] = rsqrt(Σ stat[t·stat_parts .. +stat_parts) / K + eps) — the RMSNorm scaling applied on load,
//               its statistic read from a stat buffer (1 part from rmsnorm_stat, or the producing op's per-block
//               partial sums of squares, stat_parts = its n_blocks). Rounded to BF16 like the reference's norm output.
//   EPILOGUE=1  residual add before the single rounding: y = bf16(acc·scale + residual[t][row]).
//   EPILOGUE=2  silu·mul over chunk-interleaved rows: block b holds gate rows [b·R, b·R+CHUNK) and up rows
//               [b·R+CHUNK, b·R+R) (pack_weights' interleave_chunks with chunk = CHUNK = R/2); output
//               y[t][b·CHUNK + i] = bf16(silu(gate_i) · up_i) over n_rows/2 outputs; RG must divide CHUNK.
//   STAT_OUT=1  the epilogue also writes per-block partial sums of squares of the BF16-rounded outputs,
//               stat_out[t·n_blocks + b], for the next op's NORM (the hoisted norm statistic).
#ifndef NORM
#define NORM 0
#endif
#ifndef EPILOGUE
#define EPILOGUE 0
#endif
#ifndef STAT_OUT
#define STAT_OUT 0
#endif
#ifndef CHUNK
#define CHUNK (R / 2u)
#endif
#ifndef RG
#define RG 4
#endif
#ifndef OUT_BF16
#define OUT_BF16 0
#endif
#ifndef X_PRECONVERT
#define X_PRECONVERT 1
#endif
#ifndef T_STATIC
#define T_STATIC 0
#endif
#ifndef STEP_STATE
#define STEP_STATE 0                 // 1: T comes from the bound StepState (buffer 15) and the kernel returns at once after `done`
#endif
#define KL (K / 32u)                                   // columns per lane
#define WPW WEIGHTS_PER_WORD
#define XW (WPW / 8u)                                  // uint4 words of bf16 activations per weight word
#if SCALE_GROUP > 0
#define GPW ((WPW >= SCALE_GROUP) ? (WPW / SCALE_GROUP) : 1u)     // scale groups (or fraction thereof) per word
#define WPG ((WPW >= SCALE_GROUP) ? SCALE_GROUP : WPW)            // weights per in-word group
#else
#define GPW 1u
#define WPG WPW
#endif

struct GemvParams { uint n_rows; uint n_blocks; uint n_sg; uint t_active; float out_scale; float eps; uint stat_parts; uint pad; };

static inline uint unit_word(uint lane, uint r, uint j) {
#if LANE_ORDER == 0
  return (lane * R + r) * UNIT_WORDS + j;
#else
  return (r * UNIT_WORDS + j) * 32u + lane;
#endif
}

static inline float bf16lo(uint u) { return as_type<float>(u << 16); }
static inline float bf16hi(uint u) { return as_type<float>(u & 0xFFFF0000u); }
static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline uint pack_bf16x2(float lo, float hi) {
  uint a = as_type<uint>(lo); a += 0x7FFFu + ((a >> 16) & 1u);
  uint b = as_type<uint>(hi); b += 0x7FFFu + ((b >> 16) & 1u);
  return (a >> 16) | (b & 0xFFFF0000u);
}
static inline float silu_f(float g) { return g / (1.0f + exp(-g)); }

kernel void gemv_T(device const uint4* w [[buffer(0)]], device const float* row_scale [[buffer(1)]],
                   device const ushort* x [[buffer(2)]],
#if OUT_BF16
                   device ushort* y [[buffer(3)]],
#else
                   device float* y [[buffer(3)]],
#endif
                   constant GemvParams& p [[buffer(4)]],
#if NORM
                   device const float* stat [[buffer(5)]], device const float* norm_w [[buffer(6)]],
#endif
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
#if STEP_STATE
  if (st->done) return;
  const uint T_act = st->t_this_step;          // dynamic T (≤ the compiled T), design §5.7
#elif T_STATIC
  const uint T_act = T;                        // compile-time T (plain decode programs, benches)
#else
  const uint T_act = p.t_active;               // <= T; tokens beyond it are skipped
#endif
#if NORM
  float rn[T];                                 // the per-token RMSNorm scale, from the statistic's partial sums
  for (uint t = 0; t < T; t++) {
    float ssq = 0.0f;
    if (t < T_act) for (uint i = lane; i < p.stat_parts; i += 32u) ssq += stat[t * p.stat_parts + i];
    ssq = simd_sum(ssq);                       // lane-parallel: a serial loop pays a load latency per partial
    rn[t] = rsqrt(ssq / float(K) + p.eps);
  }
#endif
  for (uint b = sg; b < p.n_blocks; b += p.n_sg) {
    device const uint4* wb = w + (ulong)b * (R * 32u * UNIT_WORDS);
#if STAT_OUT
    float ssq_out[T];
    for (uint t = 0; t < T; t++) ssq_out[t] = 0.0f;
#endif
#if EPILOGUE == 2
    float gate_v[CHUNK][T];
#endif
    for (uint r0 = 0; r0 < R; r0 += RG) {
      float acc[RG][T];
      for (uint i = 0; i < RG; i++) for (uint t = 0; t < T; t++) acc[i][t] = 0.0f;
#if SCALE_GROUP > 0
      uint scw[RG][SCALE_WORDS * 4];
      for (uint i = 0; i < RG; i++) for (uint s = 0; s < SCALE_WORDS; s++) {
        uint4 q = wb[unit_word(lane, r0 + i, PAYLOAD_WORDS + s)];
        scw[i][4 * s] = q.x; scw[i][4 * s + 1] = q.y; scw[i][4 * s + 2] = q.z; scw[i][4 * s + 3] = q.w;
      }
#endif
      for (uint j = 0; j < PAYLOAD_WORDS; j++) {
        const uint col = lane * KL + j * WPW;
#if X_PRECONVERT
        // convert the activation chunk once per word and reuse it across the RG rows (T*WPW floats of registers)
        float xf[T][WPW];
#if NORM
        float nwv[WPW];
        for (uint e = 0; e < WPW; e += 4) { float4 q = *(device const float4*)(norm_w + col + e); nwv[e] = q.x; nwv[e + 1] = q.y; nwv[e + 2] = q.z; nwv[e + 3] = q.w; }
#endif
        for (uint t = 0; t < T; t++) {
          if (t < T_act) {
            device const uint4* xp = (device const uint4*)(x + t * K + col);
            for (uint v = 0; v < XW; v++) { uint4 q = xp[v];
              xf[t][8 * v] = bf16lo(q.x); xf[t][8 * v + 1] = bf16hi(q.x); xf[t][8 * v + 2] = bf16lo(q.y); xf[t][8 * v + 3] = bf16hi(q.y);
              xf[t][8 * v + 4] = bf16lo(q.z); xf[t][8 * v + 5] = bf16hi(q.z); xf[t][8 * v + 6] = bf16lo(q.w); xf[t][8 * v + 7] = bf16hi(q.w); }
#if NORM
            for (uint e = 0; e < WPW; e++) xf[t][e] = round_bf16(xf[t][e] * rn[t] * nwv[e]);
#endif
          } else { for (uint e = 0; e < WPW; e++) xf[t][e] = 0.0f; }
        }
#else
        uint4 xq[T][XW];
        for (uint t = 0; t < T; t++) {
          if (t < T_act) { device const uint4* xp = (device const uint4*)(x + t * K + col); for (uint v = 0; v < XW; v++) xq[t][v] = xp[v]; }
          else { for (uint v = 0; v < XW; v++) xq[t][v] = uint4(0); }
        }
#if NORM
        for (uint v = 0; v < XW; v++) {
          float4 n0 = *(device const float4*)(norm_w + col + 8 * v), n1 = *(device const float4*)(norm_w + col + 8 * v + 4);
          for (uint t = 0; t < T; t++) {
            if (t >= T_act) continue;
            uint4 q = xq[t][v];
            q.x = pack_bf16x2(bf16lo(q.x) * rn[t] * n0.x, bf16hi(q.x) * rn[t] * n0.y);
            q.y = pack_bf16x2(bf16lo(q.y) * rn[t] * n0.z, bf16hi(q.y) * rn[t] * n0.w);
            q.z = pack_bf16x2(bf16lo(q.z) * rn[t] * n1.x, bf16hi(q.z) * rn[t] * n1.y);
            q.w = pack_bf16x2(bf16lo(q.w) * rn[t] * n1.z, bf16hi(q.w) * rn[t] * n1.w);
            xq[t][v] = q;
          }
        }
#endif
#endif
        for (uint i = 0; i < RG; i++) {
          uint4 q = wb[unit_word(lane, r0 + i, j)];
          float wv[WPW];
          decode_word(q, wv);
          for (uint t = 0; t < T; t++) {
#if !X_PRECONVERT
            uint xw[XW * 4];
            for (uint v = 0; v < XW; v++) { xw[4 * v] = xq[t][v].x; xw[4 * v + 1] = xq[t][v].y; xw[4 * v + 2] = xq[t][v].z; xw[4 * v + 3] = xq[t][v].w; }
#endif
            for (uint g = 0; g < GPW; g++) {
              float part = 0.0f;
              for (uint e = 0; e < WPG; e++) {
                const uint ee = g * WPG + e;
#if X_PRECONVERT
                const float xv = xf[t][ee];
#else
                const float xv = (ee & 1u) ? bf16hi(xw[ee >> 1]) : bf16lo(xw[ee >> 1]);
#endif
                part = fma(wv[ee], xv, part);
              }
#if SCALE_GROUP > 0
              const float s = decode_scale(scw[i], (j * WPW + g * WPG) / SCALE_GROUP);
              acc[i][t] = fma(part, s, acc[i][t]);
#else
              acc[i][t] += part;
#endif
            }
          }
        }
      }
      for (uint i = 0; i < RG; i++) {
        const uint r = r0 + i;
        const uint row = b * R + r;
        const float rs = (row < p.n_rows) ? row_scale[row] * p.out_scale : 0.0f;
        for (uint t = 0; t < T; t++) {
          float v = simd_sum(acc[i][t]) * rs;
#if EPILOGUE == 2
          if (r < CHUNK) { gate_v[r][t] = v; continue; }
          const uint orow = b * CHUNK + (r - CHUNK), n_out = p.n_rows / 2u;
          v = silu_f(gate_v[r - CHUNK][t]) * v;
#else
          const uint orow = row, n_out = p.n_rows;
#if EPILOGUE == 1
          if (orow < n_out && t < T_act) v += as_type<float>(uint(residual[t * n_out + orow]) << 16);
#endif
#endif
          if (orow < n_out && t < T_act) {
            const float vr = round_bf16(v);
#if STAT_OUT
            ssq_out[t] = fma(vr, vr, ssq_out[t]);
#endif
            if (lane == 0) {
#if OUT_BF16
              y[t * n_out + orow] = ushort(as_type<uint>(vr) >> 16);
#else
              y[t * n_out + orow] = v;
#endif
            }
          }
        }
      }
    }
#if STAT_OUT
    if (lane == 0) for (uint t = 0; t < T; t++) if (t < T_act) stat_out[t * p.n_blocks + b] = ssq_out[t];
#endif
  }
}
