// Shared by the attention kernels (gqa_decode.metal v1 and gqa_decode_v2.metal): the params record, the BF16
// helpers, the per-head norm + RoPE of one row held lane-per-dim (D/32 dims per lane, rotary partner on lane ^ 16).
#ifndef STEP_STATE
#define STEP_STATE 0                 // 1: position and T come from the bound StepState (the step program); 0: from params
#endif
#ifndef DRAFT
#define DRAFT 0
#endif
#ifndef CH
#define CH 64u
#endif
#ifndef RBMAX
#define RBMAX 8u
#endif
#ifndef NSG
#define NSG 12u                      // SIMD-groups per threadgroup (the crew geometry)
#endif
#define DL (D / 32u)

struct GqaParams {
  uint heads; uint kv_heads; uint t_active; uint position;
  uint n_sg; uint q_off; uint gate_off; uint k_off;
  uint v_off; uint in_stride; uint out_stride; uint ctx_max;
  float eps; float scaling; uint has_gate; uint n_chunks_max;
  uint rows_max; uint pad0; uint pad1; uint pad2;
};

static inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline ushort bf16bits(float v) { return ushort(as_type<uint>(round_bf16(v)) >> 16); }

static inline void load_dl(device const ushort* p, thread float* f) {
  for (uint e = 0; e < DL; e++) f[e] = bf16f(p[e]);
}

// per-head RMSNorm (1 + w) and RoPE of one D-vector held DL-per-lane; the reference's rounding order
static inline void norm_rope(thread float* f, device const float* nw, device const ushort* cos_row, device const ushort* sin_row,
                             float eps, uint lane) {
  float ss = 0.0f;
  for (uint e = 0; e < DL; e++) ss = fma(f[e], f[e], ss);
  ss = simd_sum(ss);
  const float rstd = rsqrt(ss / float(D) + eps);
  for (uint e = 0; e < DL; e++) f[e] = round_bf16(f[e] * rstd * nw[lane * DL + e]);
  const bool lo = lane < 16u;
  float r[DL];
  for (uint e = 0; e < DL; e++) {
    const float partner = simd_shuffle_xor(f[e], 16u);
    const float c = bf16f(cos_row[lane * DL + e]), s = bf16f(sin_row[lane * DL + e]);
    const float a = round_bf16(f[e] * c), b = round_bf16(partner * s);
    r[e] = round_bf16(lo ? a - b : a + b);
  }
  for (uint e = 0; e < DL; e++) f[e] = r[e];
}

static inline float bf16lo(uint u) { return as_type<float>(u << 16); }
static inline float bf16hi(uint u) { return as_type<float>(u & 0xFFFF0000u); }

// the chunk (keys per SIMD-group partial) of the v2 kernels for a context: the finest (32) unless larger chunks
// still give every threadgroup a (kv head, batch of NSG chunks) block — both kernels compute it from the same inputs
static inline uint chunk_of(uint ctx, uint kv, uint n_tg) {
  uint ch = 32u;
  if (kv * ((ctx + NSG * 64u - 1u) / (NSG * 64u)) >= n_tg) ch = 64u;
  if (kv * ((ctx + NSG * 128u - 1u) / (NSG * 128u)) >= n_tg) ch = 128u;
  return ch;
}
