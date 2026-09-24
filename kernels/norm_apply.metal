// norm_apply: x[t][k] = bf16(h[t][k] · r[t] · norm_w[k]), r[t] = rsqrt(Σ stat[t·stat_parts ..] / K + eps) — the RMSNorm
// scaling as its own dispatch, writing the BF16 normalized activation the reference's norm produces. The default
// over gemv_T's NORM=1: re-scaling the chunk per row group costs 5–19 % of an ALU-bound GEMV while this dispatch
// costs ~2 µs (docs/research/gemv-kernel-study.md §3d). One SIMD-group per token; K % 8 == 0.
struct NormApplyParams { uint k; uint t_active; uint stat_parts; float eps; };

static inline float bf16lo(uint u) { return as_type<float>(u << 16); }
static inline float bf16hi(uint u) { return as_type<float>(u & 0xFFFF0000u); }
static inline uint pack_bf16x2(float lo, float hi) {
  uint a = as_type<uint>(lo); a += 0x7FFFu + ((a >> 16) & 1u);
  uint b = as_type<uint>(hi); b += 0x7FFFu + ((b >> 16) & 1u);
  return (a >> 16) | (b & 0xFFFF0000u);
}

kernel void norm_apply(device const ushort* h [[buffer(0)]], device const float* stat [[buffer(1)]], device const float* norm_w [[buffer(2)]],
                       device ushort* x [[buffer(3)]], constant NormApplyParams& p [[buffer(4)]],
                       uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint t = gid / sw;
  if (t >= p.t_active) return;
  float ssq = 0.0f;                                        // the partials summed lane-parallel, then one simd_sum:
  for (uint i = lane; i < p.stat_parts; i += 32u) ssq += stat[t * p.stat_parts + i];   // a serial loop costs a
  ssq = simd_sum(ssq);                                     // load latency per partial (~2 µs for 64 partials)
  const float r = rsqrt(ssq / float(p.k) + p.eps);
  device const uint4* row = (device const uint4*)(h + (ulong)t * p.k);
  device uint4* out = (device uint4*)(x + (ulong)t * p.k);
  for (uint j = lane; j < p.k / 8u; j += 32u) {
    uint4 q = row[j];
    float4 n0 = *(device const float4*)(norm_w + 8u * j), n1 = *(device const float4*)(norm_w + 8u * j + 4u);
    q.x = pack_bf16x2(bf16lo(q.x) * r * n0.x, bf16hi(q.x) * r * n0.y);
    q.y = pack_bf16x2(bf16lo(q.y) * r * n0.z, bf16hi(q.y) * r * n0.w);
    q.z = pack_bf16x2(bf16lo(q.z) * r * n1.x, bf16hi(q.z) * r * n1.y);
    q.w = pack_bf16x2(bf16lo(q.w) * r * n1.z, bf16hi(q.w) * r * n1.w);
    out[j] = q;
  }
}
