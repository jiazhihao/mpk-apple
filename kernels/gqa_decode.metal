// gqa_decode + gqa_merge: the full-attention mixer for T new tokens (design §5.6; issue #21).
//
// Block = (kv head j, chunk c of CH key positions, row group of RBMAX query rows); a SIMD-group takes static
// slices of the kv_heads × n_chunks × n_row_groups blocks, n_chunks = ceil((position + T) / CH) computed in-kernel
// so the work grows with the context under the fixed crew geometry. Lane ℓ owns dims [ℓ·D/32, (ℓ+1)·D/32) of
// every vector. For its block a SIMD-group:
//   1. prologue — its query rows of kv head j (q head h = j·rep + i, token t; row = t·rep + i):
//      per-head RMSNorm (FP32 Σq², rsqrt(mean + eps), · (1 + w) → BF16) then RoPE in the load-time
//      head-dim permutation (pairs (i, i + D/2) = lane ℓ and lane ℓ ^ 16; cos = 1 / sin = 0 outside the rotary
//      dims), each product and the sum rounded to BF16 like the reference's elementwise BF16 math;
//   2. keys — positions < position come from the cache; the T new positions are normed + RoPE'd from the
//      projection (the block owning the chunk that holds them also appends k and v to the caches: writer and
//      readers derive them from the same input, so no read-after-write inside the dispatch);
//      scores s = bf16(bf16(q·k) · scaling) (the reference's BF16 matmul and scaling), causal inside the step;
//   3. per chunk: m_c = max s, p̃ = bf16(exp(s − m_c)) rounded like the reference's P, d_c = Σ exp(s − m_c) in
//      FP32, o_c = Σ p̃·v in FP32; the (o_c, m_c, d_c) partials go to a workspace.
// gqa_merge (one SIMD-group per (token, q head)) folds the chunks in order — deterministic — normalizes, rounds to
// BF16, multiplies by bf16(σ(gate)) (the reference's `attn · sigmoid(gate)` in BF16) and writes [T, H·D].
//
// Macros: D (head dim, multiple of 32), CH (keys per chunk), RBMAX (query rows per pass; rows beyond re-stream the
// chunk). Params carry the projection's column offsets (q | gate | k | v), the strides, position and T.
#ifndef CH
#define CH 64u
#endif
#ifndef RBMAX
#define RBMAX 8u
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

kernel void gqa_decode(device const ushort* qkvg [[buffer(0)]], device ushort* k_cache [[buffer(1)]], device ushort* v_cache [[buffer(2)]],
                       device const ushort* cos_t [[buffer(3)]], device const ushort* sin_t [[buffer(4)]],
                       device const float* q_norm [[buffer(5)]], device const float* k_norm [[buffer(6)]],
                       device float* part_o [[buffer(7)]], device float* part_md [[buffer(8)]], constant GqaParams& p [[buffer(9)]],
                       uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint rep = p.heads / p.kv_heads;
  const uint T = p.t_active;
  const uint rows = rep * T;
  const uint n_rg = (rows + RBMAX - 1u) / RBMAX;
  const uint ctx = p.position + T;
  const uint n_chunks = (ctx + CH - 1u) / CH;
  const uint n_blocks = p.kv_heads * n_chunks * n_rg;          // block = (kv head, chunk, row group)
  for (uint b = sg; b < n_blocks; b += p.n_sg) {
    const uint rg = b % n_rg, c = (b / n_rg) % n_chunks, j = b / (n_rg * n_chunks);
    const uint k0 = c * CH, k1 = min(k0 + CH, ctx);
    const uint r0 = rg * RBMAX;
    const uint nr = min(RBMAX, rows - r0);
    // prologue: this row group's queries, normed and RoPE'd, DL dims per lane
    float q[RBMAX][DL];
    for (uint r = 0; r < RBMAX; r++) {
      if (r < nr) {
        const uint row = r0 + r, t = row / rep, h = j * rep + (row % rep);
        load_dl(qkvg + t * p.in_stride + p.q_off + h * D + lane * DL, q[r]);
        norm_rope(q[r], q_norm, cos_t + (p.position + t) * D, sin_t + (p.position + t) * D, p.eps, lane);
      } else {
        for (uint e = 0; e < DL; e++) q[r][e] = 0.0f;
      }
    }
    // pass 1: scores; key k0 + g*32 + kk lands on lane kk of group g (compile-time g, r)
    float s_keep[CH / 32u][RBMAX];
    float m_c[RBMAX];
    for (uint r = 0; r < RBMAX; r++) m_c[r] = -INFINITY;
    for (uint g = 0; g < CH / 32u; g++) {
      for (uint r = 0; r < RBMAX; r++) s_keep[g][r] = -INFINITY;
      for (uint kk = 0; kk < 32u; kk++) {
        const uint key = k0 + g * 32u + kk;
        if (key >= k1) break;
        float kf[DL];
        if (key < p.position) {
          load_dl(k_cache + (key * p.kv_heads + j) * D + lane * DL, kf);
        } else {
          const uint tk = key - p.position;
          load_dl(qkvg + tk * p.in_stride + p.k_off + j * D + lane * DL, kf);
          norm_rope(kf, k_norm, cos_t + key * D, sin_t + key * D, p.eps, lane);
          if (rg == 0) {                                   // append k (normed, RoPE'd) and v to the caches once
            for (uint e = 0; e < DL; e++) k_cache[(key * p.kv_heads + j) * D + lane * DL + e] = bf16bits(kf[e]);
            for (uint e = 0; e < DL; e++) v_cache[(key * p.kv_heads + j) * D + lane * DL + e] = qkvg[tk * p.in_stride + p.v_off + j * D + lane * DL + e];
          }
        }
        for (uint r = 0; r < RBMAX; r++) {
          if (r < nr) {
            const uint t = (r0 + r) / rep;
            float dot = 0.0f;
            for (uint e = 0; e < DL; e++) dot = fma(q[r][e], kf[e], dot);
            dot = simd_sum(dot);
            const float sc = (key > p.position + t) ? -INFINITY : round_bf16(round_bf16(dot) * p.scaling);
            if (lane == kk) s_keep[g][r] = sc;
            m_c[r] = max(m_c[r], sc);
          }
        }
      }
    }
    // pass 2: p̃ = bf16(exp(s - m_c)), d in FP32 from the unrounded p, o = Σ p̃ v (lane-per-dim, p broadcast)
    float d_c[RBMAX], o[RBMAX][DL];
    for (uint r = 0; r < RBMAX; r++) { d_c[r] = 0.0f; for (uint e = 0; e < DL; e++) o[r][e] = 0.0f; }
    for (uint g = 0; g < CH / 32u; g++) {
      for (uint kk = 0; kk < 32u; kk++) {
        const uint key = k0 + g * 32u + kk;
        if (key >= k1) break;
        float vf[DL];
        if (key < p.position) load_dl(v_cache + (key * p.kv_heads + j) * D + lane * DL, vf);
        else load_dl(qkvg + (key - p.position) * p.in_stride + p.v_off + j * D + lane * DL, vf);
        for (uint r = 0; r < RBMAX; r++) {
          if (r < nr) {
            const float sc = simd_shuffle(s_keep[g][r], ushort(kk));
            const float pr = (sc == -INFINITY) ? 0.0f : exp(sc - m_c[r]);
            d_c[r] += pr;
            const float pb = round_bf16(pr);
            for (uint e = 0; e < DL; e++) o[r][e] = fma(pb, vf[e], o[r][e]);
          }
        }
      }
    }
    for (uint r = 0; r < RBMAX; r++) {
      if (r < nr) {
        const uint base = (j * p.n_chunks_max + c) * p.rows_max + r0 + r;
        for (uint e = 0; e < DL; e++) part_o[base * D + lane * DL + e] = o[r][e];
        if (lane == 0) { part_md[base * 2u] = m_c[r]; part_md[base * 2u + 1u] = d_c[r]; }
      }
    }
  }
}

kernel void gqa_merge(device const float* part_o [[buffer(0)]], device const float* part_md [[buffer(1)]], device const ushort* qkvg [[buffer(2)]],
                      device ushort* out [[buffer(3)]], constant GqaParams& p [[buffer(4)]],
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / p.heads, h = sg % p.heads;
  if (t >= p.t_active) return;
  const uint rep = p.heads / p.kv_heads;
  const uint j = h / rep, row = t * rep + (h % rep);
  const uint n_chunks = (p.position + p.t_active + CH - 1u) / CH;
  float m_g = -INFINITY;
  for (uint c = 0; c < n_chunks; c++) m_g = max(m_g, part_md[((j * p.n_chunks_max + c) * p.rows_max + row) * 2u]);
  float d_g = 0.0f, o[DL];
  for (uint e = 0; e < DL; e++) o[e] = 0.0f;
  for (uint c = 0; c < n_chunks; c++) {
    const uint base = (j * p.n_chunks_max + c) * p.rows_max + row;
    const float m_c = part_md[base * 2u];
    if (m_c == -INFINITY) continue;
    const float w = exp(m_c - m_g);
    d_g = fma(part_md[base * 2u + 1u], w, d_g);
    for (uint e = 0; e < DL; e++) o[e] = fma(w, part_o[base * D + lane * DL + e], o[e]);
  }
  const float inv = d_g > 0.0f ? 1.0f / d_g : 0.0f;
  for (uint e = 0; e < DL; e++) {
    float y = round_bf16(o[e] * inv);
    if (p.has_gate) {
      const float g = bf16f(qkvg[t * p.in_stride + p.gate_off + h * D + lane * DL + e]);
      y = round_bf16(y * round_bf16(1.0f / (1.0f + exp(-g))));
    }
    out[t * p.out_stride + h * D + lane * DL + e] = bf16bits(y);
  }
}
