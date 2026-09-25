// gdn_mixer + gdn_norm: the Gated-DeltaNet mixer for T tokens (design §5.6; issue #22), in the reference's order and
// rounding (transformers' modeling_qwen3_5.py: causal_conv1d → l2norm q/k → σ(b), −exp(A_log)·softplus(a + dt_bias)
// → torch_recurrent_gated_delta_rule → RMSNormGated).
//
// Block = (value head h, group of SPB state-column slices of SL columns); a SIMD-group takes static slices of the
// Hv × (DV / (SL·SPB)) blocks. The recurrence is column-separable, so a block owns its columns' state outright and
// writes its columns of the read-out o (FP32) to a workspace; gdn_norm (one SIMD-group per (token, head)) applies
// the gated RMSNorm over the whole head afterwards. Lane ℓ owns k-rows {ℓ, ℓ+32, …} of the [DK, DV] FP32 state and
// channels {ℓ, ℓ+32, …} of the head's q, k (its key head = h / (Hv/Hk), HF's repeat_interleave) and v.
//   1. conv + SiLU for the lane's channels over the window [conv_state | x_0 .. x_{T-1}] (FP32 taps of the BF16
//      weights, BF16 rounding, SiLU in FP32 → BF16); the last CW-1 inputs become the new conv state (q/k channels
//      are written by the first value head of the key head, v channels by their own head);
//   2. per token: β = bf16(σ(b)), g = −exp(A_log)·softplus(a + dt_bias) in FP32, q/k L2-normalized over DK
//      (rsqrt(Σx² + 1e-6)), q /= √DK;
//   3. the recurrence is column-separable, so the head is processed in DV/SL column slices with the slice's state
//      (KR × SL floats per lane) in registers: S ← S·e^g; kv = kᵀS (simd_sum per column); Δ = (v − kv)·β;
//      S += k ⊗ Δ; o = qᵀS; S is stored back once per slice. Tokens go TP at a time (registers), each pass reads
//      and writes the state once; the block's o columns go to the workspace o_part [T][Hv·DV];
//   4. gdn_norm, per token and head over DV: o → BF16, rstd = rsqrt(mean(o²) + eps),
//      y = bf16(bf16(bf16(o·rstd)·w)·silu(z)).
// Macros: DK, DV (multiples of 32), CW (conv width), SL (columns per state slice), SPB (slices per block), TP (tokens
// per pass). The conv state is written by the block with slice group 0 of each head (q/k channels only by the
// first value head of the key head).
//
// State slots (SLOTS=2, needs STEP_STATE): the two states are double-buffered by step parity — the step's pass reads
// slot (step & 1) and writes the other, so the writer of a step never aliases the window its readers replay (with
// one slot, a fast head could overwrite the conv window a slower block of the same head is still reading). In a
// speculative program (design §5.8) the same kernel with COMMIT=1 runs after the accept scan (which advanced
// `step`): it recomputes the recurrence for the committed n_inject tokens from the slot the step's pass read and
// overwrites the slot the pass wrote — the rejected positions never reach the state. Plain programs keep SLOTS=1.
#ifndef SL
#define SL 8u
#endif
#ifndef SPB
#define SPB 1u
#endif
#ifndef TP
#define TP 4u
#endif
#ifndef STEP_STATE
#define STEP_STATE 0
#endif
#ifndef SLOTS
#define SLOTS 1u
#endif
#ifndef COMMIT
#define COMMIT 0
#endif
#if (SLOTS == 2u || COMMIT) && !STEP_STATE
#error "state slots and the commit pass read StepState"
#endif
#define KR (DK / 32u)
#define VR (DV / 32u)
#define NSL (DV / SL)
#define NSG (NSL / SPB)

struct GdnParams {
  uint hv; uint hk; uint t_active; uint q_off;
  uint k_off; uint v_off; uint z_off; uint a_off;
  uint b_off; uint in_stride; uint ab_stride; uint ab_separate;
  uint out_stride; uint n_sg; uint key_dim; uint pad0;
  float eps; float pad1; float pad2; float pad3;
};

static inline float bf16f(ushort u) { return as_type<float>(uint(u) << 16); }
static inline float round_bf16(float v) { uint u = as_type<uint>(v); u += 0x7FFFu + ((u >> 16) & 1u); return as_type<float>(u & 0xFFFF0000u); }
static inline ushort bf16bits(float v) { return ushort(as_type<uint>(round_bf16(v)) >> 16); }
static inline float silu_f(float x) { return x / (1.0f + exp(-x)); }
static inline float softplus_f(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// conv + SiLU of channel c for tokens t0 .. t0+n-1 (window = conv_state row, then the projection's inputs)
static inline void conv_channel(device const ushort* proj, uint in_stride, device const ushort* conv_state, device const ushort* conv_w,
                                uint c, uint t0, uint n, thread float* y) {
  float w[CW], win[CW - 1u];
  for (uint j = 0; j < CW; j++) w[j] = bf16f(conv_w[c * CW + j]);
  for (uint j = 0; j < CW - 1u; j++) win[j] = bf16f(conv_state[c * (CW - 1u) + j]);
  for (uint t = 0; t < t0 + n; t++) {                       // replay the step's inputs up to the pass's tokens
    const float x = bf16f(proj[t * in_stride + c]);
    if (t >= t0) {
      float acc = 0.0f;
      for (uint j = 0; j < CW - 1u; j++) acc = fma(w[j], win[j], acc);
      acc = fma(w[CW - 1u], x, acc);
      y[t - t0] = round_bf16(silu_f(round_bf16(acc)));
    }
    for (uint j = 0; j + 1u < CW - 1u; j++) win[j] = win[j + 1u];
    win[CW - 2u] = x;
  }
}

static inline void conv_state_update(device const ushort* proj, uint in_stride, device const ushort* src, device ushort* dst, uint c, uint T) {
  ushort win[CW - 1u];
  for (uint j = 0; j < CW - 1u; j++) win[j] = src[c * (CW - 1u) + j];
  for (uint t = 0; t < T; t++) {
    for (uint j = 0; j + 1u < CW - 1u; j++) win[j] = win[j + 1u];
    win[CW - 2u] = proj[t * in_stride + c];
  }
  for (uint j = 0; j < CW - 1u; j++) dst[c * (CW - 1u) + j] = win[j];
}

static inline float pick(thread const float* arr, uint i) {         // arr[i] with a compile-time-indexed body
  float r = arr[0];
  for (uint k = 1; k < VR; k++) if (i == k) r = arr[k];
  return r;
}

kernel void gdn_mixer(device const ushort* proj [[buffer(0)]], device const ushort* proj_ab [[buffer(1)]],
                      device ushort* conv_state [[buffer(2)]], device float* rec_state [[buffer(3)]],
                      device const ushort* conv_w [[buffer(4)]], device const float* neg_exp_a_log [[buffer(5)]],
                      device const float* dt_bias [[buffer(6)]], device float* o_part [[buffer(7)]],
                      constant GdnParams& p [[buffer(9)]],
#if STEP_STATE
                      device const StepState* st [[buffer(15)]],
#endif
                      uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint rep = p.hv / p.hk;
#if STEP_STATE
  if (st->done) return;
#if COMMIT
  const uint T = st->n_inject;                               // the committed tokens of the step that just ended
#else
  const uint T = st->t_this_step;
#endif
#else
  const uint T = p.t_active;
#endif
#if SLOTS == 2u
#if COMMIT
  const uint rd = (st->step + 1u) & 1u, wr = st->step & 1u;  // the accept scan advanced `step`: re-read the pass's input slot
#else
  const uint rd = st->step & 1u, wr = (st->step + 1u) & 1u;
#endif
  const ulong rec_stride = (ulong)p.hv * DK * DV;
  const uint conv_stride = (2u * p.key_dim + p.hv * DV) * (CW - 1u);
  device const ushort* conv_in = conv_state + rd * conv_stride;
  device ushort* conv_out = conv_state + wr * conv_stride;
  device const float* rec_in = rec_state + rd * rec_stride;
  device float* rec_out = rec_state + wr * rec_stride;
#else
  device const ushort* conv_in = conv_state;
  device ushort* conv_out = conv_state;
  device const float* rec_in = rec_state;
  device float* rec_out = rec_state;
#endif
  const uint n_blocks = p.hv * NSG;
  for (uint b = sg; b < n_blocks; b += p.n_sg) {
    const uint h = b / NSG, grp = b % NSG;
    const uint kh = h / rep;
    for (uint t0 = 0; t0 < T; t0 += TP) {
      const uint n = min(TP, T - t0);
      // 1. conv + SiLU of this lane's channels for the pass's tokens
      float qv[TP][KR], kv[TP][KR], vv[TP][VR];
      for (uint i = 0; i < KR; i++) {
        float y[TP];
        conv_channel(proj, p.in_stride, conv_in, conv_w, p.q_off + kh * DK + lane + 32u * i, t0, n, y);
        for (uint t = 0; t < TP; t++) qv[t][i] = y[t];
        conv_channel(proj, p.in_stride, conv_in, conv_w, p.k_off + kh * DK + lane + 32u * i, t0, n, y);
        for (uint t = 0; t < TP; t++) kv[t][i] = y[t];
      }
      for (uint i = 0; i < VR; i++) {
        float y[TP];
        conv_channel(proj, p.in_stride, conv_in, conv_w, p.v_off + h * DV + lane + 32u * i, t0, n, y);
        for (uint t = 0; t < TP; t++) vv[t][i] = y[t];
      }
      // 2. per-token scalars and the q/k L2 norms
      float beta[TP], eg[TP];
      for (uint t = 0; t < TP; t++) {
        if (t < n) {
          const uint tt = t0 + t;
          device const ushort* ab = p.ab_separate ? proj_ab : proj;
          const uint abs_ = p.ab_separate ? p.ab_stride : p.in_stride;
          const float a = bf16f(ab[tt * abs_ + p.a_off + h]), b = bf16f(ab[tt * abs_ + p.b_off + h]);
          beta[t] = round_bf16(1.0f / (1.0f + exp(-b)));
          eg[t] = exp(neg_exp_a_log[h] * softplus_f(a + dt_bias[h]));
          float sq = 0.0f, sk = 0.0f;
          for (uint i = 0; i < KR; i++) { sq = fma(qv[t][i], qv[t][i], sq); sk = fma(kv[t][i], kv[t][i], sk); }
          sq = simd_sum(sq); sk = simd_sum(sk);
          const float rq = rsqrt(sq + 1e-6f), rk = rsqrt(sk + 1e-6f);
          const float scale = sqrt(float(DK));
          for (uint i = 0; i < KR; i++) { qv[t][i] = (qv[t][i] * rq) / scale; kv[t][i] = kv[t][i] * rk; }
        } else {
          beta[t] = 0.0f; eg[t] = 1.0f;
          for (uint i = 0; i < KR; i++) { qv[t][i] = 0.0f; kv[t][i] = 0.0f; }
        }
      }
      // 3. the recurrence over this block's state slices
      for (uint s = grp * SPB; s < (grp + 1u) * SPB; s++) {
        float S[KR][SL];
        for (uint i = 0; i < KR; i++) {                        // the first pass reads the step's input slot, later passes the slot the block writes
          device const float* row = ((t0 == 0u) ? rec_in : (device const float*)rec_out) + ((ulong)(h * DK + lane + 32u * i)) * DV + s * SL;
          for (uint j = 0; j < SL; j++) S[i][j] = row[j];
        }
        for (uint t = 0; t < TP; t++) {
          if (t >= n) break;
          for (uint i = 0; i < KR; i++) for (uint j = 0; j < SL; j++) S[i][j] *= eg[t];
          float delta[SL];
          for (uint j = 0; j < SL; j++) {
            float part = 0.0f;
            for (uint i = 0; i < KR; i++) part = fma(S[i][j], kv[t][i], part);
            const float kvm = simd_sum(part);
            const uint v = s * SL + j;
            const float vt = simd_shuffle(pick(vv[t], v / 32u), ushort(v % 32u));
            delta[j] = (vt - kvm) * beta[t];
          }
          for (uint i = 0; i < KR; i++) for (uint j = 0; j < SL; j++) S[i][j] = fma(kv[t][i], delta[j], S[i][j]);
          for (uint j = 0; j < SL; j++) {
            float part = 0.0f;
            for (uint i = 0; i < KR; i++) part = fma(S[i][j], qv[t][i], part);
            const float o = simd_sum(part);
            if (lane == 0) o_part[(t0 + t) * p.out_stride + h * DV + s * SL + j] = o;
          }
        }
        for (uint i = 0; i < KR; i++) {
          device float* row = rec_out + ((ulong)(h * DK + lane + 32u * i)) * DV + s * SL;
          for (uint j = 0; j < SL; j++) row[j] = S[i][j];
        }
      }
    }
    // the new conv state: the last CW-1 inputs of the step (q/k channels once per key head, v channels per head),
    // written by the head's first slice group
    if (grp == 0u) {
      if (h % rep == 0u) {
        for (uint i = 0; i < KR; i++) {
          conv_state_update(proj, p.in_stride, conv_in, conv_out, p.q_off + kh * DK + lane + 32u * i, T);
          conv_state_update(proj, p.in_stride, conv_in, conv_out, p.k_off + kh * DK + lane + 32u * i, T);
        }
      }
      for (uint i = 0; i < VR; i++) conv_state_update(proj, p.in_stride, conv_in, conv_out, p.v_off + h * DV + lane + 32u * i, T);
    }
  }
}

kernel void gdn_norm(device const float* o_part [[buffer(0)]], device const ushort* proj [[buffer(1)]], device const float* norm_w [[buffer(2)]],
                     device ushort* out [[buffer(3)]], constant GdnParams& p [[buffer(4)]],
#if STEP_STATE
                     device const StepState* st [[buffer(15)]],
#endif
                     uint gid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]], uint sw [[threads_per_simdgroup]]) {
  const uint sg = gid / sw;
  const uint t = sg / p.hv, h = sg % p.hv;
#if STEP_STATE
  if (st->done || t >= st->t_this_step) return;
#else
  if (t >= p.t_active) return;
#endif
  float ob[VR], ss = 0.0f;
  for (uint i = 0; i < VR; i++) { ob[i] = round_bf16(o_part[t * p.out_stride + h * DV + lane + 32u * i]); ss = fma(ob[i], ob[i], ss); }
  ss = simd_sum(ss);
  const float rstd = rsqrt(ss / float(DV) + p.eps);
  for (uint i = 0; i < VR; i++) {
    const uint v = lane + 32u * i;
    float y = round_bf16(ob[i] * rstd);
    y = round_bf16(norm_w[v] * y);
    const float z = bf16f(proj[t * p.in_stride + p.z_off + h * DV + v]);
    y = round_bf16(y * silu_f(z));
    out[t * p.out_stride + h * DV + v] = bf16bits(y);
  }
}
