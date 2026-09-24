# Decode kernels — the mixers on the M5 Pro

Companion of `gemv-kernel-study.md` for the non-GEMV ops of the step program (design §5.6): what each kernel does,
what it costs on the M5 Pro (20 cores, Apple10) and what the measurements say the next version must change. Evidence
tags as in the other reports: [M] measured here, [H] hypothesis.

## 1. `gqa_decode` + `gqa_merge` (#21) — `apple-m5-pro-20c_gqa.jsonl`

**Structure (v1).** Block = (kv head, chunk of `CH` key positions, group of `RBMAX` query rows); a SIMD-group takes
static slices of the `kv_heads × n_chunks × n_row_groups` blocks with `n_chunks = ceil((position + T) / CH)` computed
in-kernel, so the work grows with the context under the fixed crew geometry. Lane ℓ owns dims `[ℓ·D/32, (ℓ+1)·D/32)`.
Per block: the query rows are normed (`(1 + w)` RMSNorm) and RoPE'd in the load-time head-dim permutation (partner
dim on lane ℓ ^ 16, cos = 1 / sin = 0 outside the rotary dims); keys before `position` come from the cache and the
`T` new keys are normed + RoPE'd from the projection (the block owning their chunk also appends k and v to the
caches — writers and readers derive them from the same input, so no read-after-write inside the dispatch); scores
are `bf16(bf16(q·k)·scaling)` with the causal mask inside the step; per chunk `m_c = max s`, `p̃ = bf16(exp(s − m_c))`,
`d_c = Σ exp(s − m_c)` (FP32), `o_c = Σ p̃·v`; the partials go to a workspace and `gqa_merge` (one SIMD-group per
(token, q head)) folds the chunks in order, normalizes, rounds to BF16 and multiplies by `bf16(σ(gate))`. Results are
bit-identical across runs [M]. Against the HF-faithful layer oracle the output differs by 1–2 BF16 ULP (P is rounded
before normalization, per chunk, instead of after) [M]; against a numpy model of the kernel's own contract it is
within 2·10⁻³ of the largest output and the caches match to ≤ 1 ULP [M].

**Cost** [M], best of 10 after a 60 ms warm-up (two runs agree within 2 %), D = 256, chunk 64, 4 rows per pass:

| heads/kv | context | T=1 µs/layer | KV GB/s | T=4 µs/layer | KV GB/s |
|---|---|---|---|---|---|
| 32/4 | 1024 | 82 | 52 | 202 | 21 |
| 32/4 | 4096 | 245 | 68 | 688 | 24 |
| 32/4 | 8192 | 459 | 73 | 1431 | 24 |
| 32/4 | 32768 | 1755 | 76 | 5507 | 24 |
| 8/2 | 1024 | 82 | 26 | 81 | 26 |
| 8/2 | 4096 | 94 | 89 | 240 | 35 |
| 8/2 | 8192 | 182 | 92 | 424 | 40 |

GB/s counts the K and V bytes of the context once; at T = 4 the kernel re-streams them once per row group
(`rep·T / RBMAX` = 8 groups for 32/4 heads), so the useful figure is ~8× below the actual read rate.

**What it says.**

1. `RBMAX = 4` is the register budget: 8 rows per pass spill and run 13× slower (1.06 ms vs 82 µs at 1 K) [M]; 2 rows
   re-stream K/V twice as often and lose 25 % at 1 K [M]. `CH = 64` is the best chunk from 1 K to 32 K; 32 helps at
   ≤ 4 K by 2–5 % and 128 loses 10–45 % at short context (too few blocks) [M].
2. At T = 1 the kernel reaches 51 GB/s of KV at 1 K and 76 GB/s at 32 K of a ~300 GB/s bus: compute-bound on the
   per-(key, row) `simd_sum` and the BF16 roundings, not bandwidth-bound. For the 27B (16 attention layers, 32/4
   heads [H]: the config is not on this machine) that is 1.3 ms per token at 1 K and 4 ms at 4 K — 10–30 % of a
   ~12 ms token — and 28 ms at 32 K, where attention would dominate. Good enough for M4's end-to-end gates at short
   context; the long-context rows are the M5 work (#34).
3. T = 4 costs 2.5–3× T = 1 because every row group re-streams the chunk; at 32 K that is 5.5 ms per layer.
   Speculative verification (T = 1 + γ) therefore needs the v2 structure below before M6 measures its cost.
4. The 0.8B's 8/2 layers cost 82–182 µs at 1–8 K (6 layers: 0.5–1.1 ms per token) [M].

**v2 [H]** (M5, #34): one threadgroup per (kv head, batch of chunks) with the query rows of *all* tokens in
threadgroup memory as BF16 (32 rows × 512 B = 16 KB, shared by the 12 SIMD-groups), lane-per-key scoring (each lane
reads its own key's 512 B, no cross-lane reductions) and lane-per-dim P·V with `simd_shuffle` broadcast of p̃. That
removes the `simd_sum` per (key, row), reads K/V once per step regardless of T, and keeps the crew geometry; the
cost is a chunk-batch decomposition (parallelism at short context comes from smaller chunks). Expected: bus-bound
at long context (~4× the v1 rate) and T = 4 at ~1.2× T = 1.

## 2. `gdn_mixer` + `gdn_norm` (#22) — `apple-m5-pro-20c_gdn.jsonl`

**Structure.** Block = (value head, group of `SPB` state-column slices of `SL` columns); a SIMD-group takes static
slices of the `Hv · DV/(SL·SPB)` blocks. The recurrence is column-separable, so a block owns its columns' FP32 state
outright: lane ℓ holds k-rows {ℓ, ℓ+32, ℓ+64, ℓ+96} × `SL` columns in registers; per token `S ← S·e^g`,
`kv = kᵀS` (one `simd_sum` per column), `Δ = (v − kv)·β`, `S += k ⊗ Δ`, `o = qᵀS`, and the slice is stored back
once per pass of `TP` tokens. Every block recomputes the head's conv + SiLU (its lane's q/k/v channels over the
window `[conv_state | x_0..x_{T-1}]`), the q/k L2 norms, `β = bf16(σ(b))` and `g = −exp(A_log)·softplus(a +
dt_bias)` — cheap next to the state traffic — and writes its columns of the FP32 read-out to a workspace; `gdn_norm`
(one SIMD-group per (token, head)) applies the gated RMSNorm `bf16(bf16(bf16(o·rstd)·w)·silu(z))`. The conv state is
written by the head's first block (q/k channels by the first value head of the key head). The order and every
rounding follow the reference (`modeling_qwen3_5.py`), so the gates are the leaf ones: conv state exact, recurrent
state ≤ 8 FP32 ULP of its largest value, output ≤ 2 BF16 ULP of each element's magnitude (floored at the RMS) —
green for T = 1 / 4 / 8, Hv = Hk and Hv = 3·Hk, a|b from the same or a second projection buffer, fresh and filled
states, continuation, bit-identical repeat runs [M] (`tests/kernels/test_gdn_mixer.py`).

**Cost** [M], Hk = 16, Hv = 48, dk = dv = 128 (the 27B's shape), best of 10 after a 60 ms warm-up; GB/s counts the
6.3 MB of state read + written per token pass:

| SL | slices/block | blocks (Hv=48) | T=1 µs (GB/s) | T=4 µs (GB/s) | T=8 µs (GB/s) |
|---|---|---|---|---|---|
| 4 | 1 | 1536 | 49 (128) | 132 (48) | 254 (49) |
| 4 | 2 | 768 | 40 (159) | 99 (63) | 186 (68) |
| 4 | 4 | 384 | 31 (205) | 68 (92) | 121 (104) |
| 8 | 1 | 768 | 41 (154) | 103 (61) | 194 (65) |
| 8 | 2 | 384 | 32 (196) | 72 (87) | 133 (94) |
| 8 | 4 | 192 | 26 (244) | 56 (112) | 103 (122) |
| 8 | 8 | 96 | 31 (202) | 76 (83) | 140 (90) |
| 16 | 1 | 384 | 45 (139) | 99 (64) | 186 (68) |
| 16 | 2 | 192 | 39 (163) | 82 (76) | 157 (80) |
| 16 | 4 | 96 | 60 (105) | 133 (47) | 255 (49) |

The 0.8B's 16/16 layers: 20 µs at T = 1, 44 µs at T = 4 (slice 8, 2 per block) [M].

**What it says.** (1) `SL = 8` is the register budget: 16-column slices spill (2× slower at one slice per block)
[M]. (2) Longer blocks win: 4 slices per block amortizes the per-block conv/norm/scalar prologue and reaches
244 GB/s of state at T = 1 — the state traffic (6.3 MB per layer, 302 MB per token for the
27B's 48 layers) is the floor of this op, ~1.2 ms per token; 8 slices per block underfills the crew
(192 blocks for 240 SIMD-groups) [M]. (3) T = 4 costs ~2.2× T = 1 and T = 8 ~4×: the state is read and written
once per pass of 4 tokens and the per-token chain (2·DV `simd_sum`s per slice) is serial, so verification pays
roughly per token here — the profile's `cost_T` for the GDN layers should be re-measured with this kernel (#33/#34).

## 3. The 0.8B decode step on the M5 Pro — the per-token budget (#33)

`python -m monolith.trace --model ~/models/Qwen3.5-0.8B --pack <pack>` (per-dispatch GPU timestamps: one encoder
per op with counter samples at the stage boundaries, so the numbers carry encoder gaps the ICB replay does not
have; min of 5 profiled steps; the ICB replay of the same program runs at 6.84 ms per token) [M]:

| op kind | n | ms (min) | share | GB streamed | GB/s |
|---|---|---|---|---|---|
| gemv | 96 | 4.150 | 60.9 % | 0.995 | 240 |
| lm_head | 1 | 1.699 | 24.9 % | 0.509 | 299 |
| gdn_mixer | 18 | 0.451 | 6.6 % | – | – |
| norm_apply | 49 | 0.208 | 3.0 % | – | – |
| gqa_decode | 6 | 0.172 | 2.5 % | – | – |
| gdn_norm | 18 | 0.078 | 1.1 % | – | – |
| gqa_merge | 6 | 0.038 | 0.6 % | – | – |
| argmax + final, embed, advance, rmsnorm_stat | 5 | 0.015 | 0.2 % | – | – |
| **total** | 199 | **6.810** | | 1.504 | bound at 307 GB/s: **4.90 ms** |

What it says: (1) the `lm_head` (a 0.5 GB BF16 slab, a third of the model) already streams at 97 % of nominal;
(2) the 96 layer GEMVs stream at 240 GB/s = 78 % — the small-K (1024) shapes pay per-row overhead (4 words per lane
per row); per-shape geometry (RG, one block per SIMD-group, threadgroups per core) is the first autotune target
(#34); (3) the mixers cost 0.74 ms: the GDN mixer is latency-bound at 16 heads (64 blocks for 240 SIMD-groups,
25 µs per layer), attention 29 µs per layer; (4) the 49 `norm_apply` dispatches cost 4.2 µs each — dispatch
overhead, not work — so fusing the scaling into the small BF16 GEMVs may win here where it lost on the ALU-bound
NVFP4 shapes: another per-op autotune decision. `mlx-lm` decodes the same model at 6.2 ms per token; parity needs
~0.65 ms of the 1.9 ms between the sum of op minima and the streamed-bytes bound.

**Autotuned (#34 part 1) [M].** Per shape, the autotuner's winners on the M5 Pro: the small-K layer GEMVs move to one
block per SIMD-group with RG 4–8 (8224×1024: 65.6 → 58.9 µs with the norm fused; 5120×1024: 40.7 → 37.6 µs fused;
1024×3584 residual: 34.6 → 28.6 µs; 1024×2048 residual: 20.9 → 17.6 µs), the `gate|up` 7168×1024 keeps the crew at
RG 4 with the norm fused (53.8 → 51.9 µs), the `lm_head` keeps its default (1.70 ms), the 16-head GDN mixer takes
SL 4 / SPB 4 (24.1 → 20.7 µs). The decode step: **6.45 ms per token vs 6.91 default** (three paired runs each), the
golden unchanged; `mlx-lm` 0.31.3: 6.2 ms.
