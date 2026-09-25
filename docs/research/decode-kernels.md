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

**v2 — built and measured (#34) [M].** `gqa_decode_v2` + `gqa_merge_v2` (`kernels/gqa_decode_v2.metal`; the
profile's `engine.attention = "v2"` or `--attention v2` selects it): one threadgroup per (kv head, batch of 12
chunks) with the block's rep·T query rows normed + RoPE'd once into threadgroup memory (BF16, ≤ 32 rows), the
step's new keys appended by their batch behind a threadgroup barrier, lane-per-key scoring against the rows in
passes of `RG` rows (q broadcast from threadgroup memory, one `simd_max` / `simd_sum` per row per 32 keys instead
of a `simd_sum` per (key, row)), lane-per-dim P·V with p̃ broadcast by `simd_shuffle`, the 32-key partials folded
online in registers (exact FP32 rescaling, so the numbers equal v1's at chunk 32 folded hierarchically — the same
numpy contract, `test_gqa_decode.py`), and a chunk of 32 · {1, 2, 4} keys per SIMD-group chosen from the context so
every threadgroup keeps a block. Same table, same method (min of 8), v1 re-measured alongside:

| heads/kv | context | T | v1 µs/layer | v2 µs/layer | v2 / v1 |
|---|---|---|---|---|---|
| 32/4 | 1024 | 1 | 81.5 | 70.6 | 0.87 |
| 32/4 | 4096 | 1 | 244 | 229 | 0.94 |
| 32/4 | 8192 | 1 | 453 | 489 | 1.08 |
| 32/4 | 32768 | 1 | 1745 | 1433 | 0.82 |
| 32/4 | 1024 | 4 | 202 | 220 | 1.09 |
| 32/4 | 4096 | 4 | 684 | 820 | 1.20 |
| 32/4 | 8192 | 4 | 1431 | 1777 | 1.24 |
| 32/4 | 32768 | 4 | 5490 | 4992 | 0.91 |

`RG = 8` rows per pass is 5–30 % slower than 4 everywhere (registers), and rep·T > 32 rows (T = 8 at 32/4 heads) does
not fit the query cache (the emitter falls back to v1). What it says: the hypothesis above was wrong about the
cost. Removing the per-(key, row) reduction buys 6–18 % at T = 1 and loses at T = 4 below 32 K, so the per-pair cost
is not the `simd_sum` but the BF16 → FP32 conversions and the loads around each 8-dim word (v2 pays them for q per
(row, word) per key, v1 pays them for k per row group); K/V once per step does not matter while the kernel is
issue-bound at 20–90 GB/s. The structure that changes the per-pair cost by an order of magnitude is the
SIMD-group matrix unit — Q·Kᵀ and P·V as 8 × 8 BF16 tiles (`simdgroup_multiply_accumulate`, or MPP tensor ops on
Apple10) — which is the same path the T ≥ 2 GEMMs need (M9, #50/#51); the attention core joins that work. Until
then v1 stays the default (its T = 4 is 2.5–3× T = 1, 24 GB/s of KV); v2 is kept as the per-profile option it is
(a win at 32 K and at T = 1). The long-context rows of the 27B (16 layers, 32/4 heads): 28 ms per token at 32 K on
v1, 23 ms on v2.

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

**Barrier placement and the sibling overlap (#29, #35) [M].** Two facts first. (1) The ICB barrier flag orders the
*flagged command behind everything before it* (a `setBarrier` command waits for all preceding commands; commands
without it may start while their predecessors run) — measured with a slow writer / fast reader pair
(`tools/bench`-style probe, 2026-09-24: the reader without the flag sums a partially written buffer; with it, never);
the runtime's field is now `barrier_before`. (2) The mixers are emitted as core → gate GEMV → merge / gated norm,
the gate projection a block-aligned row range of the same slab (`[q | k | v | gate]`, `[z | qkv | a | b]`) so the
bus-bound gate GEMV runs beside the ALU-bound core (design §5.12), and the barrier pass keeps a flag only where an
op touches what the ops before it wrote. On the 0.8B: 175 dispatches, 151 barriers (the 18 GDN and 6 attention
gate GEMVs un-barriered); paired runs of 128 tokens, 4 rounds, min | median ms per token:

| program | ms / token |
|---|---|
| every op barriered (v0) | 6.848 \| 6.868 |
| barrier pass, core encoded first (`alu_first`, the profile's rule) | **6.584 \| 6.625** (−3.9 %) |
| barrier pass, gate encoded first (`bus_first`) | 6.630 \| 6.667 (−3.2 %) |

The ranges of the two orders touch (6.649 vs 6.630), so the Apple10 rule is the better one by ~1 % within noise —
kept as the profile says. The 8B (no gate: a pure chain) keeps all 223 barriers and its 26.9 ms; the drafter's
context projections and the per-T variants are the other un-barriered groups of a speculative program.

**Math modes (#34) [M].** The kernels compile in Metal's *safe* math mode (the runtime's default; the numerics
contract was met under it). Fast math (`--math fast`, `Session(fast_math=True)`), paired 4 rounds of 128 tokens:
0.8B 6.72 → 6.60 ms per token (−1.9 %, ranges overlapping at the edge), 8B 27.55 → 27.30 (−0.9 %). The 0.8B's 48
golden tokens hold under fast math (`test_fast_math_keeps_the_golden`) but its 128-token story diverges from the
safe run after the golden's length (a near-tie flipped), and the 8B's does not. 1–2 % is not worth a mode that
breaks bit-identity with the reference: safe stays the default, fast stays an option.

**The M5 gate, accounted (#34).** The 0.8B decodes at 6.58 ms per token against the 4.90 ms bound of its streamed
bytes at 307 GB/s: 74 % of nominal, below the survey's 85–90 % practical ceiling. Where the remainder is
(`monolith.trace`, per-op minima): the 96 small-K layer GEMVs at 240 GB/s (78 %; K = 1024 rows pay per-row
overhead, the autotuner's block geometry took 5–17 % off but not the rest — a GEMM-style tile over several rows
would), the `lm_head` already at 97 %, the mixers 0.6 ms (GDN latency-bound at 16 heads, attention as above), and
~0.4 ms of dispatch boundaries for 175 dispatches (1.4 µs each) that neither fusion nor barriers can remove without
multi-op kernels (D5 says no). The 8B (NVFP4) decodes at 26.9 ms against a 20.5 ms bound: 76 %, the ALU-bound
NVFP4 decode of the M1 study (59 % of nominal at T = 1 in isolation, better in the mix) — its remainder is the
NVFP4 decode itself, which the M9 GEMM path addresses for T ≥ 2 and a wider-word decode would for T = 1.

## 4. The DSpark round's kernels (#24) — `apple-m5-pro-20c_draft.jsonl`

**What exists.** The round of design §5.8 lowers to the existing op kinds plus five of its own
(`monolith/ops/draft.py`): the tapped residual streams of the committed positions are concatenated (`tap_concat`)
and go through the feature projection as a plain `gemv` + `norm_apply` over the rows `StepState.n_inject` names; the
block `[anchor, mask × (γ − 1)]` (an `embed` variant that reads the anchor from StepState) runs through the drafter's
layers, whose attention is `gqa_decode` with `DRAFT=1`: the same blocks and passes with three key sources — the
drafter's context cache, the injected positions' k/v from a second projection (normed, RoPE'd and appended to the
cache like new tokens) and the block's own k/v (never appended) — and no mask; the target's `lm_head` at T = γ; the
Markov chain as γ (`embed` W₁ row → `gemv` W₂ with the residual epilogue rounding the product first → `argmax`)
triples chained through row views of the block's logits and tokens; the confidence head (`confidence`); the
verify-length select and the accept scan as SERIAL ops on StepState. Row counts are per value: the `T` symbol reads
`t_this_step`, `N_INJ` reads `n_inject`, a static γ compiles in (`T_SRC`), so one dynamic-T program holds the target's
step, the injection and the block. The emitted draft program (40 dispatches for a two-layer synthetic drafter, `tests/kernels/test_draft_program.py`)
reproduces the drafter oracle: features cos > 0.9999, block hidden and base logits within 3 % of scale, drafts and
verify bookkeeping identical, confidences within 2·10⁻².

**Cost on the M5 Pro** (`python tools/bench/draft_bench.py`; the 8B drafter's geometry: 32 heads, 8 KV heads, head
dim 128, γ = 7; min of 20 after warm-up) [M]:

| kernel | context | µs | note |
|---|---|---|---|
| `draft_attn` (+ merge), 1 or 8 injected | 0 | 20 / 26 | the block alone: 28 query rows, 7 row groups |
| | 1 024 | 201 | 21 GB/s of K/V counted once — the 7 row groups re-stream every chunk (v1, §1) |
| | 4 096 | 793 | ×5 layers = 4.0 ms per round at 4 K: the v2 attention (#34) is on the drafter's critical path too |
| `tap_concat` (8 rows × 5 taps × 4 096) | – | 5.6 | |
| `confidence` (7 rows × 4 352) | – | 27 | one SIMD-group per row over a 4 352-long dot: latency-bound; a REDUCE form would cut it |
| `verify_select` | – | 3.5 | |
| `accept_scan` | – | 2.0 | |

What it says: at the contexts v1 is judged on (≤ 1 K) the drafter's own attention costs 1 ms per round for five
layers, below one W₂ pass of the Markov chain (the 78 MB BF16 W₂ streamed γ times ≈ 3.6 ms) and far below the
drafter's weights (1.9 GB BF16 for the 8B drafter ≈ 13 ms) — the round's cost is the drafter's GEMVs and the target's
`lm_head` at T = γ, exactly the bytes the dspark.md §3 estimate counts; the serial ops are dispatch-cost only.

## 5. The speculative step on the M5 Pro (#38) — Qwen3-8B NVFP4 with its public DSpark drafter

**What runs.** One dynamic-T program holds the whole round (design §5.8): the target's verify pass over the
pending tokens `[anchor, d_1 … d_L]`, `accept_scan`, the recurrent-state commit passes (none for this dense model),
the drafter's injection of the committed positions' tapped features, its block pass at T = γ = 7 through the target's
`lm_head`, the Markov chain, the confidence head and `verify_select`; replayed from one encode for prefill chunks and
decode alike, tokens drained from the ring — the host is idle (`host busy 0.0 %`). The target's GEMVs come as
predicated per-T variants (T = 1, 2, 4, 8): each variant is compiled for its own T and returns at once unless the
step's T falls in its range, so the ALU-bound NVFP4 work follows the verify length rather than `t_max` (without the
variants every step paid the T = 8 cost: 190 ms). The variants add 450 early-returning dispatches (~1 ms) to the 378.

**Measured** (`python -m monolith.generate … --drafter … --verify …`, 128 greedy tokens, the tokens identical to the
plain decode's in every run; `python -m monolith.trace … --drafter …` for the budget) [M]:

| decode | ms / token (GPU) | tok/s | tokens / step | mean accepted (of 7) |
|---|---|---|---|---|
| plain | 26.8 | 37.3 | 1 | – |
| speculative, cost-aware rule, story prompt (no chat template) | 37.5 | 26.7 | 1.59 | 1.05 |
| speculative, confident-prefix 0.5, story prompt | 41.2 | 24.3 | 1.59 | 1.21 |
| speculative, whole block verified (L = 7), story prompt | 76.1 | 13.1 | 1.76 | 1.74 |
| speculative, L = 0 (the round's fixed cost) | 60.0 | 16.7 | 1.00 | 0 |
| speculative, cost-aware rule, chat-template code prompt | 27.1 | 36.9 | 1.98 | 2.14 |

The step's budget with the cost-aware rule (sum of per-op minima over 5 profiled steps, story prompt): 70.4 ms —
GEMVs 52.6 ms (the target's 36 layers at T ≤ 4 ≈ 33 ms, the drafter's 5 BF16 layers at T = 7 ≈ 19 ms: its
`gate_up` alone 1.85 ms per layer = 109 GB/s), `lm_head` 15.3 ms (the drafter's block at T = 7: 11.1 ms = 112 GB/s;
the target's at T ≤ 4: 4.2 ms), attention 1.3 ms, `norm_apply` 0.9 ms, the draft attention 0.18 ms, the serial ops
< 0.05 ms. The L = 0 run measures the round's fixed cost directly: 60 ms = the plain step (27) + the draft pass (33).

What it says: (1) correctness holds — greedy speculative decode is token-identical to plain greedy decode on the
8B (and on the hybrid 0.8B with a random drafter that forces a rollback every step: the GDN commit pass); (2) on the
shader-FMA GEMV path the round does not pay for itself here: the draft pass costs 33 ms because BF16 at T = 7 runs
at ~110 GB/s (ALU-bound) and each verified draft costs the NVFP4 cost table's ×1.28 … ×1.79, so the cost-aware rule
caps L at 3 and the best case is parity (the chat-template prompt, 2.1 accepted); (3) acceptance is the drafter's,
not ours — the GPU's drafts equal the oracle's on the golden's real target features — and it depends on the prompt
format the drafter was trained on (2.14 with the chat template vs 1.05 without). The levers are the ones the design
names: a SIMD-group-matrix / MPP GEMM for T ≥ 2 (M9, #51) for both the verify pass and the drafter's block pass
(MLX's `qmm_t` streams NVFP4 at 85–91 % of nominal at T = 2–4 on this chip), the drafter's weights in FP8/NVFP4, and
the acceptance measurement on the prompt set (#40) — done: dspark.md §3 has the gate table over 11 prompts (the
cost-aware rule 36.2 ms per token vs plain 27.0; fixed L = 1 / 2 / 3 / 7: 42.6 / 40.6 / 36.9 / 78.1) and the STS
calibration (not needed: the head is calibrated as shipped).

