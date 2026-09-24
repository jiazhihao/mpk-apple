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
