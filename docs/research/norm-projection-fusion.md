# Commuted input normalization on M5

`--commute-norm` enables an experimental, relaxed-rounding fusion in generation;
`Session(..., commute_norm=True)` exposes the same option. It defaults off because
latency improves on some shapes and regresses on others. Existing packs work unchanged.

For reciprocal RMS `r(h) = rsqrt(sum(h*h)/K + eps)`, a projection can use
`r(h) * ((h * gamma) @ W.T)`. This uses the same algebra as
[MPK's normalization fusion](https://github.com/lithos-ai/mirage/blob/5beaed87bbf3ec341bde00917c400c411e3f6a33/docs/mpk/decode_linear.md).
MPK folds gamma into weights; this implementation instead writes `BF16(h * gamma)`
from the preceding residual projection, in the consumer's packed input order.
The consumer applies `r(h)` to its FP32 accumulator before SiLU/multiplication or
residual addition. The existing raw BF16 residual and its norm statistics remain intact.
No external implementation was copied.

The compiler removes the intervening norm/permutation dispatch for eligible
2–8-token tile consumers. Matching siblings share the new scratch; different
norm weights retain separate paths. The first embedding-fed normalization,
one-token decoding, larger prefill tiles and mixed shader/tile schedules retain
the original path. The producer must also be a tile covering the full token range.

Moving BF16 rounding changes results: this is an explicit exception to the
original leaf-ULP/token-equality gate, not a change to the default path's contract.
Tests check the reordered formula, unchanged residual/statistic outputs, distinct
norm weights, barriers and shrinking/growing active rows. Layer cosine below is
measured against MLX on each layer's actual Monolith input; it does not establish
whole-generation token equality or task quality.

## Measurements

[M] Apple M5 Pro, 20 GPU cores, 24 GB, macOS 26.5.1, MLX 0.32.2 / mlx-lm 0.31.3.
All checkpoint layers execute in a dependency chain with distinct weights, fixed
context 128, random BF16 input/KV prefix, and unchanged checkpoint weights. Times
are wall microseconds per layer (stack latency divided by layer count), not
isolated kernel times or acceptance-dependent generation throughput. `T=N+1`.

Each result uses nine paired repetitions of 48 replays, rotating through the six
orders of original Monolith, fused Monolith and MLX. Compilation/first-touch is
warmed outside timing; shader validation is disabled. Values are minimum [range]
over repetitions. Raw samples, cosine and settings are in
[the result file](../../tools/bench/results/apple-m5-pro-20c_commute-norm.jsonl).

| Model | N | Original µs [range] | Fused µs [range] | MLX µs [range] | Fusion change | Min layer cosine |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B INT4 | 3 | 71.30 [71.30–76.28] | 69.51 [69.51–76.10] | 71.98 [71.98–76.35] | -2.52% | 0.999970 |
| Qwen3 0.6B INT4 | 5 | 77.36 [77.36–77.46] | 75.07 [75.07–75.16] | 87.53 [87.53–87.77] | -2.95% | 0.999965 |
| Qwen3 0.6B INT4 | 7 | 79.87 [79.87–80.26] | 76.37 [76.37–76.44] | 103.28 [103.28–103.50] | -4.39% | 0.999965 |
| Llama 3.2 3B INT4 | 5 | 306.07 [306.07–317.74] | 291.54 [291.54–302.27] | 456.16 [456.16–458.50] | -4.75% | 0.999992 |
| Llama 3.2 3B INT4 | 7 | 293.44 [293.44–305.04] | 308.95 [308.95–323.35] | 566.66 [566.66–574.73] | +5.29% | 0.999992 |
| Llama 3.2 1B INT4 | 5 | 179.91 [179.91–180.26] | 182.27 [182.27–182.73] | 280.70 [280.70–281.58] | +1.32% | 0.999993 |
| Llama 3.2 1B INT4 | 7 | 183.80 [183.80–184.20] | 189.70 [189.70–189.89] | 343.39 [343.39–345.40] | +3.21% | 0.999992 |
| SmolLM2 1.7B BF16 | 5 | 499.27 [499.27–531.17] | 495.36 [495.36–525.13] | 491.14 [491.14–515.60] | -0.78% | 0.998086 |
| SmolLM2 1.7B BF16 | 7 | 499.29 [499.29–535.91] | 494.67 [494.67–533.73] | 496.77 [496.77–506.70] | -0.92% | 0.999867 |
| Qwen3 8B NVFP4 | 5 | 440.40 [440.40–475.23] | 445.02 [445.02–492.03] | 646.22 [646.22–661.04] | +1.05% | 0.999975 |
| Qwen3 8B NVFP4 | 7 | 440.15 [440.15–514.04] | 447.65 [447.65–532.55] | 825.42 [825.42–860.21] | +1.70% | 0.999976 |

Qwen 0.6B at N=5/7 and Llama 3B at N=5 have non-overlapping original/fused ranges. Qwen 0.6B N=3 and SmolLM2 show overlapping ranges, so their lower minima are not conclusive wins. Llama 1B, Llama 3B N=7 and Qwen 8B do not benefit from enabling this option. SmolLM2 N=5 also remains slower than MLX by the measured minimum.

The fusion removes 55 normalization dispatches on the 28-layer models, 31 on Llama 1B, 47 on SmolLM2 and 71 on Qwen 8B. Its added producer stores and repeated per-tile statistic reductions can outweigh the saved dispatches. Moving the reciprocal computation after MMA and using adjacent four-lane reductions reduced the preliminary 8B regression, but did not eliminate it. Further optimization and shape-aware selection before default enablement are tracked in [#131](https://github.com/jiazhihao/mpk-apple/issues/131).

Reproduce with the same target checkpoint and pack:

```sh
python tools/bench/layer_fixed_vs_mlx.py --model CHECKPOINT --pack PACK \
  --ts 6,8 --ctx 128 --reps 9 --steps 48 --norm-ab --min-cosine 0 --out results.jsonl
```

`--norm-ab` compares original/fused/MLX in one process. `--min-cosine 0` explicitly
requests finite-output sanity while recording the measured cosine; the default
remains 0.999. `--commute-norm` without `--norm-ab` compares only fused Monolith
against MLX. No claim is made that every individual layer beats MLX.
