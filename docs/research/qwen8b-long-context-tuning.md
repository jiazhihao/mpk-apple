# Qwen3 8B: 4K and 8K decode tuning on M5 Pro

Measured 2026-09-29 on Apple M5 Pro, 20 GPU cores, 24 GB, macOS 26.5.1.
Uses the implementation measured in the [serving comparison](qwen8b-serving-decode.md),
with unchanged NVFP4 target and Qwen3 0.6B affine INT4 draft weights. This study
completes missing projection-tuning entries and tests attention choices at long
context. No production kernel or default selection rule changes.

## Selected configuration

- Enable autotuning/cache reuse and commuted normalization; keep safe math.
- Keep `attention=auto`: matrix attention for N=7 target verification and v3 for
  plain decoding and the sequential draft passes.
- Reserve 8,704 context positions in both packs and the session; use 64-token
  prefill chunks for both plain and speculative measurements.
- For speculation, keep gamma=7, fixed verify length=7 and the existing one-round
  command buffers with two in flight. Plain keeps eight steps/CB and three in flight.

The [saved cache](../../tools/bench/results/qwen8b-long-context-20260929/autotune.apple-m5-pro.json)
contains 77 choices: 68 previously measured choices plus nine missing tile choices
measured here. The earlier short-context probe retained defaults on those misses;
this run uses the normal autotuner. Projection keys depend on matrix/pack shape,
not KV length, so long-context attention was screened separately.

## Controlled tuning screen

One real prefill per context, restored mutable state before each replay, one
excluded warmup and four measured repetitions. Candidate order alternates.
Plain replays 32 tokens; N=7 replays eight full rounds. Native wall milliseconds:

| Mode / configuration | 4K (4,095 tokens) | 8K (8,191 tokens) |
|---|---:|---:|
| Plain, tuning off | 18.27 / token | 21.11 / token |
| Plain, tuning on | 18.24 / token | 21.10 / token |
| N=7, tuning off | 69.66 / round | 89.60 / round |
| N=7, tuning on | 51.69 / round | 71.53 / round |

Projection tuning reduces N=7 round latency by **25.8% / 20.2%** in this paired
screen. Plain changes by less than 0.2%; that is not evidence of a plain-decode
speedup. This table is a controlled replay, not the full-generation result below.

Forcing matrix attention everywhere costs 62.58 / 90.04 ms per N=7 round;
forcing v1 costs 117.48 / 187.87. Reducing v3 to 16 or 8 SIMD-groups also loses.
The target matrix-attention grid was screened at 20/40/80/160 groups and
4/8/16 SIMD-groups. Its best alternative, 160 groups × 8 SIMD-groups, improves
whole rounds by only 1.4% / 2.5% against the interleaved control, below the
3% promotion margin in the design. The existing 80 × 8 geometry remains selected.
An explicit plain-v2 attempt failed the existing workspace-capacity check before
execution and produced no timing. It is excluded from the ranking.

Every measured replay repeats its token sequence exactly for its configuration.
The fresh-generation prefix checks below independently verify the restored-state
path. There is no claim that changing attention algorithms or projection reductions
preserves all output tokens.

## Full-generation validation

Each context uses the same task with six request-number variants: one excluded
warmup and five measured requests. Every request gets fresh prefill/KV and
generates 128 tokens through unmodified `Session.generate`; decode timing covers
the following 127 tokens. Median request-average wall milliseconds:

| Metric | 4K | 8K |
|---|---:|---:|
| Plain ms/output token | 18.26 | 21.12 |
| N=7 ms/full round | 51.92 | 71.08 |
| N=7 ms/output token | 21.27 | 30.20 |

All 24 requests completed; all speculative rounds verified seven drafts. The
fresh `r1` output prefix matches the restored-state screen exactly for both
modes and contexts. Plain/speculative full outputs match on 1/10 measured
prompt pairs; this is performance validation, not a reference-accuracy gate.
The measured N=7 round ranges are 51.64–52.21 ms at 4K and 70.998–73.60 ms at 8K.

N=7 still loses to plain decoding per output token on this low-acceptance
workload. The tuning reduces the round cost; it does not establish a useful
speculation speedup here. [Issue #133](https://github.com/jiazhihao/mpk-apple/issues/133)
tracks that remaining policy/acceptance problem. The prior report's untuned plain
run used 128-token prefill chunks and smaller cache capacity, so use this study's
controlled screen when attributing changes to tuning. MLX/vLLM/llama.cpp were not
rerun at 8K in this tuning study.


## Reproduction

Both pack directories must cover 8,704 positions. The local packs for this study
are `/tmp/mpk-long-context/target-pack` and `/tmp/mpk-long-context/draft-pack`.
They preserve every packed weight slab and extend only the BF16 RoPE tables;
every prior table prefix was checked byte-identical. A new pack with a different
physical layout is a different configuration and may need additional tuning.

Copy the archived `autotune.apple-m5-pro.json` into the matching target pack to
reuse the measured choices. Run the screen (`--geometry` adds the grid search),
then fresh generation:

```bash
python tools/bench/long_context_tune.py --mode n7 \
  --model /path/to/mlx-Qwen3-8B-nvfp4 --pack /path/to/target-pack \
  --drafter /path/to/mlx-community-Qwen3-0.6B-4bit --drafter-pack /path/to/draft-pack \
  --prompts tools/bench/results/qwen8b-long-context-20260929/prompts.json \
  --outdir /tmp/long-context-screen
# Add --generate --autotune for the full-generation validation.
# Use --mode plain and omit both drafter paths for plain decode.
```

For normal generation, the equivalent existing CLI settings are
`--commute-norm --attention auto --max-context 8704 --prefill-chunk-size 64`;
autotuning is on unless `--no-autotune` is supplied. For N=7 add
`--drafter-kind lm --draft-gamma 7 --verify fixed --verify-length 7` and both draft
paths. The benchmark HTTP adapter differs: it requires explicit `--autotune` to
preserve its old untuned-command semantics, and now accepts the context/attention
options and records their values in each result.

[Raw evidence and methodology](../../tools/bench/results/qwen8b-long-context-20260929/)
include prompts, every screen sample, timing ranges, choices and full generated
token lists. These are one workload family on one machine, with sequential full
request blocks; no tail-latency or cross-model claim is made.
