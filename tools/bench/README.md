# Kernel benches

For full-model single-request decode against vLLM-Metal, llama.cpp Metal and
Ollama, use `single_request_latency.py`. It records native decode counters,
warmups and generated text. See the [Qwen3 8B comparison](../../docs/research/qwen8b-serving-decode.md)
for measured results, exact prompts and pinned server settings.
`mlx_spec_step_latency.py` adds direct MLX-LM plain decode (`--mode plain`) and
full N=7 round timing with the same prompts, excluding prefill and shortened tail
rounds. It retains per-token timestamps; `--prefill-step-size` supports a matched
prefill check against Monolith as well as the native generator defaults.
`target_verify_latency.py` isolates the N=7 target forward (eight positions),
using identical prefixes and input IDs for Monolith and MLX, excluding drafting
and sampling. See the [long-context target measurements](../../docs/research/qwen8b-long-context-tuning.md#isolated-n7-target-verification).
`profile_spec_round.py` splits the existing N=7 dispatch stream into timed stages
and compares against unsplit controls. Optional `--cache` reuses saved tuning
choices and leaves cache misses at default, without searching. The serving
adapter originally disabled autotuning; its archived results are labeled accordingly.

`long_context_tune.py` screens Qwen3 8B plain/N=7 decode at 4K and 8K using
identical prefills, then `--generate --autotune` validates fresh full generations.
See the [long-context tuning report](../../docs/research/qwen8b-long-context-tuning.md)
for the selected configuration and saved M5 Pro choices. The serving adapter now
accepts `--autotune`, `--attention` and `--max-context`; omitting `--autotune`
retains its historical untuned behavior.

`gemv_bench.py` runs the production-shaped `kernels/gemv_T.metal` (assembled by `monolith.kernels` from a format plugin's
decode snippet and a pack geometry) on the target's shapes, checks every run against the exact format oracle (the leaf-op
gate: ≤ 2 BF16 ULPs at the output's magnitude, float32 accumulation noise < 1e-4), and streams ≥ 2 GB of identical packs
per measurement (min-of-3, GB/s of useful bytes). Knobs = the profile values of design D4/D8: rows per block `R`, tokens
`T`, activation row group `RG`, lane order, threadgroups per core or one block per SIMD-group.

```bash
python tools/bench/gemv_bench.py --format nvfp4 --shape 17408x5120 --rows 16 --t 1 --lane-order interleaved16
python tools/bench/gemv_bench.py --sweep m1 --out tools/bench/results/<chip>_gemv_m1.jsonl
python tools/bench/gemv_fusions_ab.py --out tools/bench/results/<chip>_gemv_fusions.jsonl   # cost of the norm/residual/stat fusions
python tools/bench/gqa_bench.py --heads 32 --kv 4 --ctx 1024,4096,8192,32768 --t 1,4 --out tools/bench/results/<chip>_gqa.jsonl   # decode attention vs context
```

Results are JSON lines under `results/` (one file per chip and study); commit them like probe results. The M1 sweep is
the table plan M1's gate is read from; `p13` in `probes/` is the standalone precursor of this harness.

## Fixed-token decoder layers versus MLX

`layer_fixed_vs_mlx.py` measures a dependency chain of distinct checkpoint decoder
layers with embeddings, vocabulary projection, sampling and acceptance excluded.
Each replay verifies exactly `T` rows at the same context position in both engines.
The reported microseconds per layer are the directly measured stack latency divided
by the number of selected layers; they are **not individual-layer measurements**.
Attention and GDN stacks can be selected independently for hybrid models.

For the measured M5 NVFP4 layout, repack the same checkpoint with
`tools/pack_weights.py --model <checkpoint> --out <pack> --scale-placement block --scale-order payload`.
The v3 manifest records the physical scale order; the original codes and dequantized weights are unchanged.
Existing packs remain readable and retain their recorded layout. Include the pack path and layout in reports.

```bash
python tools/bench/layer_fixed_vs_mlx.py \
  --model ~/models/mlx-community-Qwen3-0.6B-4bit --pack /tmp/pack-06b \
  --ts 1,4,6,8 --ctx 128,1024 --attention auto --reps 5 --steps 32 \
  --out tools/bench/results/<chip>_fixed_layers.jsonl --fail-on-regression
# For a hybrid checkpoint, repeat with --kind attention and --kind gdn.
```

Pack the same checkpoint with sufficient RoPE capacity first (at least
`max(ctx) + max(T) + 256`, the benchmark's cache allocation). The compiler rejects
undersized constant tables. Both engines use the original checkpoint weights and
BF16 activations, with identical seeded nonzero KV prefixes by default. GDN starts
from the same zero recurrent/convolution input state on each replay; this is a
fixed-state kernel comparison, not a generation or acceptance-rate benchmark.
`--kv-prefix zero` permits comparison with older zero-prefix measurements.

The harness warms both engines and alternates AB/BA order with two evaluations in
flight. It saves every paired wall-time sample, MPK GPU time, minimum wall time,
software versions, selected layer indices and the minimum output cosine against
MLX, feeding the same input to each corresponding layer. A cosine below 0.999
aborts. `--fail-on-regression` exits nonzero if any minimum-time ratio is at least
one; `faster_in_every_pair` separately reports consistency across repetitions.
A faster stack mean does not establish that every individual layer is faster.
Run without Metal shader validation for timing, and use validation for correctness.

Use `--individual` to measure every selected layer separately, or `--layers 0,13,27`
to narrow the checkpoint indices. These isolated replays have different weight
cache residency and host-overhead amortization; keep streaming stack runs as a
companion check. See [the M5 performance report](../../docs/research/m5-native-code.md)
for the final result summary and archived native-code investigation tools.

`layer_grid_search.py` screens launch geometries for the 20-core M5 Pro's short
Qwen 8B tiles. It compares each operation role across all checkpoint layers,
checks outputs, and records every paired sample. Single-operation roles amortize
submission over a batch; their weights may be cache-resident. A leaf winner must
be confirmed in the full dependency chain, with an unchanged-program control and
relevant context lengths, before changing the compiler. `variant(program, role,
config)` can apply a recorded configuration to a complete program for that check.

```bash
python tools/bench/layer_grid_search.py --model CHECKPOINT --pack PACK \
  --ts 6,8 --ctx 128 --out grid-search.jsonl
```

## Experimental static GDN fusion

`gdn_static_bench.py` compares the production N=7 GDN core with whole-head and
fixed-worker single-kernel schedules, including a fenced stage-barrier variant.
Use `--confirm --layers 48 --repeat 2` to stream distinct layer states rather
than repeatedly reusing one small state. See the [experiment and results](../../docs/research/gdn-static-megakernel.md)
for scope, correctness checks, and the measured regressions. Write raw samples to
`/tmp` or another local results path; these schedules are not enabled by default.

`gdn_block_bench.py` extends that experiment to the **entire GDN block**, including
all input/output projections and residual addition, with seeded FP8/BF16 weights
at the 27B shapes. It compares production with a matching geometry/operand control
and a single static megakernel. The [full-block follow-up](../../docs/research/gdn-static-megakernel.md#full-gdn-block-including-matrix-projections)
records near parity on M5 Pro and the remaining M5 Max handoff requirements.
