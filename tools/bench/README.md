# Kernel benches

`modelopt_layer_bench.py` compares fixed eight-row NVIDIA ModelOpt decoder layers
against native MLX-LM, including its original-code NVFP4 and FP8 matrix paths.
`--fusion --config <geometry.json>` evaluates two static megakernels per layer;
`modelopt_mega_tune.py` searches each half's geometry with a matched multi-dispatch
control. Both tools exclude generation and speculative-decoding performance.
See the [M5 Max 27B study](../../docs/research/m5max-27b-megakernel.md) for measured
results, correctness checks, the faster-of-two MLX baseline and reproduction commands.
The [further tuning study](../../docs/research/m5max-27b-megakernel-tuning.md)
adds explicit `--configs` lists, independent K splits and per-projection settings,
compact reductions, barrier/scheduling variants and recurrence geometry.
`--reference-config` pairs the previous geometry with a new candidate;
`--fusion-scope mixer-prefix` evaluates a three-dispatch mixer/native-MLP hybrid
while preserving the full layer's normalization boundary. `modelopt_payload_pack.py`
creates an isolated, byte-verified scale-layout experiment without changing the source pack.

The M5 Max profile now selects the measured GDN mixer/native-MLP hybrid for
matching static eight-row layers. `--default-fusion` compares this automatic
compiler selection against original Monolith and MLX-LM. The `production`
benchmark column explicitly disables fusion to preserve the original baseline.
Attention retains native kernels under automatic selection. The
[GDN optimization follow-up](../../docs/research/m5max-gdn-mixer-optimization.md)
records the packed FP8 recipe and its comparison with the earlier default.
`--comparison-control-config` adds an independently tuned packed multi-dispatch
baseline, separate from the candidate's matched control. Both tools retain
per-round samples and generated shader hashes; output and model state must stay
bit-identical after fixed-input timing replays. Keep shader validation disabled
for performance measurements and enabled for separate correctness audits.

The [attention tuning study](../../docs/research/m5max-27b-attention-tuning.md)
adds independent attention task counts (`attention_groups`), query tile rows
(`attention_qm`), active merge SIMD groups (`merge_sgs`) and merge load-ahead
(`merge_unroll`). Set `--kind attention-prefix --ctx 128,8192` in the tuner to
screen the complete attention/native-MLP layer. Use
`--reference-fusion-scope mixer-prefix` with `--reference-config` to pair two
mixer-only fusion geometries while retaining the original Monolith baseline.

The [task-based attention follow-up](../../docs/research/m5max-27b-attention-tasks.md)
extends the search to 4K/8K/16K/32K using one shared `attention-config.json`. The
experimental compiler accepts `schedule: queue`, `task_grain: tile`,
`task_batch`, `attention_task_tiles` and `task_seed`; `task_stats` adds optional
per-worker counters for audits. Keep counters and shader validation disabled
for timing. `modelopt_extend_tables.py` clones a pack and appends larger,
prefix-identical position tables. The layer benchmark's `--capacity` keeps
Monolith KV/RoPE capacity fixed across context tiers.

The [full-attention optimization follow-up](../../docs/research/m5max-27b-attention-optimization.md)
adds bounded local key partitions (`attention_chunk_tiles`), optional Q/K
preparation, scratch reuse and compact partial buffers. It retunes projection
geometry and task scheduling for the 40-core M5 Max. Its context map references
deduplicated recipe files; select the entry for the measured prefix length and
pass that file with `--config`. Attention fusion remains opt-in.
`--mlx-cache-lifetime both --fp8-mode both` compares six MLX reference variants:
both FP8 handling paths crossed with three cache evaluation/lifetime choices.
The per-pair gate uses the fastest MLX variant in that round. Use `--kind
attention` to isolate the mixer and `--kind attention-prefix` to retain the two
native MLP dispatches in complete-layer tuning.

The [MLP optimization study](../../docs/research/m5max-27b-mlp-optimization.md)
compares lossless NVFP4 operand packing, native device-tensor projections and
one-dispatch MLP fusion. `--mlp-config` adds a selected MLP recipe and its matched
control; `--mlp-control-config` adds an independently tuned native reference.
With `--part layer --fusion-scope mixer-prefix`, both preserve the mixer's
residual/normalization boundary. A single MLP recipe serves every measured
context tier. MLP tuning remains explicit; automatic generation selection is
unchanged.

`modelopt_mlp_suffix_tune.py --configs <list.json>` isolates the MLP at that
producer boundary. It snapshots only external inputs, checks the isolated
output against the complete-layer program, and records randomized paired
samples. Use `--control-only` for native recipes. The standalone
`modelopt_mega_tune.py --kind mlp` includes its own normalization/permutation
dispatches, so its timings have a different scope. `nvfp4_layout: tile` creates
content-addressed operand files without changing the original weight pack.
The study retains accepted/rejected trials, shared shader tests and all-layer
comparisons against original Monolith and the fastest tested MLX-LM path.

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

Keep compact summaries, reproducibility metadata, and reusable configurations in `results/`.
Bulky sweep logs, per-request outputs, and repeated prompts belong in an archive linked
from the report. The [results README](results/README.md) explains how to restore the
historical raw evidence, including the M1 sweep used by `m1_gate_table.py`.
`p13` in `probes/` is the standalone precursor of the M1 harness.

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

`vllm_target_verify.py` and the `ollama_target_verify_test.go` overlay measure
full eight-row target forwards on the M5 Max at 128/4K/8K/16K/32K context.
They include all eight vocabulary-logit rows and exclude drafting/acceptance.
Both require documented checkpoint compatibility adapters; vLLM-Metal's hybrid
speculative scheduler is unsupported, so its result uses the paged prefill path.
See the [backend verification study](../../docs/research/m5max-27b-backend-verification.md)
for timings, numerical checks, rejected runs, and exact reproduction metadata.

`dspark_round_latency.py` measures complete non-terminal DSpark rounds after real
prefill: eight-row target verification, acceptance and recurrent-state commit,
then the next seven-token draft block. It also times the same dispatches as three
ICB stages, checking split/full token identity. `dspark_tune.py` compares native,
normalized and fused draft-layer recipes with exact control/fusion and cache
checks. Add `--compare-draft-fusion` to the round benchmark for an alternating
full-step comparison that changes only draft fusion, sharing resident weights.
`--profile-draft` records every draft dispatch with GPU timestamp counters and
cross-checks the attribution with contiguous ICB spans. It preserves actual
target features and verifies next-proposal identity against normal replay.
`--compare-config` alternates full rounds and draft spans against a previous
draft recipe while holding the target fixed. `--generation-tokens` sets the
length of the four actual generation checks (64 by default).

`dspark_kernel_tune.py` screens explicit projection, mixer, MLP, scalar fallback
and Markov recipes on packed checkpoint weights. It keeps numerical failures,
checks replay identity, and records alternating timings; `--inject 1` exercises
the one-row feature/KV fallbacks. Use full real-prompt rounds as the final gate,
since hot-weight microbenchmarks do not reproduce whole-draft cache behavior.
`dspark_compare_packs.py` pairs complete rounds from two draft packs after
independent real prefills. It shares only identical file-backed weights and
keeps each precision's caches and recurrent state private. It reports acceptance
with latency and checks equal committed target prefixes; use actual generation
checks to evaluate the throughput effect of changed acceptance.
See the [M5 Max DSpark study](../../docs/research/m5max-27b-dspark.md)
for checkpoint revisions, context recipes, memory handling and measurements.
The [draft refinement study](../../docs/research/m5max-27b-dspark-refinement.md)
records the later projection, task scheduling, scalar fallback and precision
searches, with paired full-round checks against the preceding selected recipe.
