# Kernel benches

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
