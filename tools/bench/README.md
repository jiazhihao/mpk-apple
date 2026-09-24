# Kernel benches

`gemv_bench.py` runs the production-shaped `kernels/gemv_T.metal` (assembled by `monolith.kernels` from a format plugin's
decode snippet and a pack geometry) on the target's shapes, checks every run against the exact format oracle (the leaf-op
gate: ≤ 2 BF16 ULPs at the output's magnitude, float32 accumulation noise < 1e-4), and streams ≥ 2 GB of identical packs
per measurement (min-of-3, GB/s of useful bytes). Knobs = the profile values of design D4/D8: rows per block `R`, tokens
`T`, activation row group `RG`, lane order, threadgroups per core or one block per SIMD-group.

```bash
python tools/bench/gemv_bench.py --format nvfp4 --shape 17408x5120 --rows 16 --t 1 --lane-order interleaved16
python tools/bench/gemv_bench.py --sweep m1 --out tools/bench/results/<chip>_gemv_m1.jsonl
```

Results are JSON lines under `results/` (one file per chip and study); commit them like probe results. The M1 sweep is
the table plan M1's gate is read from; `p13` in `probes/` is the standalone precursor of this harness.
