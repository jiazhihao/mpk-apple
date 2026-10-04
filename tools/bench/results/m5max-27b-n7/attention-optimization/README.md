# M5 Max full-attention optimization evidence

See the [research report](../../../../../docs/research/m5max-27b-attention-optimization.md) for results, scope, methodology and limitations.

- `all16.csv` / `all16.summary.json`: every attention layer at five context tiers, compared with original Monolith, the previous recipe, matched and separately selected multi-dispatch controls, and the fastest of six MLX reference variants.
- `selected-contexts.json`: the five-tier recipe lookup. `selected-controls.json` records the independent control selection. Files use content-based names and would be shared if two recipes were identical. The final ten recipes are distinct.
- `finalists-ranking.json`: four-layer confirmation, paired-ratio summaries, selection eligibility and the minimum-time/stability bands.
- `isolated.summary.json`: one-dispatch mixer timings on layers 3, 31 and 63, excluding the two native MLP projections.
- `task-audit.summary.json`: shader-validated logical worker counts. These are not physical per-core timings.
- `coverage.json`: the finite adaptive search, explicit values, accepted/rejected points, source hashes and selection-policy amendments.
- `cache-evaluation-audit.jsonl`: identical-output checks and paired timings for the three MLX cache evaluation/lifetime choices with both FP8 handling paths.
- `final-validation.json`: final numerical, replay, paired-performance and shared shader-regression results.
- `source-manifest.json`: Git base, branch, checkpoint revision, versions and exact source hashes.
- `raw-evidence.tar.gz` / `.sha256`: raw samples, tested configurations, failures, exclusions, reproduction scripts, logs and source snapshots. No checkpoint, pack or derived weight binaries.

The raw archive retains superseded diagnostics: the discarded 32K architecture tail, the initial seed-batch assertion failure and the initial separate-parameter compaction rejection. `measurement-exclusions.json` identifies the invalid timing rows and their serial replacements. Instrumented timings and rejected trials are never performance evidence. The published maps identify final recipes; unreferenced recipes in the raw study directory may come from preliminary selection.

GPU benchmarks must run serially. Reproduction scripts use local checkpoint/pack paths and refuse appending to existing final output files. Adjust those paths after extraction. This is fixed T=8/N=7, seeded-state replay at a fixed position; it is not speculative decoding or generation throughput. Attention fusion remains opt-in.
