# M5 Max GDN optimization evidence

See the [research report](../../../../../docs/research/m5max-gdn-mixer-optimization.md) for scope, results, configuration and limitations.

- `all48.csv` / `all48.summary.json`: complete GDN layer comparisons with original Monolith, the previous recipe, both packed controls and MLX-LM.
- `selected.json`: the installed megakernel recipe; `selected-control-layer.json`: independently selected packed control.
- `coverage.json`: completed finite screening domains, accept/reject counts and hashes.
- `chain5.json`, `chain-finalists-ranking.json`, `chain64-real.json`: chain checks and fixed-input timings. The 64-layer result uses real prompt embeddings and zero initial state.
- `chain64-diagnostic.json`: 256 passing local checks; the synthetic full-depth accumulated gate remains a documented failure for old and new recipes.
- `final-validation.json`: validation outcomes and evidence hashes.
- `raw-evidence.tar.gz` / `.sha256`: raw paired samples, all tested configurations, rejected trials, scripts, logs and source snapshots. No weight binaries.

The archive retains two invalid preliminary diagnostics, explicitly labeled `stale-scratch` and `layout-mismatch`. Use the final `study/chain_diagnostic.py`, which captures inputs before each layer and translates normalized layouts; the historical generator does not include those corrections. Instrumented audit timings and rejected trials are never performance evidence.

GPU benchmarks must run serially. Reproduction scripts use local model/pack paths and refuse appending to existing final outputs; adjust paths after extraction. `source-manifest.json` records the exact source hashes, Git base/branch and checkpoint revision.
