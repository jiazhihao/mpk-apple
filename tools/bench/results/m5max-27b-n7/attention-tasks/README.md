# M5 Max task-based attention evidence

See [the report](../../../../../docs/research/m5max-27b-attention-tasks.md) for
scope, interpretation and reproduction commands. All measurements use N=7
(eight fixed rows) on NVIDIA's Qwen3.8-27B-NVFP4 checkpoint. GPU jobs ran serially.

- `all-layers.csv`: 80 full-layer cases, including 64 cases at the requested
  4K/8K/16K/32K tiers and 16 separate 128-token sanity cases. Latencies are minima
  over nine repetitions of 32 replays. The short case uses capacity 384; the
  four main tiers use capacity 33024.
- `paired-samples.csv`: all 720 paired wall-time samples (576 at the main tiers).
- `tuning.csv`: 327 context points from 100 configuration trials; includes the
  executed config and matched-control numerical screen. These counts include
  repeated/equivalent trials, not 100 distinct kernels.
- `attention-config.json`: shared input for 4K/8K/16K/32K. All four independent
  searches selected the same settings.
- `short-config.json`: the separate 128-token queue setting.
- Search config lists: initial queue, adaptive task sizing, confirmation and
  final tier selection. Their execution order is retained.
- `worker-tasks.csv`: per-stage/per-worker task counters from both audited
  configurations at all five contexts. Stage 0 runs statically and has zero
  queue counts. `selected` identifies the applicable configuration per context.
- `worker-summary.csv`: heavy-phase worker coverage for the five selected cases.
  Counts are not physical-core occupancy or elapsed work. Instrumented timings
  are excluded from performance tables.
- `summary.json`: aggregate results, including both per-layer and per-repetition
  ratios. Medians of ratios need not equal ratios of median latencies.
- `metadata.json`: hardware/software, checkpoint revision, pack-extension
  details, validation, source/raw/compact hashes and local archive location.

The local raw archive contains every timing sample, numerical check, shader
audit, fixed implementation failures and source snapshots. It excludes the
checkpoint and the two copy-on-write pack clones. Search sources changed during
development; the metadata maps each stage to its archived source. Final source
snapshots cover the retained implementation and benchmark. The original weight
pack was preserved; only larger prefix-identical RoPE tables were appended to a
clone. The historical archive retains the original per-tier config filenames;
the shared config has identical contents. Source hashes describe the archived
measurement snapshots; current documentation reflects this consolidation.
