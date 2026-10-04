# Chip profiles

One file per measured chip (design §5.7: profiles are *measured, never hard-coded*). The measurement record — the
probe blocks — is derived by hand from the probe results in `probes/results/`; every value carries the probe it came
from, ranges are min–max over the same-day runs, single numbers are min-of-N. The `engine` block is written by the
autotuner at install time, `tools/profile_writer.py` (#49): it runs the kernel harnesses on the machine and writes
`profiles/<chip>-<cores>c.json`, merging into an existing file (the probe blocks stay; the previous `engine` block is
kept under `writer.previous_engine`, the raw numbers and the reasons for each decision under `writer`):

```bash
python tools/profile_writer.py --dry-run          # ~5 minutes: measure and print the engine block
python tools/profile_writer.py                    # write (or refresh) this machine's profile
python tools/profile_writer.py --nominal-gbps 307 # a new chip: the spec bandwidth (else a measured stand-in, flagged)
```

It measures the lane order and the threadgroups per core (the T = 1 GEMV rate), `cost_T` per format (the shader
GEMV at T = 1, 2, 4, 8, the best geometry per T, in units of the T = 1 pass), the tile's rows (`accelerator_<fmt>`
at 8 / 16 / 32 token rows, the same unit) and from the two the accelerator switch and `accelerator_min_t` per format,
and the attention kernel (v1 vs v2). `sibling_order` (p11) and `max_cb_ms` (p6/p6b) are the probes' and carry over
(the safe defaults `either` / 16 on a new chip). Every decision follows the autotuner's rule (a 3 % noise margin; a
tie keeps the file's value) — `monolith/core/profile_writer.py`, tested without a GPU in the contract tier.

| File | Chip | Results files |
|---|---|---|
| `apple-m3-pro-18c.json` | M3 Pro, 18-core GPU, 36 GB, macOS 26.6.2, on battery | `Apple-M3-Pro_18c_macOS26.6.2_20260919-223834.txt` |
| `apple-m5-pro-20c.json` | M5 Pro, 20-core GPU, 24 GB, macOS 26.5.1, on AC | `Apple-M5-Pro_20c_macOS26.5.1_20260922-*.txt` (11 files: full suite, p12 ×3, p6/p6b ×3 repeats, p13, p14 ×2) |

## The `engine` block

Everything else in a profile is the measurement record; `engine` is the normalized part the compiler reads
(`monolith.core.profile.Profile`): `family` (the kernel-binding key), `lane_order` of the weight pack
(`contiguous` | `interleaved16`), `scale_placement` (`inline` | `block`: where a pack keeps its block scales — the
block region drops the lane-row unit's padding where a lane's scales fit one word, #101), `threadgroups_per_core`,
`sibling_order` (`alu_first` | `bus_first` | `either`), `max_cb_ms`, `attention` (`v1` | `v2` | `v3` | `auto` = v3:
core and merge in one dispatch, a threadgroup per query row, faster than v2 and v1 at every row count and context
measured on the M5 Pro — #113), `attention_v2_threadgroups` (v2's threadgroups per core, 2 by default),
`attention_rows` (v1's query rows per pass over a chunk, 4 by default), `accelerator` (`on` | `off`: T > 1 GEMVs on
the tensor-ops tile) with `accelerator_min_t` per format, and `cost_T` — per format, the cost of a T-token pass
relative to T = 1 (the shader rows, and `accelerator_<format>` rows at the tile's TM), which the verify-length rule
(design §5.8) optimizes against. Costs are exact at measured T and linear between them; the loader refuses to
extrapolate. The writer (`tools/profile_writer.py`) measures the lane order, the scale placement (at K = 4096),
the threadgroups, the cost tables, the accelerator decision and the attention kernel; the rest carries over.

## GDN mixer fusion

The 40-core M5 Max profile's `engine.gdn_mixer_fusion` selects the measured
fixed-eight-row GDN mixer recipe. Its six `shape` entries are hidden width,
key heads, value heads, key dimension, value dimension and convolution width.
The remaining fields describe the static worker geometry, lossless FP8 operand
packing and each projection's tile, K split and task grid. The current recipe
uses 80 workers with 8 SIMD groups, TN=32, eight-column recurrence slices,
tile-block=8 and subtraction decoding. QKV/AB/Z/output grids are 240/240/120/120;
their K splits are 2/8/8/4 (AB uses TN=16). Empty or absent
means native dispatches. Selection requires an exact shape match, FP8/BF16
projections, the complete MLP normalization boundary, static T=8 and commuted
normalization; dynamic and speculative programs retain native kernels.
The native MLP and attention kernels remain selected. See the
[M5 Max optimization study](../docs/research/m5max-gdn-mixer-optimization.md)
for the search, selected configuration and comparisons. Input normalization and
dual permutation are combined only where a preceding native residual has not
already produced the normalized layout. Derived operand files are cached outside
the repository; the original weight pack is unchanged.
