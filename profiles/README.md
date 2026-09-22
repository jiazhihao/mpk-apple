# Chip profiles (provisional)

One file per measured chip, derived by hand from the probe results in `probes/results/` (design §5.7: profiles are
*measured, never hard-coded*; the autotuner will eventually write them). The schema is provisional until the runtime
exists; every value carries the probe it came from. Ranges are min–max over the same-day runs; single numbers are
min-of-N.

| File | Chip | Results files |
|---|---|---|
| `apple-m3-pro-18c.json` | M3 Pro, 18-core GPU, 36 GB, macOS 26.6.2, on battery | `Apple-M3-Pro_18c_macOS26.6.2_20260919-223834.txt` |
| `apple-m5-pro-20c.json` | M5 Pro, 20-core GPU, 24 GB, macOS 26.5.1, on AC | `Apple-M5-Pro_20c_macOS26.5.1_20260922-*.txt` (11 files: full suite, p12 ×3, p6/p6b ×3 repeats, p13, p14 ×2) |
