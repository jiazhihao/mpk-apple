# GEMV kernel study — the M1 harness on the M5 Pro

Status: measured 2026-09-24 on the M5 Pro (20-core GPU, 24 GB, macOS 26.5.1, AC power) with
[`tools/bench/gemv_bench.py`](../../tools/bench/gemv_bench.py) (plan M1, issues #9 and #10). Every point is min-of-3
over ≥ 2 GB streamed, checked against the exact format oracle (≤ 2 BF16 ULP at the output's magnitude, accumulation
noise < 1e-4); the raw JSON lines are in [`tools/bench/results/`](../../tools/bench/results). Nominal = 307 GB/s.

## 1. The kernel

`kernels/gemv_T.metal`: static slices over the block-lane-major pack (design D8), 16-byte weight loads in either lane
order, the format plugin's decode snippet, in-word block scales, the per-row tensor scale from the pack, BF16
activations converted once per word and reused across `RG` rows, FP32 accumulation, one `simd_sum` per (row, token).
Knobs: rows per block `R`, tokens `T`, row group `RG`, lane order, and the geometry — the crew (12 SIMD-groups per
core, `×n` = n threadgroups of 384 per core) or one block per SIMD-group in small threadgroups (the MLX/llama.cpp
shape, "1blk/SG tg64").

## 2. M1 sweep (reference NVFP4 decode, R = 16, RG = 2) — `apple-m5-pro-20c_gemv_m1.jsonl`, 252 points

Best geometry per shape and T; the crew ×1 number in the last column:

| format | shape | T = 1 | T = 2 | T = 4 | crew ×1, T = 1 |
|---|---|---|---|---|---|
| FP8 | 17408×5120 (gate/up) | 260 GB/s (85 %) | 248 (81 %) | 109 (36 %) | 254 |
| FP8 | 5120×17408 (down) | 279 (91 %) | 238 (77 %) | 124 (41 %) | 201 |
| FP8 | 10240×5120 (GDN qkv) | 277 (90 %) | 234 (76 %) | 118 (39 %) | 250 |
| FP8 | 12288×5120 (q + gate) | 264 (86 %) | 255 (83 %) | 104 (34 %) | 229 |
| FP8 | 6144×5120 (z) | 231 (75 %) | 255 (83 %) | 89 (29 %) | 231 |
| FP8 | 5120×6144 (o / out) | 273 (89 %) | 242 (79 %) | 124 (40 %) | 196 |
| FP8 | 248320×5120 (lm_head) | 291 (95 %) | 288 (94 %) | 125 (41 %) | 274 |
| NVFP4 | 17408×5120 | 157 (51 %) | 110 (36 %) | 78 (26 %) | 118 |
| NVFP4 | 5120×17408 | 143 (47 %) | 110 (36 %) | 80 (26 %) | 88 |
| NVFP4 | 248320×5120 | 184 (60 %) | 127 (41 %) | 89 (29 %) | 131 |

Also at the crew geometry, T = 1, 17408×5120: BF16 284 GB/s (93 %), INT8 266 GB/s (87 %).

## 3. NVFP4 decode study — `apple-m5-pro-20c_nvfp4_decode.jsonl`

Three exact decodes of the E2M1 nibbles (`monolith/formats/nvfp4.py`, macro `NVFP4_DECODE`):

* **V0** — per nibble, float bit construction with two selects (the `p13` decode; ~14 ALU ops per weight);
* **V1** — nibble pairs decoded into a `half2` with packed 16-bit integer arithmetic (both halves of a 32-bit word
  at once), then converted to `float2`;
* **V2** — the eight magnitudes as small integers (`value × 2`) in one 32-bit constant, four bits each, an
  `int → float` conversion and a sign select; the `× 0.5` folds into the block scale.

T = 1, `interleaved16`:

| shape | geometry | V0 | V1 | V2 |
|---|---|---|---|---|
| 17408×5120 | crew ×1, R = 16 | 118 GB/s (39 %) | 150 (49 %) | 161 (52 %) |
| 17408×5120 | crew ×4, R = 16 | 146 (47 %) | 179 (58 %) | 211 (69 %) |
| 17408×5120 | 1blk/SG tg64, R = 8 | 168 (55 %) | 203 (66 %) | 243 (79 %) |
| 17408×5120 | 1blk/SG tg64, R = 4, RG = 4 | — | — | **252 (82 %)** |
| 5120×17408 | 1blk/SG tg64, R = 4, RG = 4 | — | — | 226 (74 %) |
| 248320×5120 | 1blk/SG tg64, R = 4 | — | 246 (80 %) | **273 (89 %)** |

T > 1 does not benefit from the decode (V2, 1blk/SG: T = 2 130–151 GB/s, T = 4 89–99): once the decode is cheap the
FMAs per weight dominate, which is the accelerator path's territory (design §5.6, `p14`).

## 3b. The remaining knobs (#11) — `apple-m5-pro-20c_kernel_knobs.jsonl`, 108 points

* **Rows per block at T = 1** (one block per SIMD-group / crew ×1, GB/s): FP8 R = 2: 287/273, 4: 287/270, 8:
  285/259, 16: 275/259, 32: 280/222; NVFP4 (V2) R = 2: 256/175, 4: 249/169, 8: 242/163, 16: 225/161, 32: 231/136;
  INT8 R = 2: 282/281 … 32: 281/243; BF16 flat at 287–293. Small blocks win for the decoded formats; the crew ×1
  geometry pays 5–10 % on FP8 and ~30 % on NVFP4, more at large R (tail quantization: 17408/32 = 544 blocks over
  240 SIMD-groups is 76 % efficient).
* **T > 1 on the shader ALUs** (best of R ∈ {4, 8, 16} × RG ∈ {2, 4, 8}, one block per SIMD-group): FP8 T = 2 278
  GB/s (91 %), T = 4 139 (45 %), T = 8 88 (29 %); NVFP4 T = 2 156 (51 %), T = 4 102 (33 %), T = 8 19 (6 %; register
  spills). RG = 4 is the right row group for T ≥ 2 (the M1 sweep's RG = 2 was 20 % worse at T = 4).
* **`safe` vs `fast` math**: identical throughput (284.6 vs 284.7 GB/s FP8; 251.9 vs 251.7 NVFP4) and identical
  outputs — the kernel has no transcendental or division; stay in `safe`.

## 3c. Against MLX on the same shapes (#12) — `apple-m5-pro-20c_mlx_baseline.jsonl`

MLX 0.32.2 `quantized_matmul` (`mode="nvfp4"`, group 16: the same bytes per weight as our pack; affine 4-bit group 64;
BF16 matmul), wall-clock over ≥ 2 GB of weight copies, min-of-5 (`tools/bench/mlx_baseline.py`; the table is printed
by `tools/bench/m1_gate_table.py`):

| shape | T | ours FP8 | ours NVFP4 (V2) | MLX nvfp4 | MLX affine-4 | MLX bf16 |
|---|---|---|---|---|---|---|
| 248320×5120 | 1 | 291 (95 %) | 273 (89 %) | 286 (93 %) | 285 | 288 |
| 17408×5120 | 1 | 287 (94 %) | 256 (83 %) | 281 (91 %) | 279 | 272 |
| 5120×17408 | 1 | 279 (91 %) | 226 (74 %) | 284 (92 %) | 282 | 287 |
| 17408×5120 | 2 | 278 (91 %) | 156 (51 %) | 280 (91 %) | 279 | 282 |
| 17408×5120 | 4 | 139 (45 %) | 102 (33 %) | 262 (85 %) | 253 | 285 |
| 6144×5120 | 4 | 89 (29 %) | 62 (20 %) | 245 (80 %) | 235 | 284 |

(The narrow-shape NVFP4 rows at T = 1 are being re-measured with V2; the table in `m1_gate_table.py` is the record.)

Readings:

1. **T = 1, NVFP4: 0.80–0.95× MLX.** MLX's `qmv` streams NVFP4 at 91–93 % of nominal on every shape; our V2 kernel is
   at 83–89 % on the wide shapes and 74 % on `down` (K = 17408). The M1 gate asked for ≥ 1.10× MLX; **that is not
   met**, and the honest reading is that a 4-bit GEMV on this chip is a solved problem at ~92 % — the remaining lever
   for plain decode is fusion and GPU autonomy, as the design's §2 ledger already said, not the GEMV geometry.
   FP8 (which MLX does not have) is at 91–95 %.
2. **T = 2–4: MLX is 2–4× faster than our shader kernels.** MLX runs T > 1 through its `qmm_t` path — dequantize a
   tile and multiply with `simdgroup_multiply_accumulate` (SIMD-group 8×8 matrix FMAs, available on every Apple GPU
   since Apple7) — and stays at 85–91 % of nominal at T = 2–4 on the 5120-K shapes. Our T > 1 kernels issue one FMA
   per weight per token on the shader ALUs and are ALU-bound from T = 2 (NVFP4) or T = 4 (FP8), and `p14`'s MPP
   `matmul2d` path (61 % at T = 8) is also behind MLX at T = 4. **Consequence for the design:** the verify pass of
   speculative decoding (design §5.8) must use a SIMD-group-matrix kernel for T ≥ 2 — with it, a T = 4 pass should
   cost ~1.1× a T = 1 pass on this chip instead of the ×1.8–2.7 the profile's `cost_T` table currently records from
   the shader kernels, which makes DSpark's block of 4–7 drafts pay much better than the FMA numbers suggested.

## 4. What it says

1. **FP8 is bus-bound at T = 1** (85–95 % of nominal); the crew geometry is at parity on the wide shapes and 5–30 %
   behind on the narrow ones (fewer blocks per SIMD-group: 6144-row matrices give 1.6 blocks per crew SIMD-group, so
   tail quantization and the lack of latency hiding both bite). One block per SIMD-group in 64-thread threadgroups is
   the safer default; the crew geometry with 2–4 threadgroups per core recovers most of the gap.
2. **NVFP4 was ALU-bound with the reference decode and is close to bus-bound with V2**: 82 % on gate/up, 89 % on
   lm_head, 74 % on down (K = 17408: 34 scale bytes per lane-row spill into a third scale word and the unit pads 306 →
   320 bytes). V2 is the default. Remaining ideas for the last 10–20 %: fold the scale bytes into the payload words for
   long K, a `half`-domain dot for the 16-weight group (a numerics-gate question), and the tail-quantization-aware
   block count.
3. **Small R wins for NVFP4** (R = 4 > 8 > 16 at one block per SIMD-group): fewer registers per row group and finer
   slices; for FP8 R = 8–16 is flat. `RG = 2` beats 4 for FP8 at the crew geometry (254 vs 223 GB/s) and RG = 1 is
   worse everywhere.
4. **T = 2 costs ×1.05–1.15 for FP8 and ×1.6–1.8 for NVFP4** relative to the same shape at T = 1 (best geometries);
   T = 4 costs ×2.3 (FP8) and ×2.7 (NVFP4) with RG = 2 — worse than `p13`'s ×1.11 / ×1.79 at RG = 4, so the T > 1
   kernels need their own tuning (issue #11) and the accelerator path from T ≈ 3–5.

## 5. Against plan M1's gate

| gate | status on the M5 Pro |
|---|---|
| FP8 shapes ≥ 100 GB/s effective | met (231–291 GB/s) |
| NVFP4 T = 1 ≥ 80 % of nominal | met on gate/up (82 %) and lm_head (89 %); 74 % on down — the K = 17408 layout item above |
| NVFP4 T = 1 ≥ 1.10× MLX's kernel on the same machine | **not met**: 0.80–0.95× (MLX `qmv` is at 91–93 % of nominal; ours 74–89 %) — §3c |
| outputs within 2 ULP (BF16) of the oracle | met on every point (max 0.0 ULP at the output's magnitude, max relative error 1.8e-7) |
