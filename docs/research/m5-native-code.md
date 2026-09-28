# M5 Pro native-code investigation — 2026-09-28

The fixed-token latency gate in #113 remains open. This investigation supports a
small matrix-reduction improvement; it does not establish that all layers beat MLX.
The separate follow-up is PR #122; PR #121 has already merged.

## Evidence and limits

Machine: Apple M5 Pro, 20 GPU cores, 24 GB, macOS 26.5.1. Baseline: `42bd649`.
Reference: MLX 0.32.2 (`ml-explore/mlx` tag `v0.32.2`, commit
`1f8e74e3f12f31365464a6867c6579f0e9b29d85`), mlx-lm 0.31.3.

Runtime compilation through `MTLBinaryArchive` successfully exports native GPU
Mach-O code. It requires only Command Line Tools and the public Metal runtime.
The machine has no Xcode Metal disassembler. The available open-source decoders
are unsuitable for this GPU/compiler combination: dougallj/applegpu at
`4c5bae61086b8067231120c98b4756d7696d399c` predates it; the base-M5 decoder in
niklassheth/agx-re at `e25433b1d348a809ab86484ce149ff7d459e8cdf` loses instruction
boundaries and labels compute instructions as texture operations. **No instruction
counts, assembly listings, or spill-load counts from those decoders are evidence.**

Instead, compare exact machine-code bytes, pipeline shared-memory allocation,
and raw compiler metadata. `metal_codegen.py` preserves native code and raw
metadata fields; its statistic hints are explicitly experimental. The metadata
field interpretation comes from the author's own
[M5 register experiment](https://github.com/niklassheth/agx-re/blob/e25433b1d348a809ab86484ce149ff7d459e8cdf/experiments/EXP-M5-21-gpr-machine-model/report.md).
It was independently checked on this M5 Pro by compiling cyclic dependent FMA
chains with 8, 32, 64, and 128 live scalar values:

| Live values | Field 0: register footprint | Fields 14/41: scratch bytes |
|---:|---:|---:|
| 8 | 13 | 0 |
| 32 | 43 | 0 |
| 64 | 83 | 0 |
| 128 | 126 | 400 |

Field 28 also matches the public `staticThreadgroupMemoryLength` property.
These are compiler allocations, not hardware occupancy measurements. Scratch
can contain addressable local arrays as well as spills, so scratch alone does
not prove register-pressure spilling.

## Findings

1. **Default threadgroup limits are not the cause in the sampled kernels.**
   All 55 production specializations emitted identical main instruction bytes
   with the default pipeline limit and with the actual dispatch limit plus
   `threadGroupSizeIsMultipleOfThreadExecutionWidth`. Smaller limits therefore
   offer no instruction-generation improvement in this sample.
2. **NVFP4 matrix kernels have a different resource problem from BF16.**
   The four sampled NVFP4 matrix projections use 126 registers and 80–96 bytes of
   scratch. BF16 matrix projections use 60–79 registers with zero scratch.
   Reverting NVFP4 decode to V2 grows a sampled projection from 96 to 272 bytes
   of scratch. Half decode temporaries, explicit element decoding, and forced
   scale-index selection did not eliminate scratch. These experiments were not
   retained.
3. **MLX's actual BF16 path is a small-token SIMD kernel.** A temporary diagnostic
   interposer records pipeline creation and archives the exact function through
   public Metal APIs. The T=4 GDN layer uses
   `gemv_wide_bfloat16_nv4_kl32_nc0_axpby0`: 103 registers, zero scratch, and zero
   static threadgroup memory. Its GDN recurrence uses 33 registers and zero
   scratch; MPK's prepared recurrence uses 51 and zero. These differences motivate
   experiments but do not by themselves prove a latency cause. MPK's existing
   SIMD projection fallback regressed versus its matrix path (T=4: 233.80 vs
   203.51 µs/layer; T=8: 638.93 vs 208.60). A new full-row prefetch prototype also
   regressed. Neither was retained.
4. **K-split reductions process padded token rows.** The hardware accumulator's
   layout includes 16 rows even for short verification batches. MPK stored and
   summed those padded rows in shared memory. A compact scratch layout omits
   unreachable rows, maps live slots back to the same accumulator elements,
   and omits the final reuse barrier when no subsequent tile exists. A persistent
   threadgroup still executes that barrier before reusing scratch.

The resulting native resource changes at T=4 include:

| Projection | Registers before → after | Shared bytes before → after |
|---|---:|---:|
| INT4 QKV | 73 → 71 | 1,024 → 256 |
| INT4 attention output | 83 → 69 | 7,168 → 1,792 |
| INT4 gate/up | 107 → 85 | 15,360 → 3,840 |
| INT4 down | 85 → 70 | 7,168 → 1,792 |
| BF16 main GDN projection | 79 → 60 | 7,168 → 1,792 |

At T=8 the scratch layout halves shared memory. The compiler rejects a
four-row matrix operand (`M must be a multiple of 8 or 16`), so this optimization
reduces the surrounding reduction work without changing the hardware matmul.

## Paired whole-layer measurements

Same fixed inputs, T, and context 128; 7 alternating AB/BA repetitions of 32
steps, two command buffers in flight; minimum wall time divided by the number
of distinct checkpoint layers. All paired output cosines are 1.0 (within FP64
roundoff); an additional INT4 before/after check confirms exact output equality.

| Format / layer kind | T | Original µs | Compact µs | Decision |
|---|---:|---:|---:|---|
| INT4 attention | 4 | 88.14 | 86.01 | Enable |
| INT4 attention | 8 | 90.18 | 86.88 | Enable |
| BF16 GDN, refreshed geometry | 4 | 200.72 | 200.70 | Essentially tied |
| BF16 GDN, refreshed geometry | 8 | 205.58 | 205.00 | Small improvement |
| NVFP4 attention | 4 | 488.49 | 490.12 | Keep original |
| NVFP4 attention | 8 | 492.24 | 492.73 | Keep original |

Only INT4 and BF16 opt into compact partials. The autotune key distinguishes
this implementation, because changing the reduction cost can change the best
K-split geometry. Some NVFP4 samples show background-load spikes; neither their
minimum nor their paired results justify enabling the change.

Raw native reports: `tools/bench/results/apple-m5-pro-20c_native_codegen_20260928.jsonl`.
Raw paired samples: `tools/bench/results/apple-m5-pro-20c_live_partials_ab_20260928.jsonl`.

## Individual-layer gate and measurement caveat

The benchmark now supports `--individual` and `--layers`. Its original stack
mode is unchanged by default. A selected diagnostic on INT4 layers 0, 13, and
27 at T=1/4 and context 128 reports lower MPK wall latency for all six cases.
This **does not close #113**: isolated layer replay changes cache residency and
amortizes MLX's Python/command-submission overhead differently. The streaming
stack still has regressions and remains a required companion check. Do not
substitute these isolated wins for a GPU kernel performance conclusion.

## Reproduce the native inspection

```sh
python tools/bench/layer_fixed_vs_mlx.py --model /path/to/checkpoint \
  --pack /path/to/pack --layers 0 --ts 4 --ctx 128 \
  --export-kernels /tmp/layer-kernels
clang++ -std=c++17 -fobjc-arc -framework Foundation -framework Metal \
  tools/bench/metal_archive.mm -o /tmp/metal-archive
/tmp/metal-archive /tmp/layer-kernels/layers-0_t4_ctx128/002_gemm_tile.json /tmp/kernel.metallib
python tools/bench/metal_codegen.py /tmp/kernel.metallib --out /tmp/kernel-native
```

Choose the exported specialization filename for the desired dispatch. The JSON
contains source, macros, and dispatch metadata, never weights. Optional JSON
fields `max_threads` and `simd_multiple` control the pipeline-limit experiment;
`fast_math`, `language_version`, and `support_icb` preserve runtime compile settings.
Do not benchmark with capture, the pipeline interposer, or shader validation on.

Apple documents the public archive workflow in
[Creating binary archives](https://developer.apple.com/documentation/metal/creating-binary-archives-from-device-built-pipeline-state-objects)
and recommends measuring rather than assuming a benefit from
[threadgroup limits](https://developer.apple.com/documentation/metal/mtlcomputepipelinedescriptor/maxtotalthreadsperthreadgroup).

After enabling compact partials and refreshing tuning, the streaming MLX check is:

| T | Context | MPK µs | MLX µs | MPK faster in every pair |
|---:|---:|---:|---:|---|
| 4 | 128 | 84.93 | 71.74 | False |
| 4 | 1024 | 107.04 | 95.20 | False |
| 6 | 128 | 86.24 | 87.60 | True |
| 6 | 1024 | 110.14 | 120.36 | True |
| 8 | 128 | 87.12 | 103.42 | True |
| 8 | 1024 | 112.11 | 146.75 | True |

T=4 remains slower at both contexts. Raw samples are in
`apple-m5-pro-20c_fixed_int4_compact_20260928.jsonl`.
