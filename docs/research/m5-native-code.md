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

## Prepared recurrence and packed BF16 SIMD follow-up

Further inspection captured the actual MLX T=4 INT4 projection
`affine_qmv_wide_bfloat16_t_gs_64_b_4_nv_4_kl_8_batch_0` (56 registers,
no scratch) and `sdpa_vector_bfloat16_t_128_128_nomask_qnt_c_nosinks`
(33 registers, no scratch, 4,352 shared bytes). These are archived pipeline
specializations, not an inference from dispatch names in source code.

Two changes were retained after paired whole-layer measurements:

- Prepared GDN processes **four adjacent state columns per SIMD group**, with
  four SIMD groups per threadgroup. It shares the prepared query/key loads
  across those columns while preserving each column's FP32 recurrence order.
  Both the parameter's work count and the dispatched grid shrink by four.
- **BF16 K=1024, compiled T=4, interleaved packs** use a SIMD projection with
  16 lanes per output row. Eight weight vectors are prefetched and reused over
  four token vectors. The kernel reads the existing permuted activation layout
  and implements row ranges, residual/statistic outputs and fused gate/up
  permutation. Larger K and T retain their matrix kernels. The prefetch and
  row-reduction structure is adapted from MLX's MIT-licensed `GemvWide`; the
  source and license are recorded in `third_party/NOTICE`.

The BF16 SIMD specialization reports 98 registers for plain projections and
101 for gate/up, both with zero scratch. Prepared GDN changes from 51 to 69
registers, also with zero scratch. These faster kernels use **more** registers:
register count alone is not a performance ranking. The recurrence trades those
registers for input reuse, and the SIMD projection avoids padded matrix work.

Same-day A/B against `b8edaaf`, random BF16 inputs (seed 17), zero recurrent input
slot, 18 distinct GDN layers, context 128, 7 alternating repetitions of 32 steps,
two command buffers in flight:

| T | Before µs/layer | After µs/layer | Faster in every pair |
|---:|---:|---:|---|
| 4 | 208.61 | 201.87 | Yes |
| 6 | 206.26 | 203.63 | Yes |
| 8 | 209.31 | 205.18 | Yes |

T=6/8 exercise only the recurrence change and have identical measured outputs.
The T=4 SIMD reduction changes floating-point summation order; its whole-stack
before/after cosine is 0.999390. The stricter same-input **per-layer** comparison
against MLX passes with minimum cosine 0.999982 at T=4. Leaf tests independently
check projection, epilogue, state-predicate and permutation correctness.

The updated MLX gate (7 repetitions of 48 steps) remains **open**:

| T | MPK µs/layer | MLX µs/layer | MPK / MLX |
|---:|---:|---:|---:|
| 4 | 200.80 | 182.44 | 1.101 |
| 6 | 202.33 | 182.64 | 1.108 |
| 8 | 203.64 | 188.45 | 1.081 |

Raw samples and native reports are in
`apple-m5-pro-20c_gdn_bf16_ab_20260928.jsonl`,
`apple-m5-pro-20c_fixed_gdn_bf16_20260928.jsonl`, and
`apple-m5-pro-20c_gdn_bf16_native_20260928.jsonl` under `tools/bench/results`.

Rejected experiments include direct BF16 tensor loading, larger SIMD prefetches
and T=8 SIMD projections, register-softmax attention, and context-predicated
attention dispatches. A single-SIMD register-softmax prototype allocated 400
scratch bytes per thread and took about 125 versus 85 µs per INT4 layer at
T=4/context 128. Independent smaller SIMD tiles reduced that loss but still
lost. The existing v3 attention is faster at short T=4 contexts, but adding
predicated v3/core/merge dispatches saved only 2.6 µs there and cost 3.0 µs at
context 1024. Neither change was enabled.

The initial source-replacement harness for GDN geometry did not update the
compiler's `HANDLERS` table. Those measurements are invalid and were discarded.
The retained A/B rebuilds the emitter and asserts the emitted recurrence slice
width and number of SIMD projection dispatches before timing.

Validation also exposed an **intermittent, independently reproduced baseline
failure** when reusing a session for a long prompt after the oracle and short
prompt checks. It is tracked separately in
[#123](https://github.com/jiazhihao/mpk-apple/issues/123). Fresh-session checks
and repeated sequences passed, but a pass on rerun does not resolve this issue.
The failing dynamic-T=8 programs do not contain the new BF16 SIMD kernels.

The combined validation run passed **342 tests**, with 3 skipped, under Metal
shader validation: contracts, matrix and GDN kernel tests, the 25 new BF16 SIMD
cases, and the real-model speculative rollback golden. The real-model layer
oracle and twice-repeated 48-token short golden also passed. The independently
reproduced session-reuse issue above remains open.

## Shared affine scale runs

The K=1024 affine INT4 pack stored each 64-column group's scale/bias pair twice:
its two 32-column lane stripes carried separate copies. Sharing that run removes
64 bytes per weight row without changing payload order or arithmetic. On the
0.6B checkpoint, the pack shrinks from 372,015,104 to 343,932,928 bytes, including
the embedding. The decoder alone sheds 655,360 bytes per layer.

`scale_lane_divisor` records the address rule explicitly. New manifests use
version 2 so an older reader refuses them; this reader still supports version 1.
The packer verifies identical scale bytes before dropping copies. GEMV, matrix
fill, embedding gather, unpacking and the autotuner use the same metadata. The
sharing decision is based on format and stripe/group geometry, not model identity.
Only affine INT4 opts in; no unmeasured NVFP4 layout change is included.

With identical tuning decisions, seven alternating AB/BA pairs of 32 steps,
random BF16 input and nonzero KV prefix, all 28 layer outputs are byte-identical:

| T | Context | Original µs/layer | Shared scales µs/layer |
|---:|---:|---:|---:|
| 1 | 128 | 60.74 | 58.85 |
| 4 | 128 | 85.37 | 82.40 |
| 4 | 1024 | 108.14 | 103.97 |
| 8 | 128 | 86.76 | 83.31 |

Every paired sample improves. The actual QKV native specialization retains
71 registers, zero scratch, 256 shared bytes and 3,808 code bytes; gate/up retains
85 registers, zero scratch and 3,840 shared bytes (8,392 → 8,406 code bytes).
The measured gain is consistent with lower weight traffic, not reduced register
allocation. This is resource/byte evidence, not decoded ISA or a hardware counter.

With fresh tuning of the compact layout, the paired MLX gate is still open:

| T | Context | MPK µs/layer | MLX µs/layer | MPK faster in every pair |
|---:|---:|---:|---:|---|
| 1 | 128 | 58.97 | 57.48 | False |
| 1 | 1024 | 71.57 | 68.18 | False |
| 4 | 128 | 80.84 | 71.35 | False |
| 4 | 1024 | 102.51 | 94.86 | False |
| 6 | 128 | 81.51 | 87.32 | True |
| 6 | 1024 | 105.03 | 120.17 | True |
| 8 | 128 | 82.94 | 103.28 | True |
| 8 | 1024 | 107.78 | 146.38 | True |

These are streaming stack means, not proof that every individual layer wins.
The minimum same-input per-layer cosine against MLX is above 0.9999. Two
additional generation comparisons (5- and 19-token prompts, 32 generated tokens
each) match the old pack exactly. Validation: 154 contract tests and 274 kernel
tests passed, 3 kernel tests skipped, with Metal shader validation enabled.
Raw samples and native reports are `apple-m5-pro-20c_shared_scales_*_20260928.jsonl`
and `apple-m5-pro-20c_fixed_shared_scales_20260928.jsonl` in the results directory.

Repack with `tools/pack_weights.py --scale-placement block`; add
`--no-share-scales` for a controlled old-layout pack. Existing pack directories
are not silently rewritten.

A separate attention experiment reduced v3's reduction storage to one component
at a time, matching MLX's smaller allocation but adding barriers. It regressed
T=1 by about 1 µs and T=4 short-context by about 4 µs. Transposing the full buffer
also regressed. Neither change is retained: smaller allocation alone is not a
latency result.


## Four-token attention and GDN convolution-state ownership

For D=128, two query heads per KV head and compiled T=4, matrix attention now
uses eight query rows. It processes 32 keys per tile up to 256 active context
positions, then 64. Core and merge select the same chunk size; workspace capacity
covers the smaller chunks. Separate query and score arrays remove their reuse
barrier, and the final scratch barrier runs only when a threadgroup has another
block. Other geometries and compiled token counts retain the previous path.

Seven alternating pairs, 32 steps each, against ac19b06:

| T | Context | Before µs/layer | After µs/layer |
|---:|---:|---:|---:|
| 4 | 32 | 78.38 | 74.89 |
| 4 | 128 | 81.45 | 76.68 |
| 4 | 252 | 82.46 | 77.51 |
| 4 | 512 | 85.28 | 84.64 |
| 4 | 1024 | 103.68 | 102.61 |

This trades more native code/scratch for less padded matrix work: the exact
exported core retains 72 registers, changes main code 8,418 → 17,128 bytes,
scratch 112 → 208 bytes and threadgroup storage 22,528 → 21,504 bytes.
These metadata fields do not establish spills, occupancy or instruction counts.
Earlier mixed 8/16-query variants and extending selection to T=6/8 had regressions
and were rejected. Metal validation passed 71 attention and compiler contract
tests, including runtime lengths, context boundaries, gates, inactive rows,
cache contents and persistent-group scratch reuse. Matching fixed-chunk kernels
produce byte-identical outputs.

Prepared GDN's final-token preparation groups now copy the final convolution
window into the opposite state slot. Query/key writes have one owner per key
head; value writes have one per value head. The recurrence no longer carries
those address calculations. The emitter declares the preparation's state writes,
so existing producer/consumer barriers cover the transfer. Single-slot and commit
paths keep their previous ownership.

The exact native recurrence drops from 69 to 60 registers and 5,886 to 1,848
main-code bytes, with zero scratch. Preparation rises from 47 to 48 registers
and 8,762 to 10,298 bytes, also without scratch. Seven alternating pairs across
18 distinct GDN layers improve T=4 201.11 → 198.97 µs, T=6 202.74 → 201.43 µs,
and T=8 204.70 → 202.79 µs. Every pair wins; all layer outputs, convolution
state and recurrent state are byte-identical. Forty kernel/contract/real-model
rollback tests passed with Metal validation; the new write-ownership contract
also passes. A run overlapping unrelated native compilation was discarded;
the committed timing samples come from the clean repeat.

Fresh paired MLX checks still fail the overall target:

| Layer kind | T | Context | MPK µs/layer | MLX µs/layer |
|---|---:|---:|---:|---:|
| INT4 attention | 4 | 128 | 76.10 | 71.67 |
| INT4 attention | 4 | 1024 | 101.29 | 95.14 |
| BF16 GDN | 4 | 128 | 198.66 | 182.98 |
| BF16 GDN | 6 | 128 | 200.99 | 183.15 |
| BF16 GDN | 8 | 128 | 203.59 | 188.38 |

These use seven repetitions of 48 steps and the same seeded BF16 inputs and
prefix. Minimum same-input per-layer cosine exceeds 0.99994. As elsewhere,
streaming stack means do not prove every individual layer wins; #113 and the
independently observed session-reuse correctness issue #123 remain open.
Raw samples and native metadata are the `adaptive_attention`, `gdn_conv_store`,
`fixed_adaptive_attention`, `fixed_gdn_conv_store`, and `adaptive_and_conv_native`
20260928 result files.


## Fusing the prepared recurrence and gated normalization

Prepared DK=DV=128 heads at compiled T=4/6/8 now use a combined recurrence and
normalization dispatch when there are at least 16 value heads and the read-out
has exactly one consumer. Each 1,024-thread group owns all 128 state columns of
one head. It publishes the FP32 read-out into 4 KB of threadgroup memory, then
uses the standalone norm's lane mapping and rounding order. The compiler defers
the recurrence until its gate projection is available; its declared recurrent
state and output writes preserve downstream barriers. Commit and other geometries
retain their separate kernels.

The exact native fused kernel uses 63 registers, zero scratch, 4,096 threadgroup
bytes and 4,370 main-code bytes; the preceding recurrence alone used 60 registers,
zero scratch/shared storage and 1,848 bytes. This combines two dispatches without
changing the recurrence or norm arithmetic. An eight-column fused variant lost
at T=8 and was rejected.

Seven alternating production A/B pairs of 32 steps against 483a8e5:

| T | Before µs/layer | Fused µs/layer |
|---:|---:|---:|
| 4 | 198.87 | 197.17 |
| 6 | 200.91 | 200.04 |
| 8 | 203.79 | 202.87 |

Every pair improves. Outputs of all 18 layers and both kinds of state buffer are
byte-identical. The subsequent paired MLX gate (7 × 48 steps) remains open:
T=4 196.12/182.66 µs, T=6 199.11/183.01 µs, T=8 202.66/187.98 µs (MPK/MLX).
Different absolute values across runs are not attributed to code changes.

Metal validation passed 51 kernel/contract checks, including ten new exact
fusion comparisons with filled state, continuation, zero/partial/full active
lengths, separate scalar projections, output permutation, done predicates and
multi-pass scratch reuse. Another 26 compiler/lowering checks passed, including
fused selection, gate dependencies, declared state writes and output barriers.
Raw samples and native metadata are the `gdn_fused_norm_ab`,
`fixed_gdn_fused_norm` and `gdn_fused_norm_native` 20260928 result files.

Further rejected experiments: delayed GDN query loads and vectorized float4
state updates lost at T=8; smaller BF16 projection threadgroups and wider/narrower
lane/prefetch combinations did not improve the layer. Attention first-key
peeling and delaying value loads also lost. A 128-key matrix-attention tile
exceeded the device's 32 KB threadgroup-memory limit and was not benchmarked.

The real-model prefill layer oracle, twice-repeated 48-token greedy golden,
sampling reproducibility check and speculative rollback golden also pass with
Metal validation (four tests). The previously reported long-prompt session-reuse
case in #123 was not part of this run and remains unresolved.


## Loading only the affine scale pairs a lane needs

The shared-scale address change exposed another code-generation cost: a lane's
four- or eight-byte affine scale run was still loaded through a 16-byte vector
and indexed using its position inside that vector. GEMV and matrix fill now load
only those one or two uint scale/bias pairs, with zero-based register indexing.
Inline scales, other scale formats and larger runs retain the previous loader.
The pack format, decoded values and accumulation order are unchanged.

Exact production native code confirms smaller allocation and code, without
scratch changes:

| Specialization | Registers before → after | Main bytes before → after |
|---|---:|---:|
| T=1 QKV | 84 → 80 | 4,908 → 4,756 |
| T=1 gate/up | 95 → 90 | 6,530 → 6,350 |
| T=4 QKV | 71 → 67 | 3,808 → 3,652 |
| T=4 output projection | 69 → 65 | 5,104 → 4,954 |
| T=4 gate/up | 85 → 85 | 8,406 → 8,268 |

All have zero scratch before and after. Seven alternating production A/B pairs,
32 steps each, against 4afe870 improve every paired sample:

| T | Context | Before µs/layer | After µs/layer |
|---:|---:|---:|---:|
| 1 | 128 | 59.04 | 58.35 |
| 1 | 1024 | 71.54 | 70.94 |
| 4 | 128 | 76.72 | 74.71 |
| 4 | 1024 | 101.98 | 101.47 |
| 6 | 128 | 82.71 | 81.94 |
| 8 | 128 | 83.73 | 83.16 |
| 8 | 1024 | 108.36 | 107.92 |

All 28 layer outputs are byte-identical. Separate runs of 32 generated tokens
after five- and nineteen-token prompts also match exactly under Metal validation.
GEMV, matrix and fused-epilogue kernel suites pass (304 passed, three skipped),
plus ten added checks of K=1536/3072 scale-group boundaries. These include BF16
and F16 scale pairs, shared and unshared layouts, row tails and both scale
placements. An early matrix-only prototype was essentially tied at long context;
the production table above is the retained implementation's paired comparison.

The refreshed MLX gate (seven pairs, 48 steps) is still open:

| T | Context | MPK µs/layer | MLX µs/layer | Faster in every pair |
|---:|---:|---:|---:|---|
| 1 | 128 | 58.40 | 58.05 | False |
| 1 | 1024 | 71.03 | 67.91 | False |
| 4 | 128 | 75.11 | 71.27 | False |
| 4 | 1024 | 100.63 | 94.63 | False |
| 6 | 128 | 81.17 | 87.38 | True |
| 6 | 1024 | 104.77 | 120.06 | True |
| 8 | 128 | 82.16 | 103.18 | True |
| 8 | 1024 | 107.24 | 146.53 | True |

Raw samples/native metadata are the `narrow_affine_scales_ab`,
`fixed_narrow_affine_scales` and `narrow_affine_scales_native` 20260928 files.
Native archives were compiled without shader validation; correctness checks
used validation separately.

Other experiments since the preceding checkpoint were rejected: merging the GDN
main and gate projections costs 1.2–2.6 µs; contiguous key ownership or transposed
recurrent state had no consistent win; head-major/32-key-blocked KV caches did
not improve matrix attention. A minimal four-vector INT4 SIMD projection was
still slower than the matrix path. None is included in the production kernels.


A follow-up control shares weights, state and activation allocations between the
scale-load variants. It also includes a separately constructed identical-source
baseline. Both baseline controls remain slower than the narrow loads in every
measured configuration; control-to-control differences reach 0.58 µs, so the
smallest gains should not be interpreted more precisely than that allocation /
pipeline-construction spread. Outputs were captured immediately after each
variant ran, before a shared output could be overwritten. Shared-buffer
minimum latencies (baseline / narrow / identical-source control), in µs/layer:

| T | Context | Baseline | Narrow | Control |
|---:|---:|---:|---:|---:|
| 1 | 128 | 59.03 | 58.26 | 58.66 |
| 1 | 1024 | 71.57 | 71.00 | 71.23 |
| 4 | 128 | 75.86 | 75.12 | 76.00 |
| 4 | 1024 | 101.81 | 101.48 | 101.78 |
| 6 | 128 | 81.97 | 81.54 | 82.09 |
| 8 | 128 | 83.45 | 82.63 | 83.64 |
| 8 | 1024 | 108.48 | 107.83 | 107.90 |

Raw shared-buffer samples are `narrow_affine_scales_shared_ab_20260928.jsonl`.


## Specializing immutable parameter records

The fixed layer graphs supplied head counts, strides, row ranges, launch widths
and workspace sizes through constant buffers even though those records never
change after compilation. The emitter now exposes those values as compile-time
macros. The macros participate in the pipeline key: different gate/QKV row
ranges and fused GDN norm strides cannot accidentally reuse a specialization.
Values are decoded from the packed record, preserving its exact FP32 rounding.
Active token counts, cache positions, recurrent slots and done predicates remain
runtime inputs. The standalone kernel interfaces remain parameterized.

This is enabled for attention and GDN, INT4/NVFP4 projections, and multi-token
BF16 projections. Single-token BF16 projection specialization was slightly slower
and is excluded. Other formats keep their existing projection path.

Native code from the production mixer specialization, compared with bc4decd:

| Kernel | Main bytes before → after | Registers before → after |
|---|---:|---:|
| D=128 T=1 attention | 6,534 → 4,828 | 52 → 52 |
| D=128 T=4 adaptive matrix attention | 17,128 → 15,854 | 72 → 72 |
| T=4 attention merge | 3,586 → 1,626 | 52 → 51 |
| T=1 GDN recurrence | 16,650 → 15,826 | 94 → 90 |
| Prepared GDN convolution | 10,298 → 9,290 | 48 → 44 |
| Fused GDN recurrence/norm | 4,370 → 4,132 | 63 → 67 |

Scratch and shared-memory sizes are unchanged. These are native binary sizes
and experimental resource metadata, not an instruction disassembly. The timing
improvement despite unchanged (or slightly increased) register allocation points
to work removed by specializing address and control calculations; register count
alone did not predict the gain.

Rejected follow-ups: distributing attention softmax scores across lanes was
slower; fusing normalization into the small BF16 projection also lost. Extending
the BF16 SIMD path to eight vectors increased layer time by 7–19 µs and is not
retained. A combined preparation/recurrence prototype is being investigated
separately; it is not part of this parameter-specialization change.


The combined production A/B against bc4decd uses shared weight/state/activation
buffers, seven alternating pairs of 32 steps, and immediate captures of every
layer output and recurrent/conv state. All captures are byte-identical; every
paired timing improves in this run:

| Format / layer | T | Context | Before µs | After µs |
|---|---:|---:|---:|---:|
| int4 | 1 | 128 | 58.56 | 54.64 |
| int4 | 1 | 1024 | 71.29 | 67.64 |
| int4 | 4 | 128 | 78.24 | 70.92 |
| int4 | 4 | 1024 | 104.20 | 97.04 |
| int4 | 6 | 128 | 83.35 | 77.04 |
| int4 | 8 | 128 | 84.43 | 78.02 |
| int4 | 8 | 1024 | 109.94 | 103.41 |
| bf16 | 1 | 128 | 192.05 | 190.55 |
| bf16 | 4 | 128 | 198.07 | 195.46 |
| bf16 | 6 | 128 | 200.51 | 195.67 |
| bf16 | 8 | 128 | 203.61 | 198.99 |

An additional NVFP4 projection-only A/B, on top of mixer specialization, is
437.63→435.66 µs at T=1, 485.47→476.86 at T=4 and 490.53→481.16 at T=8.
All layer outputs match exactly. These runs are not directly comparable to each
other's absolute baseline times.

Validation: 162 contract tests, 104 attention/GDN/selected real-model checks and
12 projection-tail/partial-token checks pass. GPU checks use Metal validation.
The eight new mixer cases reuse specialized pipelines across changing cache
positions, partial/empty/full steps, recurrent slot changes and done predicates.
The existing long-prompt session-reuse issue #123 remains separate and unresolved.


The refreshed INT4 MLX gate now passes all eight measured configurations, with
MPK faster in every one of seven alternating pairs (48 steps per sample):

| T | Context | MPK µs/layer | MLX µs/layer |
|---:|---:|---:|---:|
| 1 | 128 | 54.58 | 58.11 |
| 1 | 1024 | 67.37 | 68.26 |
| 4 | 128 | 68.60 | 71.38 |
| 4 | 1024 | 93.99 | 94.79 |
| 6 | 128 | 75.10 | 87.62 |
| 6 | 1024 | 98.86 | 120.08 |
| 8 | 128 | 75.92 | 103.16 |
| 8 | 1024 | 101.25 | 146.54 |

These are streaming layer-stack means; individual-layer checks are tracked
separately. BF16 remains open: GDN T=1/4/6/8 is 190.15/195.31/194.91/198.58 µs
versus MLX 172.90/183.19/183.05/188.77 µs. Full-attention BF16 layers still lose
at context 128 for T=1 (157.31/151.62) and T=4 (165.57/162.63); the other six
measured configurations win every pair. Exact 32-token INT4 generations after
five- and nineteen-token prompts also pass under Metal validation.


## Whole-head GDN preparation and narrow byte-scale loads

The GDN recurrence now prepares q/k/v inside its whole-head threadgroup and
shares those values through threadgroup memory. This removes the preparation
dispatch and its device workspace, and extends the prepared recurrence/norm path
to T=1 for the measured 128-dimensional head geometry. Preparation uses the same
helper and arithmetic as the standalone path. The final token writes the opposite
convolution-state slot; the combined dispatch declares both state writes. Commit
replay retains the existing separate path.

Seven alternating pairs of 48 steps against f1fc251, using shared buffers and
immediate captures of all layer outputs and both states:

| T | Before µs/layer | After µs/layer |
|---:|---:|---:|
| 1 | 189.77 | 180.29 |
| 4 | 195.19 | 192.55 |
| 6 | 195.59 | 194.62 |
| 8 | 197.45 | 196.18 |

All captured bytes match. T=1 native allocation falls from 90 registers for the
recurrence plus 20 for normalization to 65 for the combined kernel, with zero
scratch. Shared memory is 2,056 bytes at T=1, scaling to 16,448 at T=8. The T=4
combined kernel uses 67 registers and 8,224 shared bytes. These are resource
measurements, not decoded instruction listings. Merging the main and gate
projections was also retested: it loses at T=1/4 and gives only small mixed gains
at T=6/8, so is not retained.

The short aligned scale loader now also handles byte-scale runs of four or eight
bytes. NVFP4 shared-buffer A/B (five alternating pairs, 24 steps) improves
435.36→432.45 µs at T=1, 475.46→470.51 at T=4 and 480.28→475.65 at T=8, with exact
layer outputs. T=1 QKV scratch decreases 48→32 bytes and fused gate/up scratch
80→48; QKV registers fall 70→69. Matrix kernels remain at 126 registers, and some
scratch allocations increase despite the timing improvement. Scratch alone is
not a performance ranking or proof of register spills.

Public Metal pipeline capture of MLX 0.32.2 identifies NVFP4 T=1 `qmv_fast`
(49 registers, 3,992 native bytes) and its two-row variant (30, 2,218), and T=4
`qmv_wide` with four vectors and 16 K lanes (62, 2,372). All three report zero
scratch and threadgroup memory. The corresponding MLX source streams one quant
group per iteration; MPK's scalar path keeps lane scale arrays and its matrix
path decodes into cooperative tiles. This motivates testing the lifetime and
indexing of temporary arrays, without assuming the metadata proves spills.

The full NVFP4 gate before these byte-scale changes still loses at T=1 for both
contexts (435.30/390.99 and 459.28/408.50 µs, MPK/MLX), and at T=4
(476.56/432.20 and 499.28/491.99). T=6/8 win all measured pairs. The latency target
remains open; streaming means do not establish that every individual layer wins.


Validation for this follow-up: 73 GDN kernel/barrier cases pass, including exact
filled-state continuation, partial/zero/done steps, separate scalar projection,
output permutation and TP=1/3/8. A further 285 projection and real-model cases
pass with 3 skips under Metal validation, including greedy generation, layer
oracles, sampling and speculative rollback. The existing #123 long-prompt reuse
case is not treated as fixed. Four additional K=2048 byte-scale boundary cases pass under Metal validation. Hygiene and whitespace checks pass.

A subsequent isolated T=1 NVFP4 A/B rejected fully unrolling the payload loop
(434.84→523.35 µs) and loading scales directly per quantization group
(434.84→492.29 µs). Unrolling only the two-group loop gives a small mixed change
(433.92 µs minimum) and is not retained. All outputs match. One earlier run
overlapped validation and was discarded; these are the isolated rerun values.


## Smaller NVFP4 live operands

The scalar fused gate/up projection now indexes its gate temporary relative to
its row-split slice. The matrix path uses a 16×128 tile for NVFP4 K≥4096 at TM=8,
with uncached scales, rather than 16×256. These choices preserve the pack layout;
the matrix change alters summation order. Explicit tile choices remain available.

Native metadata directly supports the operand-lifetime hypothesis:

| Specialization | Registers before → after | Scratch bytes before → after | Native bytes before → after |
|---|---:|---:|---:|
| T=1 gate/up | 105 → 93 | 48 → 48 | 13,152 → 8,180 |
| T=4 QKV | 126 → 82 | 96 → 0 | 8,666 → 5,278 |
| T=4 output projection | 126 → 82 | 96 → 0 | 9,266 → 6,002 |
| T=4 gate/up | 126 → 83 | 80 → 0 | 13,104 → 9,564 |
| T=4 down projection | 126 → 90 | 80 → 0 | 10,104 → 6,146 |

These are experimental native metadata fields, not decoded ISA or direct proof
of spilling. MLX's captured scalar NVFP4 kernels still use fewer registers and
zero scratch. The remaining scalar gap therefore warrants further experiments.

Seven alternating shared-buffer pairs of 32 steps against f6ffbf2 measure
433.27→426.62 / 470.80→447.95 / 473.11→451.66 / 475.83→452.97 µs/layer at
T=1/4/6/8, respectively. Every pair improves. T=1 outputs are exact. Matrix
outputs change with accumulation order; the same-input per-layer MLX checks
pass at every configuration (minimum cosine 0.999964).

The separate MLX refresh still leaves the strict target open:

| T | Context | Monolith / MLX µs/layer | Wins every pair |
|---:|---:|---:|---|
| 1 | 128 | 426.95 / 391.19 | no |
| 1 | 1024 | 450.64 / 408.97 | no |
| 4 | 128 | 450.94 / 433.52 | no |
| 4 | 1024 | 473.32 / 492.76 | no (4 of 7) |
| 6 | 128 | 452.44 / 648.69 | yes |
| 6 | 1024 | 489.87 / 719.91 | yes |
| 8 | 128 | 455.11 / 824.58 | yes |
| 8 | 1024 | 495.93 / 921.16 | yes |

T=4/context 1024 has a strong ordering effect: Monolith samples are 473–474 µs
when measured first and 502–505 µs when measured second, while MLX stays at
492–494 µs. No cause has been established; a lower minimum is not a robust win.
Raw paired samples and native reports are in `tools/bench/results` under
`nvfp4_operand_ab`, `nvfp4_operand_native`, and `fixed_nvfp4_operands`.

The runtime now declares residency only for buffers bound by the program's
ICB. It continues owning all buffers needed for session reuse. An individual
INT4 layer-0 T=8/context-128 A/B reduces resource declarations from 80 to 25
and latency from 77.01 to 72.77 µs; all pairs improve with exact outputs. Other
individual-layer samples remain noisy, and full-stack gains are small. This
does not establish the individual-layer latency target.

The first individual INT4 sweep stopped after 193 of 224 cases on layer 24,
T=1/context 1024, with cosine 0.998958 against MLX. Two completed cases also
missed latency by less than 2 µs. Numerical investigation is tracked in #124:
MPK's QKV is closer to an independent FP64 affine projection than MLX's BF16
projection, and MLX with FP32 activations/scales matches the independently
rounded reference exactly. This does not establish full-layer correctness.
The benchmark's optional `--continue-on-oracle-failure` records the failure,
finishes the sweep and still exits nonzero; the default still stops immediately.
No threshold has been relaxed.

Validation for the smaller-operand change: 220 projection/barrier cases pass
(3 skipped), and all seven runtime plus real NVFP4 model tests pass under Metal
shader validation. The model check includes every prefill layer and all 48
golden greedy tokens. The five new scalar cases cover independent row-split
slices and empty steps.


## Keep BF16 activations packed until use

For T=1, K≤4096 BF16 projections without input normalization, or with a fused
SiLU/multiply epilogue, keeping activation words packed until the dot product
reduces native live operands. Norm-fed plain projections retain their hoisted
FP32 activations. The fused gate/up allocation falls from 91 to 46 registers
and from 5,932 to 3,508 native bytes; the 2048-column output projection falls
41→38 registers and the 3584-column down projection 51→49. Scratch remains zero.

Seven alternating pairs of 48 steps with shared buffers measure GDN
180.23→177.65 µs/layer and attention 157.19→154.04 µs/layer. All output bytes
match and every pair improves. This is an incremental gain; both short-context
T=1 targets remain unresolved until the MLX gate is refreshed. Applying the
same policy to NVFP4 was slower (429.16→435.18 µs); it is not enabled there.

Other rejected NVFP4 T=1 experiments: computing paired gate/up row groups
together (427.81→428.24 µs), and decoding each weight at its dot-product use
(426.92→485.47 µs). Both were byte-exact. The smaller register lifetime of an
expression is not sufficient reason to retain it without faster measurements.

BF16 validation: 182 projection and selected real-model cases pass under Metal
shader validation, including prefill-layer oracles, all golden greedy tokens,
sampling reproducibility and rollback. The known long-prompt issue #123 remains
excluded and unresolved.

The continuation sweep completes all 32 configurations for layers 24–27. Every
latency pair wins. Layer 24 at T=1/context 1024 reproduces cosine 0.9989583394;
all other numerical checks pass. The command exits 1 as intended. Combined with
the initial sweep, all 224 unique configurations have now been measured, but
the original two latency misses and issue #124 are still unresolved.
