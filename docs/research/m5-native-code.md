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


## Selective BF16 parameter specialization and numerical audit

After shortening the BF16 activation lifetime, immutable-parameter specialization
is beneficial on that path too. Plain norm-fed T=1 projections retain live
parameter records; T=1 BF16 projections with packed activation operands now use
the existing specialization mechanism. Seven alternating pairs measure GDN
177.62→176.58 µs/layer (six pairs improve) and attention 154.08→153.00 (all pairs
improve), with exact outputs. Native gate/up allocation increases 46→49 registers
while some down-projection specializations fall 49→47; code size changes are
small. Resource counts alone would not have predicted the measured result.
All 207 selected compiler, projection and real-model checks pass with Metal
validation, including greedy tokens, layer oracles and sampling.

The MLX refresh before this last ~1 µs improvement still misses GDN at every
configuration: T=1/4/6/8 at context 128 is 177.01/192.88/194.17/196.35 µs against
173.28/183.32/183.35/188.85. BF16 attention still loses at short context for T=1
(154.04/152.18) and T=4 (165.46/162.99); all six other cases win every pair.
The strict target remains open.

The two original isolated INT4 latency misses were rerun with nine alternating
pairs of 64 steps after bound-resource residency was introduced. Layer 0,
T=8/context 128 measures 72.34/128.96 µs and wins every pair. Layer 3,
T=1/context 128 measures 55.23/129.08, but one pair narrowly loses. Its GPU time
stays near 51 µs while wall latency ranges 55–130 µs, demonstrating host/queue
variability. The original failed samples remain committed rather than replaced.

The independent full-layer audit resolves the accuracy question for issue #124:
with the same BF16 input and KV prefix, MPK versus the declared CPU contract
oracle has cosine **0.999966** with BF16 dequantized weights and **0.999978** with
FP32 dequantized weights. MLX versus those references has **0.998928/0.998935**.
Thus MPK passes the declared 0.999 layer contract in this case; the failed
direct MPK/MLX comparison is not evidence of an MPK accuracy regression.
`tools/bench/layer_oracle_audit.py` reproduces both references without timing.
The timing benchmark still records/rejects the direct MLX mismatch; no threshold
or projection arithmetic has been changed to conceal it.


## Multi-token GDN overlap and native BF16 operands

The whole-head recurrence/norm fusion remains faster at compiled T=1. At
T=4/6/8, separating the final gated norm restores useful overlap between the
recurrence and the gate projection. Preparation remains local to a threadgroup:
16 groups cover four columns each at T=4; 32 groups cover two columns at T=6 or
four at T=8. Split heads have exactly one convolution-state writer, disjoint
recurrent-state columns, and a shared-memory barrier before preparation storage
is reused for another token pass. Gated normalization retains its dependency
on both recurrence and gate outputs.

The BF16 matrix fill now copies stored BF16 bits directly into its cooperative
operand, avoiding an FP32 expansion and conversion back. The BF16 format plugin
explicitly declares this storage type; other eight-value formats do not inherit
that interpretation. Same-day seven-pair A/B versus 5115d65, 48 replays of the
18 distinct GDN layers, context 128:

| T | Before → after µs/layer | Exact outputs and states |
|---:|---:|---|
| 1 | 176.26 → 176.27 | yes |
| 4 | 192.21 → 190.92 | yes |
| 6 | 193.53 → 189.41 | yes |
| 8 | 195.97 → 191.61 | yes |

T=6/8 improve in every pair. T=4 has two noisy slower candidate samples; raw
samples are retained. This is an improvement against the previous implementation,
not a claim that the MLX target is met. Kernel/barrier validation passes 234 cases
with three skips under Metal shader validation. New exact-equivalence checks
cover split heads, repeated token passes, shared key heads, separate scalar
projections, filled initial state, continuation, partial/empty steps and done.
All four selected real-model tests also pass under shader validation: per-layer
oracles, 48 golden tokens twice, sampling, and the rejected-draft rollback path.

Captured MLX 0.32.2 BF16 T=1 projections use 30 registers and 2,246–2,258 bytes
of code for row-parallel variants, or 44 registers/2,602 bytes with a K reduction.
Its recurrence uses 33 registers and 912 bytes. These are native compiler
metadata and code sizes, not disassembled instruction listings. The differing
fusion boundaries mean they cannot be compared directly with one MPK kernel's
latency. A compact row-register BF16 prototype passed memory/output checks but
was slower for the whole layer and was rejected.

A Metal 4 submission prototype reused the identical ICB and kernels with explicit
buffer/ICB/pipeline residency and queue/encoder barriers. Corrected toy replay
matched 1,000-step state and tokens. The eight BF16 A/B cases showed lower host
CPU work but no useful wall-latency improvement: GDN T=1 176.10/176.22 µs,
T=4 191.90/192.00; attention T=1 152.98/153.40 and T=8 163.12/164.55. The prototype
was removed. Shader instrumentation also crashed in Metal's report-decoding path;
no production correctness claim relies on that experimental submission path.

## Two output rows per SIMD group at T=1

A format/geometry specialization streams one NVFP4 quantization group per lane,
reusing four activation vectors across two output rows. Two reduction chunks
are unrolled; four chunks are slower. It consumes the existing matrix input
permutation and both inline and block scale placements, without changing the
packed weights. For compiled T=1, interleaved NVFP4 packs with K≥4096 divisible
by 1024 and R=8/16 use this path. Other token counts retain their existing path.

The native projection specializations report 53–58 registers, zero scratch and
5,042–6,330 code bytes. The gate projection is 58 registers / zero scratch /
6,296 bytes, versus the preceding scalar kernel's 93 / 48 / 8,180. This is
compiler metadata, not a decoded M5 instruction listing. An isolated decode
A/B rejects sign-bit half construction and paired-half conversion: 448.6 and
436.0 µs/layer versus 410.7 for the existing magnitude/select decoding.

The same-input MLX gate passes numerically at both contexts but still loses
latency (seven alternating pairs, 32 steps, all 36 distinct layers streamed):

| T | Context | MPK µs/layer | MLX µs/layer | Minimum layer cosine |
|---:|---:|---:|---:|---:|
| 1 | 128 | 410.78 | 391.13 | 0.999977 |
| 1 | 1024 | 434.23 | 408.77 | 0.999955 |

BF16 uses a related two-row specialization, folding RMS normalization into the
activation loads when needed. The measured geometry is interleaved R=8/16,
K≥1024 divisible by 256, compiled T=1. Normalized inputs retain their natural
order; other inputs retain producer-fused permutation. The output epilogues
preserve residual rounding, SiLU/multiply, permutation and statistics. Statistics
use separate scratch from the gate/up exchange so writers cannot race readers.
Separate five-pair kind A/Bs measure GDN 176.56→175.83 µs and attention 153.22→150.07 µs.
The all-layer T=1 streaming comparison still loses to MLX, 168.78/166.48 µs at
context 128 and 173.64/169.03 at 1024. Both numerical comparisons pass.

Each new kernel passes 19 tests under Metal validation, including independent
FP64 dot products, partial final blocks and nonzero row starts, inline/block
scales or fused normalization, every row-count source, inactive/done/range
predicates, output permutation, both output dtypes, residual rounding and fused
statistics. The BF16 run also passes 13 barrier contracts. Raw results have
`nvfp4_rows_native`, `fixed_nvfp4_rows`, `fixed_bf16_rows`, and `bf16_rows_ab`
20260928 suffixes. These changes reduce the gap; they do not close the latency gate.

All five selected real-model tests also pass with Metal validation: NVFP4
prefill layers and the 48-token greedy golden, BF16 per-layer oracle, greedy
golden twice, sampling reproducibility, and speculative rollback. The known
long-prompt session-reuse issue #123 remains outside this passing selection.

## Single-token projection/convolution fusion

A sole GDN consumer can receive the convolution and SiLU directly from its
BF16 input projection at compiled T=1. The projection preserves the raw BF16
values in the opposite convolution-state slot and keeps the reference tap order,
BF16 convolution boundary, and BF16 activation boundary. Other projection rows
(scalar gates) remain raw. The recurrence consumes the activated intermediate
and no longer writes the convolution state. Speculative programs retain raw
projections for commit recomputation; multi-token programs retain their existing
path. Selection follows format, geometry, consumer ownership and state layout,
without model-name checks. The interpretation is local to one compilation.

Seven alternating A/B pairs over the 18 distinct GDN layers measure
174.94→172.76 µs/layer; captured outputs and both state buffers are byte-exact.
Two-way unrolling of the BF16 row reduction also saves approximately 0.8 µs in
the preceding paired experiment. Direct physical input addressing in the NVFP4
row kernel saves 410.86→408.13 µs across all seven pairs, with exact outputs.
The optional one-row BF16 geometry was tested but is not selected in production.

The final same-input MLX refresh (9 pairs × 48 replays) gives:

| T | Context | MPK µs/layer | MLX µs/layer | Every pair faster |
|---:|---:|---:|---:|---|
| 1 | 128 | 172.26 | 173.09 | yes |
| 1 | 1024 | 172.61 | 173.15 | yes |
| 4 | 128 | 190.89 | 183.05 | no |
| 4 | 1024 | 191.18 | 183.04 | no |
| 6 | 128 | 190.37 | 183.31 | no |
| 6 | 1024 | 190.24 | 183.65 | no |
| 8 | 128 | 192.06 | 188.35 | no |
| 8 | 1024 | 192.08 | 188.62 | no |

All numerical gates pass. These are streaming stack means per layer, not proof
that each isolated layer passes. T=1 wins are narrow; the full target stays open.
Validation passes 143 kernel/compiler checks under Metal instrumentation, plus
five selected real-model checks covering layer oracles, greedy continuation,
sampling and speculative rollback. The compiler test verifies state bindings,
write ownership, barriers and reuse of the same IR across token counts and
speculative/non-speculative compilation. The known #123 exclusion is unchanged.

Rejected experiments are retained in the results: NVFP4 normalization inside
the projection slows 409.25→422.33 µs even with exact outputs. Moving GDN decay
outside its dot products or computing readout independently of the state update
is neutral/slower at T=4/6 and offers no consistent T=8 gain. A T=4 projection/
convolution prototype was also slower (191.73→192.20 µs, exact state/output).
No such arithmetic or multi-token fusion is included.

Exact final T=1 archives report 37–39 registers and zero scratch for the fused
projection, with 2,718–2,778 code bytes. The recurrence/norm specialization reports
70 registers, zero scratch and 2,056 threadgroup bytes. These are compiler
resource metadata, not decoded instructions; lower whole-layer latency is the
acceptance evidence. The archive report is `gdn_projection_conv_native`.

## Removing repeated recurrence, normalization and attention work

The local GDN geometry launches every state slice and allocates one pass for
all compiled tokens. A `SINGLE_PASS` specialization removes the unnecessary grid
and token-pass loops; the general kernel remains for other geometries and commit.
Partial active lengths and step-slot selection remain live. Seven paired runs
measure T=4 191.62→189.77, T=6 190.28→189.50 and T=8 192.68→192.26 µs/layer,
with every captured output and state byte equal. Recurrence/barrier validation
passes 86 cases, including eight new filled-state continuation comparisons.
All four selected BF16 real-model checks pass under Metal validation.

The four-token BF16 vector projection now builds a normalized input tile once
per threadgroup and shares it across the block's output rows. This removes the
separate input permutation/normalization dispatch without recomputing RMSNorm
for every weight row. The measured K=1024 tile needs 8 KB shared storage. A/B
measurements give GDN 190.71→189.61 and attention 165.45→162.98 µs/layer, with
exact outputs. The analogous NVFP4 experiment is slower and is not selected.
Tests retain independent FP64 dot-product references, R=8/16 tails and ranges,
partial/empty/done predicates, permutation, fused epilogues/statistics and
1/65/517-part normalization reductions.

Single-token attention previously rescaled its running softmax state after
every cached key. Two cached keys now share a maximum and prior-state rescale;
individual score and probability BF16 boundaries remain. Unpaired cached tails
and new causal keys retain the original handling. This changes rounding order,
so exact equality is not claimed. The unchanged attention kernel contracts pass,
including the independent 0.99999 cosine / two-ULP limits. A paired A/B saves
about 6.1 µs for NVFP4 and 8.4 µs for BF16 attention at context 1024. The fresh
NVFP4 MLX comparison still loses (426.83 vs 409.34 µs at context 1024), but its
minimum individual-layer cosine is 0.999939 and passes the unchanged 0.999 gate.
Raw full-stack trajectories drift more than independent layer comparisons;
both measurements are retained rather than conflated.

Further rejected candidates: a 4,096-entry exact BF16 lookup for scaled NVFP4
values slows 408.43→455.50 µs; matrix-projection convolution is neutral at T=6
and slower at T=8 despite exact outputs and state. Neither is in production.

All five selected real-model checks also pass with the shared-normalization and
paired-attention candidates under Metal validation. A subsequent recurrence
geometry sweep finds two-column slices faster at T=8 (193.22→191.92 µs, exact
outputs and states); T=4/6 retain their previous geometry. The final kernel/
compiler selection passes 241 checks, including 18 attention cached-pair boundary
cases and filled-state continuation for the T=8 slice change.

MLX's allocator uses untracked shared buffers. An isolated MPK experiment tried
untracked read-only file mappings and read-only ICB resource declarations. Its
whole-layer results were mixed: NVFP4 T=1 minima differed by less than 1 µs,
while BF16 T=4 regressed about 1 µs. Distinct mappings also exposed repeatable
run-order/cache effects in the NVFP4 samples. The runtime experiment was removed
and the original native module rebuilt; no hazard-tracking change is retained.

The V3 query normalization/RoPE calculation was repeated by every key group.
Sharing one prepared query per threadgroup and selecting a single-block body
for the compiler's one-block-per-threadgroup grid saves another ~1 µs: paired
minima are NVFP4 408.79→407.65 / 427.29→426.48 and BF16 attention
149.95→148.60 / 162.93→161.64 at context 128/1024. All captures are exact.
The general grid-stride body remains available; the separate query workspace
keeps query reads independent of the later softmax fold writes.

The compact scale region now also covers aligned, interleaved 24-byte NVFP4
scale runs. For K=12288, each lane-row payload stays 192 bytes instead of being
padded to 224 bytes with inline scales. The block includes the existing 16-byte
over-fetch margin. Old packs remain readable; obtaining this layout requires
repacking with `--scale-placement block`. A byte-preserving repack measures
T=1/4/6/8 408.85/450.77/452.54/454.92→406.77/446.21/450.10/450.23 µs/layer.
Distinct weight mappings cause clear run-order/cache effects in the original
samples, so all samples are retained and subsequent MLX gates use the new pack.
Widening the narrow-scale-load specialization provides no consistent extra gain
and is not included. Ragged scale runs keep their previous policy.

Packing round trips pass all 36 contracts. Exact inline-versus-block matrix
outputs pass R=8/16, K-splits 1/2/8 and partial row blocks, along with the
independent ULP oracle. The long K=12288 unsplit reduction has ~2.65e-6 relative
error on the original N=272 fixture even with the inline layout; this exceeds
an extra 2e-6 assertion used by shorter-reduction tests but is only 0.0014 ULP
at RMS scale. Existing tests and their thresholds are unchanged. Dedicated
long-stripe tests check the declared ULP contract and exact layout equivalence.

Rejected follow-ups: caching NVFP4 scale words increases T=1 latency by ~1.3 µs;
forcing different BF16 output-projection paths costs ~7–11 µs at T=4; sharing
normalization inside the BF16 matrix kernel costs ~24–26 µs at T=6/8 even with
exact outputs. None of these experiments is selected.


The refreshed same-input gate now passes all eight BF16 attention configurations
in every pair: T=1/4/6/8 at context 128 measures MPK
147.79/161.62/164.57/164.29 versus MLX 152.30/162.98/167.66/173.67 µs;
at context 1024, 160.62/184.39/195.44/200.93 versus
191.72/228.36/246.64/268.30. Numerical gates pass throughout.
GDN still trails at T=4/6/8 by ~2–6 µs; T=1 is close enough that the long-context
refresh does not win every pair. NVFP4 T=1 and short-context T=4 remain ~14–15 µs
behind MLX, while T=6/8 win consistently. The long-context T=4 minimum wins,
but one paired sample does not. These stack means do not establish every
individual checkpoint layer's latency.

Exact archive metadata shows the simplified T=1 GDN recurrence/norm at 59
registers (previously 70), zero scratch and 6,384 code bytes. T=4 recurrence is
54 registers / 11,106 bytes; T=6/8 is 50 / 10,726, all with zero scratch.
Shared-normalization BF16 projections use 79–81 registers and 8–8.25 KB shared
storage, without scratch. The new V3 attention reports 44 registers / 5,856
bytes for D=128 and 64 / 9,662 for D=256. Compact NVFP4 row projections report
55–59 registers, 4,956–6,218 bytes and zero scratch. These remain resource
metadata, not decoded M5 instructions. Raw reports are `compact_query_native`.

All five selected real-model checks pass again under Metal validation after the
query-sharing, T=8 geometry and packing changes. The final additional 17 checks
pass exact compact-layout equivalence (including the N=272 error fixture),
partial/full/empty active lengths, cache updates and specialized V3 geometry.
The known #123 exclusion remains unchanged. The selected NVFP4 golden fixture
uses its original inline pack; compact packing additionally passes full fixed-T
MLX layer gates and the exact packing/kernel comparisons above.

## Hoisting addresses and sharing preparation work

NVFP4's 512-column loop step advances whole activation tiles. Hoisting each
lane's permuted input offset out of that loop, and expressing the interleaved
weight address directly as a row stride plus the linear chunk offset, removes
repeated bit arithmetic. The stored pack and arithmetic order stay unchanged.
At T=4, SIMD-groups now divide the four BF16 input norms among themselves rather
than each repeating all four reductions. With equal GDN key/value dimensions,
q/k/v share the convolution body; only q/k take the L2-normalization branch.
Unequal dimensions retain the original preparation code.

A combined nine-pair A/B against 4ce4870 preserves every captured layer output,
convolution state and recurrent state byte. GDN T=1/4/6/8 improves
173.16/189.20/190.41/191.82→172.45/186.55/188.62/190.92 µs/layer. NVFP4 T=1
improves 405.85→403.11 at context 128 and 424.36→421.13 at context 1024;
both NVFP4 cases win every pair. All 193 selected kernel/compiler tests pass
under Metal validation, including filled-state continuation and normalization
with 1/65/517 statistic parts. The previous five real-model checks passed before
these exact-output simplifications; they are not mislabeled as a new model run.

Further experiments remain candidates: BF16 matrix geometry changes save up
to ~3.5 µs at T=6/8 but alter accumulation order. Retesting compact NVFP4 matrix
partials saves ~3.3 µs at T=4 with changed accumulation rounding. Neither is
selected without fresh numerical/model validation. Removing the matrix tile
loop or adding an explicit threadgroup-size cap is neutral/slower and rejected.

Autotuning currently times unspecialized matrix parameters and uses a whole
slab's row count even for a partial projection. A prototype matching production's
constant geometry saves ~1.3–1.6 µs at T=6/8; additionally tuning the smaller row
ranges regresses ~4.4–4.5 µs. More faithful individual-kernel measurements do not
automatically choose the fastest overlapping layer schedule. The production
autotuner is unchanged, and raw choices/timings are retained.

The exhaustive individual-layer sweep of these three exact simplifications
covers all 24 BF16, 36 NVFP4 and 28 INT4 checkpoint layers at T=1/4/6/8 and
context 128/1024: **704/704 latency minima are below MLX** (seven alternating
pairs, 64 replay steps). BF16 wins every pair in 192/192 cases, NVFP4 in 286/288,
and INT4 in 222/224. Worst minimum ratios are 0.91774, 0.97983 and 0.97767,
respectively. Raw `individual_*_hoisted` files retain every sample.

Numerical checks pass 703/704: INT4 layer 24, T=1, context 1024 reproduces the
same direct-reference mismatch tracked in #124 (cosine 0.99895838). The earlier
independent CPU audit favors MPK for that case; no threshold or failed result
has been changed. These individual replays have different cache reuse and host
amortization from the streaming stack. They meet the measured per-layer latency
minimum gate, but do **not** close the streaming GDN/NVFP4 gaps above or establish
an all-pairs win for the four noisy cases.

## Matrix scheduling and scale decoding follow-up

With the hoisted/shared baseline, a six/eight-token BF16 matrix schedule using
K-split 8 for K=1024 plain projections, 4 for gate/up, and 2 for wider output
projections reduces GDN T=6/8 from 189.15/191.25 to 185.49/187.29 µs in paired
whole-stack measurements. The compiler selects it for the measured 16×64
small-BF16 tile family at T=6/8. The independent layer-vs-MLX numerical minimum
is 0.9999834/0.9999861; unchanged threshold 0.999. Fresh streaming gates show
T=8 wins every pair at both context lengths (~187.0 vs ~188.9 µs). T=6 still
trails (~185.1–185.5 vs ~183.4–183.5); changing reduction geometry alone does
not close that gap. Whole-trajectory A/B cosines are lower (~0.998) because
rounding differences accumulate across layers; they are not the independent
layer oracle.

NVFP4 scale decoding can use a half conversion to handle both normal and
subnormal E4M3 magnitudes without a separate exponent-zero branch. Applying
the exact exponent-bias correction in FP32 preserves every finite scale bit,
including signed zeros. Whole-stack captures remain byte-identical; paired
T=1/4 measurements improve 402.85/445.67→402.40/444.31 µs. This follows MLX's
`fp8.h` conversion and has provenance in the source and NOTICE. All 254 finite
byte codes pass an exact GPU-vs-independent-CPU bit comparison. The matrix,
NVFP4-row and barrier suites pass another 214 checks (three existing skips).

Unselected experiments: changing the hoisted NVFP4 loop from unroll-2 to 1/3/4
costs ~4/8/13 µs, despite exact outputs; performing dequantized multiplication
in half costs ~16–17 µs, also exact. Selective projection fast-math changes
outputs for only ~0–1.4 µs benefit and is not selected. Smaller paired NVFP4
threadgroups save less than 1 µs and add statistic-layout complexity. Reducing
BF16 prefetch depth helps GDN by less than 1 µs with little attention benefit;
staging four vectors at a time preserves exact sums but remains a candidate.
Compact NVFP4 partials pass numerical gates, but the refreshed streaming run
still loses short-context T=4 (~448.7 vs ~432.8 µs); they remain unselected.

All five selected real-model checks pass again with the new BF16 schedule and
scale decoder under Metal validation: prefill layer oracles, greedy goldens,
sampling reproducibility and speculative rollback. The known #123 exclusion
remains unchanged. The routing override is confined to interleaved16 weights;
other layouts retain the tuner selection.

## Immutable scale tables and fixed active lengths

The compiler now inspects each projection's actual FP32 row-scale table. A table
whose representations are all identical and finite can become a bit-preserving
Metal constant. Mixed signed zeros, varying scales and nonfinite values retain
the ordinary loads. This is data-driven constant propagation, not an assumption
about a checkpoint or quantization format. It is selected for NVFP4 row/matrix
projections and BF16 matrix projections; the small BF16 SIMD kernel showed no
benefit. Paired captures are byte-identical. NVFP4 T=4 improves 444.57→437.95 µs
with this change alone (`uniform_row_scale_ab` raw results).

Fixed-length programs additionally propagate the active row count into projection
kernels. Dynamic programs keep their runtime lengths and StepState sources. The
standalone specialization helper defaults to retaining runtime lengths. Against
the uniform-scale baseline, seven alternating pairs measure NVFP4 T=1/4 at
399.66/437.18→399.33/433.41 µs and GDN T=4/6 at
187.16/183.55→182.37/182.29 µs. Every captured layer output is byte-identical.
Specializing the separate input permutation offered no further gain and is not
selected.

A fresh independent MLX gate on the selected scale/length changes measures:

| GDN T | Context | Monolith µs/layer | MLX µs/layer | Wins every pair |
|---|---|---:|---:|---|
| 1 | 128 | 171.46 | 173.97 | no |
| 1 | 1024 | 171.64 | 173.57 | yes |
| 4 | 128 | 181.78 | 183.59 | yes |
| 4 | 1024 | 181.99 | 183.39 | yes |
| 6 | 128 | 181.56 | 183.79 | yes |
| 6 | 1024 | 181.64 | 183.93 | no |
| 8 | 128 | 182.99 | 189.01 | yes |
| 8 | 1024 | 183.12 | 189.44 | yes |

All eight numerical gates pass unchanged. All GDN minima now win; timing noise
prevents an all-pairs claim. The mixed 24-layer BF16 stack wins seven of eight
minima; T=1/context 1024 measures 168.50 versus 168.46 µs and remains unresolved.
All eight mixed-stack numerical gates pass. These stack results are distinct
from the earlier 704-case individual-layer sweep.

The NVFP4 refresh before enabling compact scratch still loses T=1 and
short-context T=4. The original refresh mistakenly restricted compact scratch
to TK=256, while this family uses TK=128; its raw results are retained as
`fixed_active_nvfp4_mlx`, and must not be labeled a compact-scratch measurement.
A paired candidate at the correct geometry improves T=4/8 from
433.00/438.61→431.84/436.65 µs. Sharing normalization across four SIMD-groups
only saves another ~0.17 µs at T=4 and regresses T=8, so it remains unselected.
The corrected compiler condition needs its own fresh MLX gate.

Validation before that condition correction: 369 kernel/compiler checks pass
under Metal validation, three existing skips. These include exact loaded versus
constant scale/length outputs with positive, negative and non-unit scales,
partial row ranges, inactive rows, residual/gated epilogues and statistics;
persistent compact reductions, dynamic program isolation and row sources are
also covered. Four repository hygiene checks pass.

Rejected T=1 NVFP4 candidates: four rows per SIMD-group regress 12–19 µs; one
row with unroll-4 saves ~1.2 µs, while unroll-6/8/12 regress 8–12 µs. A tensor
matrix T=1 path remains 22–54 µs slower despite constant active lengths. All
raw samples are retained; these candidates do not change production routing.

### One-row crews and shared normalization

The single-token NVFP4 path now uses one row per SIMD-group with an unroll factor
of four, eight SIMD-groups per threadgroup on plain/residual projections, and a
whole slab block for gate/up so paired values still share a barrier. Its statistic
partial count follows the actual threadgroup count. This geometry is selected
only for fixed one-token, R=16 programs; dynamic variants retain their common
statistic layout. Four neighboring input-permutation SIMD-groups share one norm
fold, preserving the exact reduction order. All four groups belong to the same
token, including inactive/done steps.

Seven paired runs measure 400.13→395.68 µs/layer for the smaller crews; an
additional combination run measures 399.15→394.95→394.34 µs for the original,
one-row, and one-row plus shared-fold versions. All captured trajectory cosines
are 1.0. The dedicated GPU checks compare exact output bits across both scale
placements, partial row ranges, residual/gated outputs and the new statistic
layout; shared folds match the original exactly at 0/1/3/4/7 active rows, done
steps and 1/65/517 input statistic parts.

The integration checks caught a scratch-key mismatch that disconnected fused
permutation producers from their consumers. Keeping the existing key fixes it:
threadgroup geometry changes no output bytes or layout. All 14 barrier contracts
pass after the fix. All 30 new row-group/shared-fold GPU cases and all 27 combined
scale/length/compact-reduction cases pass under shader validation.

The native archives compare the routing/scale-decoder baseline with constant
scales, fixed active counts, compact NVFP4 scratch and one-row crews:

| Projection | Registers before → after | Code bytes before → after | Shared bytes before → after |
|---|---:|---:|---:|
| BF16 T=4 input | 79 → 74 | 3,974 → 3,540 | 8,192 → 8,192 |
| BF16 T=4 gate/up | 83 → 78 | 6,088 → 5,472 | 8,448 → 8,448 |
| BF16 T=6 gate/up | 66 → 66 | 6,080 → 3,520 | 1,536 → 1,536 |
| NVFP4 T=1 input | 54 → 66 | 4,192 → 4,166 | 0 → 0 |
| NVFP4 T=1 gate/up | 54 → 66 | 5,458 → 4,778 | 64 → 64 |
| NVFP4 T=4 input | 83 → 82 | 5,100 → 4,454 | 7,168 → 1,792 |
| NVFP4 T=4 gate/up | 83 → 82 | 9,388 → 6,704 | 7,168 → 1,792 |

Every sampled kernel reports zero scratch. The faster NVFP4 one-row schedule
uses **more** registers, showing why register count alone cannot explain latency.
These remain native code/resource reports, not decoded M5 instruction listings.

Further rejected experiments: specializing mixer active lengths is neutral or
slower; byte loads for NVFP4 scales, later scale decoding, and leader-only output
epilogues all preserve captured bytes but regress ~0.3–0.7 µs. Smaller attention
crews save ~0.8 µs at short context but lose ~5 µs at long context. Four-key
softmax batches save ~1.4 µs at long context and remain a numerical-validation
candidate. None of these candidates changes the production attention policy.

All five selected real-model checks pass again with the scale/length constants,
corrected compact-scratch selection, one-row NVFP4 crews and shared input norms,
under Metal shader validation. This includes per-layer oracles, greedy goldens,
sampling reproducibility and rollback. The existing #123 long-prompt exclusion
is unchanged. The combined row-group suite passes 263 checks after the scratch-key
fix, with three existing skips; the ten scale/compiler contracts also pass.


## Remaining streaming gaps: normalization dispatches

[M] Extending the constant-row-scale specialization to the two-row BF16
single-token projection preserves all captured output bytes. Nine alternating
pairs improve the mixed 24-layer stack from 165.41→165.04 µs at context 128 and
168.78→168.02 µs at context 1024. A fresh independent MLX comparison measures
164.73/167.27 and 167.97/169.61 µs respectively (Monolith/MLX); both numerical
gates pass, and Monolith wins every pair. All 36 loaded/constant-scale kernel
cases pass under Metal shader validation, including the new BF16 row path.

The remaining NVFP4 experiments keep the same fixed token counts and checkpoint
weights. A prototype scale region in physical payload order preserves bytes and
saves about 1.7 µs at T=1/4. Independent MLX numerical checks pass, but T=1 still
loses 393.05/391.57 and 411.08/409.52 µs at short/long context; T=4 short is
433.82/433.54 µs. This is an explicitly experimental, version-99 pack, rejected
by the normal reader; its wrapper implements only the measured row/matrix paths.
It is **not** a supported pack format or a production result.

Pairing gate/up rows within a SIMD-group removes their shared-memory barrier
but regresses 395.11→395.80–400.08 µs despite byte-identical outputs. Direct FP16
sign-bit construction in the FP4 decoder is also exact but regresses
395.74→433.79 µs; changing only the FP8 scale sign is neutral. These source-level
simplifications are rejected rather than assumed to reduce native work.

The more useful lever is the small input-normalization dispatch. Increasing its
SIMD-groups per token from 16 to 64 and reducing gather unrolling from four to
one improves T=1 by ~2.8 µs with byte-identical outputs. At T=4, 64 groups measure
433.87→430.21 µs and 128 groups in pairs measure 429.70 µs. The output layout
and reduction order remain unchanged. Four-key attention batches separately
save ~1.4 µs at long context but almost nothing at short context; eight-key
batches add no useful gain, while sixteen-key batches regress. Attention batch
changes remain candidates pending independent numerical/model validation.

Raw samples: `bf16_t1_uniform_ab`, `bf16_uniform_t1_mlx`,
`nvfp4_payload_scales_ab`, `nvfp4_payload_scales_matrix_ab`, `payload_scales_mlx`,
`nvfp4_gate_paired_ab`, `nvfp4_signbits_ab`, `nvfp4_permute_width_ab`,
`nvfp4_permute_groups_ab`, `nvfp4_permute_t4_ab`, and `nvfp4_attention_batch_ab`
under `tools/bench/results/apple-m5-pro-20c_*_20260928.jsonl`.


### Supported scale layout and ordinary-reader gate

The accepted layout is now opt-in through `--scale-placement block --scale-order
payload`, with manifest version 3, explicit reader guards, complete CPU unpacking,
and readers for the row, matrix, legacy GEMV and embedding paths. Other formats,
ragged/sub-word stripes and inline scales retain lane order. Old manifests remain
readable. Autotuning keys and synthetic packs include scale order. The whole
checkpoint was rebuilt with the normal packer, with unchanged codes/scale values.

The selected normalization geometry is 64 SIMD-groups per row in single-group
threadgroups at fixed T=1, and 128 in pairs at fixed T=4; gathers use unroll one.
Dynamic programs retain their previous policy. The producer/consumer scratch key
is unchanged because the output layout does not change. All 180 expanded
normalization/scale/compiler/barrier checks pass under shader validation;
58 packing/reader checks and 60 initial scale-layout GPU checks also pass.

Fresh ordinary-reader, ordinary-autotuner comparisons (nine alternating pairs,
48 fixed steps, 36 distinct checkpoint layers):

| T | Context | Monolith µs/layer | MLX µs/layer |
|---|---|---:|---:|
| 1 | 128 | 390.23 | 390.81 |
| 1 | 1024 | 409.02 | 409.55 |
| 4 | 128 | 428.10 | 430.41 |
| 4 | 1024 | 450.77 | 490.36 |
| 6 | 128 | 433.59 | 649.94 |
| 6 | 1024 | 472.57 | 727.73 |
| 8 | 128 | 436.03 | 824.19 |
| 8 | 1024 | 478.03 | 921.02 |

All eight minima beat MLX; all independent layer numerical checks pass. T=1 is
still a narrow minimum-time win, not an every-pair or noise-free claim. Raw
`nvfp4_payload_production_mlx` retains host stalls and every timing pair. No
attention-batch arithmetic change is selected: the four-key candidate helps long
context but does not establish a robust short-context gain. The final individual
sweep and real-model checks are recorded below when complete.

All six selected real-model checks pass under Metal validation, including both
legacy and payload-order NVFP4 packs, per-layer prefill oracles, greedy token
goldens, BF16 sampling reproducibility and rejected-step rollback. The existing
#123 long-prompt exclusion remains unchanged. The expanded contract/tile suite
passes 227 checks after making the old cached 256-column geometry test explicit
and separately asserting the measured 128-column default. All three alternate
matrix-tile payload-reader checks pass, bringing that GPU suite to 63 checks.


### Native explanation of the final changes

[M] The fixed T=1 input-normalization kernel shrinks from 2,500 to 1,558 native
code bytes. Its register field remains 35, local scratch stays zero, and shared
memory falls 4→0 bytes. The active row now spans 64 independent threadgroups
instead of four; measured latency improves without a register-count reduction.
This points to dispatch parallelism and shorter gather code, rather than spills.

Payload-order scales reduce the NVFP4 input projection from 66→61 registers and
4,166→3,934 native code bytes; gate/up falls 66→61 and 4,778→4,540. Coalesced
scale addressing both simplifies generated code and improves paired latency.
The rejected sign-bit rewrite goes in the opposite direction: the original
input projection grows 66→72 registers and 4,166→5,834 bytes, consistent with
its measured ~38 µs/layer regression. These are native archive/resource
observations, not decoded M5 instruction counts or a direct occupancy measure.

BF16 constant scales shrink gate/up from 2,778→2,714 bytes and 41→39 registers.
The fused convolution projection shrinks 2,718→2,664 bytes while registers rise
37→39: again, resource counts alone do not determine speed. Every sampled
specialization reports zero local scratch. Raw evidence: `native_normalization`.


### Final exhaustive fixed-token gate (`40a78fd`)

[M] Seven alternating pairs of 64 fixed verification steps cover every layer
at T=1/4/6/8 and contexts 128/1024. All **704/704 individual minimum-latency
comparisons beat MLX**, and **700/704 win every pair**:

| Format | Individual minimum wins | Every-pair wins | Worst Monolith/MLX minimum ratio |
|---|---:|---:|---:|
| BF16 | 192/192 | 192/192 | 0.909986 |
| NVFP4 | 288/288 | 287/288 | 0.956222 |
| INT4 | 224/224 | 221/224 | 0.977543 |

Fresh complete-stack comparisons also pass **24/24 minimum-latency gates** and
all 24 numerical checks. These means are distinct from the individual-layer
measurements. NVFP4 T=1 margins are narrow and timing ranges overlap; the result
is the predefined minimum-of-paired-runs gate on this machine, not an every-run
or universal hardware claim.

The individual direct-MLX numerical check passes 703/704. INT4 layer 24,
T=1/context 1024 still records cosine 0.998958383 (#124); its benchmark exits
nonzero as before. The independent CPU audit already shows Monolith passing
both BF16- and FP32-dequantized contract references (0.999966/0.999978), while
MLX does not (0.998928/0.998935). No threshold or arithmetic workaround was added.

Evidence: `final_layer_summary` and `final_{individual,stack}_{bf16,nvfp4,int4}`
under `tools/bench/results/apple-m5-pro-20c_*_20260928.*`. The implementation
commit is `40a78fd`; subsequent evidence-only commits change no kernel behavior.
The fixed-token latency target is met for the measured suite. Issue #113's
separate speculative-round throughput follow-up is outside this task's scope.
