# Implementation plan — megakernel inference engine for Apple silicon

Status: drafted 2026-09-19. Design: [`docs/design/design.md`](../docs/design/design.md). Measured hardware facts:
[`docs/research/apple-gpu-probes.md`](../docs/research/apple-gpu-probes.md). Landscape and reuse notes:
[`docs/research/apple-inference-systems.md`](../docs/research/apple-inference-systems.md).

First target: `nvidia/Qwen3.8-27B-NVFP4`, **batch-1 decode latency**, M3/M4/M5 families, macOS 26+.
Bring-up machine: M3 Pro, 18-core GPU, 36 GB (21 GB of text weights + state fit in its ~28–30 GB working set).

## 0. Scope

**In scope (v1).** Text-only decode for the Qwen3.5-hybrid architecture from the NVFP4/FP8 checkpoint; greedy and
stochastic sampling on the GPU; MTP speculative decoding; an extension path (model / quant format / op) proven on two
more models; per-chip profiles for the M3/M4/M5 parts we can get hands on.

**Explicit non-goals for v1** (decided 2026-09-19): prefill/TTFT as a gate (prompts run through decode-shaped steps at
T = T_max), multi-request serving, energy targets, the vision tower, multi-Mac parallelism, iOS/iPadOS.

**Success metrics** — same machine, same prompt set, paired A/B, min-of-N:

| Metric | Gate |
|---|---|
| Correctness | greedy tokens equal to the reference (HF on dequantized weights) except at exact logit ties; repeated runs bit-identical |
| Plain decode | ≥ **1.10×** the better of MLX / llama.cpp on the same machine; stretch ≥ 80 % of the chip's nominal bandwidth bound (6.9 tok/s on the M3 Pro). The survey puts the practical ceiling at 85–90 % and today's engines at 57–80 % on dense models, so this gate is deliberately near the limit of what plain decode can give |
| Speculative decode | ≥ **1.5×** our own plain decode, token-identical in greedy mode |
| Host cost | < 5 % of one CPU core during generation; **zero** CPU↔GPU synchronizations per token or per speculative round on the critical path |
| Generality | 2nd model with **no** kernel/runtime change; 3rd model + 2nd quant format via the documented plugin paths only |

## 1. Milestones

Estimates are engineer-weeks (ew) for one engineer working with coding agents; M1‖M2 and M8‖M9 parallelize.
Critical path: M0 → M1 → M3 → M4 → M5 → M6.

### M0 — Characterize and baseline · 1.5 ew · *partly done*

* [x] Probe suite on the M3 Pro (`probes/`, 13 probes, geometry derived from the GPU core count; reference run in
      `probes/results/`): core model, in-flight limit, atomics handoff, no preemption within a dispatch, sharing at
      dispatch granularity, launch overhead, bandwidth vs access pattern, threadgroup memory, clock SIMD-group,
      in-kernel claim protocol vs dispatch boundaries, inter-op overlap and bus saturation vs cores.
* [ ] Baselines on the M3 Pro: `mlx-lm` (NVFP4 mode and affine 4-bit) and `llama.cpp` (Q4_K_M) on Qwen3.8-27B and on a
      small same-architecture model: tok/s, effective GB/s, CPU utilization, dispatches and command buffers per token.
* [ ] Reference tooling: exact NVFP4/FP8 → BF16 dequantizer; HF golden scripts (adapt
      `mirage/tests/runtime_python/models/qwen38/hf_golden.py`): full goldens for the small model; per-layer goldens
      for the 27B produced layer-streamed (54 GB of BF16 does not fit in 36 GB) or on a larger machine.
* [ ] On-screen frame-pacing check: compositor frame times while command buffers of 8 / 16 / 33 / 66 ms run back to
      back → the default `max_cb_ms`.
* [ ] **Measure M4 and M5.** `./probes/remote_run.sh user@host` (or `./probes/run_all.sh` on the machine itself, ~4 min,
      Command Line Tools only) on bare-metal M4-family and M5-family Macs; commit the results files and `profiles/*.json`.
      M4 / M4 Pro are rentable as AWS EC2 Mac dedicated hosts (`mac-m4.metal` $1.23/h, `mac-m4pro.metal` $1.97/h, 24-hour
      minimum ≈ $30 / $47); no bare-metal M5 rental was found as of 2026-09, so M5 needs a physical machine. Virtualized
      macOS runners are useless here (paravirtual GPU). The hypotheses to test and the outcomes that would change the
      design are listed in the hardware report, §3 (H1–H8): chiefly `p10` (dispatch boundary vs in-kernel barrier),
      `p6b` (sharing granularity), `p11` (cores needed to saturate the bus; whether an ALU-bound op still hides inside a
      bus-bound one when bandwidth per core is 12–17 GB/s instead of 8.5). **M4 is next** (the user continues there).

Exit: baseline table, goldens, ≥ 1 profile.

### M1 — The GEMV proof · 2 ew · **go/no-go #1**

The largest single-token claim is a kernel-geometry and layout claim. Prove or kill it before building on it.

* Standalone bench harness (C++ + MSL): `gemv_T` for NVFP4 and FP8-E4M3 in block-lane-major packs, crew geometry
  `cores × 384`, static slices; shapes 17408×5120 (gate/up), 5120×17408 (down), 10240×5120, 6144×5120, 5120×6144,
  12288×5120, 248320×5120 (lm_head); T ∈ {1, 2, 4}.
* Kernel study, in this order: wide packed loads (`uint32/uint64`); activation-stripe reuse across R rows × T tokens;
  scale placement (inline vs leading; E4M3 vs pre-decoded `half`); R ∈ {8, 16, 32}; lane = column stripe vs lane = row;
  FP32 vs mixed accumulation; `safe` vs `fast` math; one 384-thread threadgroup per core vs MLX/llama.cpp-style
  64-thread threadgroups.
* Baselines on identical shapes: MLX `quantized_matmul` (nvfp4 and affine-4; `qmv_fast`), llama.cpp `mul_mv`.

Exit gate: NVFP4 T = 1 ≥ **1.10×** MLX's kernel throughput on the M3 Pro and FP8 shapes ≥ **100 GB/s** effective;
outputs within 2 ULP (BF16) of a torch oracle. *If missed:* keep the engine plan — fusion, GPU autonomy and
speculation stand on their own — drop the bandwidth claim from the design and adopt MLX's GEMV structure.

### M2 — Runtime core and weight packer · 3 ew · parallel with M1

* C++/ObjC++ runtime: device + residency set; mmap'ed pack loader (`newBufferWithBytesNoCopy`, buffers split under
  `maxBufferLength`); pipeline cache (function constants, `MTLBinaryArchive`); `program.json` loader; **ICB builder**
  (per-op parameter records in a buffer — ICBs have no `setBytes` and 32-bit bind offsets, so weights are addressed by
  64-bit GPU address from the record; barriers only on real dependencies); re-encode fallback path; host pump (a few
  command buffers in flight, each replaying one ICB *range* worth ≤ `max_cb_ms` of work — default ~16–33 ms with a
  display attached, since the measured worst case for another GPU client is a wait for one whole command buffer);
  token ring; `StepState`.
  `nanobind` bindings; small C API. References: tinygrad `runtime/graph/metal.py` (ICB), gpt-oss `context.c`
  (multi-token submission).
* `pack_weights`: safetensors (streaming, per shard) → BLM pack with format plugins (NVFP4, FP8-E4M3, BF16) and the
  model's transforms (row-stacking `q|k|v` and `in_proj_qkv|a|b`, with the mixer's gate projection either stacked or
  packed as its own op — design §5.12; `gate/up` interleave; partial-RoPE head-dim permutation; `(1+w)` norm weights).
* Contract tests (no GPU): pack ↔ checkpoint round trip bit-exact after dequantization; program schema;
  parameter records never alias; arena plan alias-free.

Exit: a two-op toy program replayed for 1,000 self-advancing steps from one encode, tokens drained from the ring with
no `waitUntilCompleted` on the hot path; the full checkpoint packs and verifies.

### M3 — Kernel library v1 · 4 ew

Block bodies + kernel wrappers, each with a torch oracle and a leaf test:

| Op | Port from / cross-check with | Gate |
|---|---|---|
| `embed`, `rmsnorm_stat`, fused-norm GEMV input | MPK `rmsnorm_v2`, `docs/mpk/decode_linear.md` (γ-fold / `r[m]` scaling) | ≤ 2 ULP BF16 |
| `gemv_T` + fusions (residual epilogue, `gate|up → silu·mul`, output gates, row-stacked outputs) | M1 kernels | ≤ 2 ULP |
| `gqa_decode` (+ q/k norm, partial RoPE, KV append, sigmoid gate) | MPK `gqa_decode_sm100_v2.cuh` (online-softmax state merge); MLX `sdpa_vector.h`, llama.cpp `fa.metal` | max-abs ≤ 1e-3; repeat runs bit-identical |
| `gdn_mixer` (conv+SiLU, L2-norm, gates, delta rule, gated norm) | MPK GDN variant of `kda_fused_recurrent_v2.cuh`, `kda_short_conv_v2.cuh`, `kda_gated_norm_v2.cuh`; MLX gated-delta update | state ≤ 8 ULP FP32, output ≤ 2 ULP; fresh + continuation; T = 1 and T > 1 |
| `lm_head` + argmax / Gumbel-max / top-k / top-p | MPK `argmax_*`, `tasks/common/sampling.cuh`; gpt-oss `sample.metal`, `topk.metal` | exact argmax; distribution tests for sampling |

Then composites on real layer weights vs HF modules (one GDN layer, one attention layer, MLP): cos > 0.999 and bounded
max-abs (MPK's `test_layer_cores.py` bars).

Exit: all leaf + composite gates green.

### M4 — Compiler and end-to-end decode · 4 ew

* IR (typed graph, symbolic `T`/context, op metadata: reads/writes, block domain, class, cost).
* `nn` module library with the three-method contract (`forward` oracle / `lower` / `load_weights`), model registry
  keyed by HF `architectures[0]`, config dataclasses (adapted from MPK `layers_v2/_base.py`, `models/_registry.py`,
  `configs/`).
* `models/qwen3_5`: structure and weight map adapted from MPK `models/qwen38/modeling.py` (drop TP sharding).
* Passes: canonicalize → fuse → select packs → partition → place barriers → memory plan → emit (`program.json`,
  kernel wrappers, pack manifest). Coverage guard: an IR op without a kernel for the target profile fails the build.
* `generate` CLI + Python API; tokenizer via HF `tokenizers`.

Exit: (a) small same-architecture model — 48 greedy tokens equal to the HF golden, in CI; (b) 27B-NVFP4 —
`--num-layers-override 4/8` hidden-state gates, then full-model greedy equal to the reference; (c) decode tok/s ≥ the
MLX baseline (parity); (d) host < 5 % of a core, no per-token synchronization.

### M5 — Performance pass · 3 ew · **go/no-go #2**

Per-op GPU timestamps → a per-token budget (GB streamed, ms, % of bound) → close the gap: fusion completeness,
norm-stat hoisting into producer epilogues, `lm_head` cost (4 % of traffic), barrier count, attention at 8 K / 32 K,
per-op autotune (R, block size), math modes. **Sibling overlap** (design §5.12): emit each mixer's gate projection
(`in_proj_z`; the gate half of `q_proj`) as an un-barriered sibling of the ALU-bound mixer core, both at full crew
geometry; keep it per chip only where the A/B shows a gain (expected ~2–4 % on the M3 Pro, less on M4/M5 where
bandwidth per core is higher).

Exit gate: the plain-decode success metric. *If 1.10× is missed but parity holds:* proceed to M6 — speculation does
not depend on it — and record why.

### M6 — MTP speculative decoding · 4 ew

MTP module and weights (BF16; optional load-time FP8), `T > 1` kernels tuned, GDN/conv checkpoint slots, KV rollback,
on-GPU strict verify + accept scan, rejection sampling for temperature > 0, dynamic-`T` ops driven by `StepState`,
draft length k chosen per chip by measurement (semantics from MPK `mtp_verify_strict`, `spec_decode/`). Known hazard:
llama.cpp reports MTP speculation as a net loss on an M1 Max — verification cost on compute-poor parts is real.

Exit gate: the speculative-decode success metric; the trace shows no CPU synchronization inside or between rounds.

### M7 — In-kernel runtime re-evaluation and intra-op stealing · 1.5 ew · time-boxed, off the critical path

On the M3 Pro a dispatch boundary (1.8 µs) beats every in-kernel barrier we built (2.6–5.4 µs), so multi-op kernels
are not part of the design. This milestone (a) re-runs `p10`/`p6b` on Max-class and M5 parts and on small models,
where the ratio could differ, and (b) adds *own-slice + steal* to ops with uneven blocks (long-context attention, MoE
experts) if per-op traces show tail skew.

Exit: a short written result per chip; stealing enabled only for ops where it gains ≥ 2 %.

### M8 — Generality proof · 3 ew

* Model 2, existing ops only (a dense Llama/Qwen3-class model): **no kernel or runtime edits allowed** — that is the
  test.
* Model 3, new ops (a Qwen3.5-MoE-class model: router + expert GEMV indexed by GPU-resident expert ids) — exercises the
  new-op path and data-dependent indexing inside a static program. The survey shows this is where an overhead-free
  engine has the most headroom (today's engines reach only 36–55 % of the bound on 3B-active MoE).
* Format 2 (MXFP4 or affine INT4 groups) — exercises the format-plugin path.
* Porting guide written from the three logs (time-to-port recorded).

### M9 — M4/M5 family tuning · 3 ew · hardware-dependent

Profiles + autotune on M4 Pro/Max, M5, M5 Pro/Max (Ultra if available); MPP TensorOps block for `T > 1` on M5
(dequantize → cooperative tensor → `matmul2d<…, execution_simdgroups<N>>`; reference: MLX `steel/gemm/nax.h`,
`quantized_nax.h`), including the one pipelining experiment that is M5-only — dequantize tile n+1 on the shader ALUs
while the neural accelerator multiplies tile n (design §5.12); MSL 4.1 on macOS 27. Exit: per-chip results table next
to each chip's bound.

### Backlog (post-v1)

Dedicated prefill path; KV quantization and long-context attention; serving/batching; vision tower; energy
measurements; Swift package / C API hardening; multi-Mac over Thunderbolt-5 RDMA (CPU-driven between dispatches).

## 2. Repository layout

```
docs/design/design.md                     docs/research/{apple-gpu-probes,apple-inference-systems}.md
plans/implementation-plan.md              probes/                  (hardware characterization, done)
monolith/                                 Python: front-end + compiler   (package name = working codename)
  ir/  nn/  models/{registry.py,qwen3_5/}  formats/{nvfp4,fp8,bf16}.py
  compiler/{passes/,memory_plan.py,cost_model.py,codegen_msl.py}
  profiles/*.json  autotune/  trace/  generate.py
kernels/                                  MSL block bodies + templates
  common/{simd.metal,nvfp4.metal,fp8.metal,rng.metal,steal.metal}
  gemv.metal  attention.metal  gdn.metal  norm.metal  embed.metal  sample.metal  mtp.metal
runtime/                                  C++/ObjC++ core + nanobind bindings + C API (CMake)
tools/{pack_weights.py,bench/,viewer/}    tests/{contract,kernels,layers,models,runtime,perf}/
third_party/NOTICE                        (Apache-2.0 attributions for code adapted from mirage; MIT for MLX/llama.cpp/tinygrad-derived code)
```

## 3. Test strategy

| Tier | Runs on | What |
|---|---|---|
| Contract | any machine, no GPU | pack round-trip, program schema, memory plan, IR passes, registry, format plugins (MPK's `tests/v2_contract` idea: most of the compiler is testable without hardware) |
| Leaf kernels | Apple GPU | each block body vs torch oracle, ULP gates, T = 1 and T > 1, repeat-run bit-identity |
| Composite | Apple GPU | one layer of each kind on real weights vs HF modules (cos > 0.999, max-abs bound) |
| Model | Apple GPU | small-model full goldens in CI; 27B reduced-layer + full greedy match nightly |
| Runtime | Apple GPU | ICB replay ≡ re-encode, self-advancing steps, early exit after `done`, stealing exactly-once where enabled |
| Perf | dedicated machine | paired alternating A/B (MPK `ab.sh` discipline), min-of-N, thermal state logged; tok/s reported with GB/s and % of bound |

CI: hosted macOS runners for the contract tier; a self-hosted Apple-silicon runner for GPU tiers. Profiling is a
compile-time switch only. Every perf claim in a PR carries its A/B table. GPU tests never contain an unbounded loop:
a dispatch is not preemptible.

## 4. Reuse map

| Need | Source | How |
|---|---|---|
| Qwen3.5-hybrid structure, weight names, partial-RoPE permutation, RoPE tables | `mirage/python/mirage/mpk/models/qwen38/{configuration,modeling}.py`, `plans/qwen38-mpk-v2-tp8.md` | adapt (drop TP sharding) |
| Module contract, streaming weight load, registry, configs | `layers_v2/_base.py`, `models/_registry.py`, `configs/*`, `weight_loader.py` | adapt |
| GDN recurrence, short conv, gated norm; CUDA-core GQA decode; sampling | `tasks/blackwell_v2/kda/*`, `gqa_decode_sm100_v2.cuh`, `tasks/common/sampling.cuh` | port algorithms to MSL block bodies |
| In-kernel batch advance, token streaming | `persistent_kernel.cuh::prepare_next_batch`, `docs/mpk/online_output_streaming.md` | re-host in the per-step serial op / token ring |
| MTP verify semantics | `mtp_*` layers, `spec_decode/` | adapt |
| Static-schedule ideas | PR #278: layer-type templates, monotone counters, init-once, compile-time profiling, gate/A-B scripts | design input |
| Goldens and gates | `tests/runtime_python/models/qwen38/*` | adapt |
| Trace format, decoder, viewer | `python/mpkprof`, `tools/mpkv2_viewer` | reuse with a new emitter |
| Quantized GEMV, SDPA decode, gated-delta, M5 TensorOps usage | MLX `quantized.h`, `fp_quantized.h`, `sdpa_vector.h`, `steel/gemm/nax.h` (MIT) | reference + baseline |
| GEMV / flash-attention kernels, barrier placement, GDN fusions | llama.cpp `kernels/mul_mv.metal`, `fa.metal`, `ggml-metal-common.cpp`, `ggml-metal-fusion.cpp` (MIT) | reference + baseline |
| ICB build/replay; multi-token submission + GPU sampling | tinygrad `runtime/graph/metal.py` (MIT); gpt-oss `context.c`, `sample.metal` (Apache-2.0) | reference |

All MPK-derived files keep their Apache-2.0 headers; `third_party/NOTICE` lists origins. No MPK/Mirage naming in the
new engine.

## 5. Risk register

| Risk | Signal | Mitigation / decision point |
|---|---|---|
| Bandwidth advantage does not survive real kernels | M1 gate missed | adopt MLX-style GEMV structure; value rests on fusion, GPU autonomy, speculation |
| NVFP4 GEMV is ALU/load-bound on base/Pro chips | GB/s ≪ FP8 shapes | wide loads, activation reuse, pre-decoded scales; accept a lower % of bound on small chips |
| Plain-decode headroom is small for a dense 27B | M5 gate missed with parity held | expected by the survey; proceed to speculation, which is where the multiple is |
| Compute-ICB driver bugs | replay ≠ re-encode, hangs | re-encode fallback (0.16 ms/token, still sync-free); keep ICB use to the documented command set |
| Our command buffers stall other GPU clients (sharing is only *usually* per dispatch) | frame-pacing check; `ImpactingInteractivity` errors | `max_cb_ms` ≤ ~16–33 ms while a display is attached; longer buffers only in a headless profile |
| Firmware behaviour differs on other chips/OS | probe-suite deltas | profiles are measured, not assumed; correctness depends only on documented Metal semantics |
| MTP acceptance low or T > 1 compute-bound | M6 gate missed | k per chip; FP8 MTP head; TensorOps on M5 |
| W4A16 reference disagrees with NVIDIA's W4A4 runtime | token drift vs Blackwell outputs | our contract is HF-on-dequantized-weights; report accuracy deltas on a small eval set |
| 36 GB headroom (21 GB weights + KV + states + OS) | memory pressure | no vision tower, capped context, optional FP8 MTP; document a 36 GB minimum |
| Baselines improve (MLX/llama.cpp ship faster NVFP4/GDN paths) | re-measured each milestone | claims are relative, same machine, same day |
| Generated-kernel compile time / code size | build > 60 s | shared bodies, function constants, binary archives; ~12 pipelines serve all layers |

## 6. First PRs

1. Repo skeleton, `third_party/NOTICE`, CI for the contract tier; `probes/` + research docs (this commit's content).
2. NVFP4/FP8 exact dequantizer + golden scripts + small-model goldens.
3. `pack_weights` with BLM packs and round-trip tests.
4. GEMV bench harness + first NVFP4/FP8 kernels + MLX/llama.cpp baseline scripts (M1).
5. Runtime core: pack loader, pipeline cache, ICB builder + re-encode fallback, host pump, token ring (M2).
