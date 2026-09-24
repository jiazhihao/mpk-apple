# Implementation plan — megakernel inference engine for Apple silicon

Status: drafted 2026-09-19; M0 updated 2026-09-22 with the M5 Pro measurements; **revised 2026-09-23 to start
building in this repo** — a standalone codebase (MPK/mirage code is copied in with its headers, never depended on;
design D15, §5.13), model-agnostic by construction (D16, §5.14), with **DSpark** instead of the checkpoint's MTP head
for speculative decoding (D10, §5.8, [research note](../docs/research/dspark.md)). Design: [`docs/design/design.md`](../docs/design/design.md). Measured hardware facts:
[`docs/research/apple-gpu-probes.md`](../docs/research/apple-gpu-probes.md). Landscape and reuse notes:
[`docs/research/apple-inference-systems.md`](../docs/research/apple-inference-systems.md).

First target: `nvidia/Qwen3.8-27B-NVFP4`, **batch-1 decode latency**, M3/M4/M5 families, macOS 26+.
Bring-up machines: an M3 Pro, 18-core GPU, 36 GB (21 GB of text weights + state fit in its ~28–30 GB working set; the
only machine on hand that hosts the model) and an M5 Pro, 20-core GPU, **24 GB** (GPU characterization and kernel work
only: its 19 GB working-set limit cannot host the 27B).

## 0. Scope

**In scope (v1).** Text-only decode for the Qwen3.5-hybrid architecture from the NVFP4/FP8 checkpoint; greedy and
stochastic sampling on the GPU; DSpark speculative decoding with a public drafter, entirely on the GPU; an extension
path (model / quant format / op / drafter) proven on two more models and a second target–drafter pair; per-chip
profiles for the M3/M4/M5 parts we can get hands on.

**Explicit non-goals for v1** (decided 2026-09-19, amended 2026-09-23): prefill/TTFT as a gate (prompts run through
decode-shaped steps at T = T_max), multi-request serving, energy targets, the vision tower, multi-Mac parallelism,
iOS/iPadOS, the checkpoint's MTP head and tree verification (both fit the `Drafter` contract; neither is built in v1),
drafter training (a GPU-box dependency, not an engine feature).

**Success metrics** — same machine, same prompt set, paired A/B, min-of-N:

| Metric | Gate |
|---|---|
| Correctness | greedy tokens equal to the reference (HF on dequantized weights) except at exact logit ties; repeated runs bit-identical |
| Plain decode | ≥ **1.10×** the better of MLX / llama.cpp on the same machine; stretch ≥ 80 % of the chip's nominal bandwidth bound (6.9 tok/s on the M3 Pro; 13.7 tok/s on an M5 Pro with enough memory). The survey puts the practical ceiling at 85–90 % and today's engines at 57–80 % on dense models, so this gate is deliberately near the limit of what plain decode can give |
| Speculative decode | ≥ **1.5×** our own plain decode and ≥ llama.cpp's `draft-dspark` decode with the same drafter on the same machine; greedy output token-identical to our non-speculative greedy decode |
| Host cost | < 5 % of one CPU core during generation; **zero** CPU↔GPU synchronizations per token or per speculative round on the critical path |
| Generality | 2nd model (with its own DSpark drafter) with **no** kernel/runtime change — the CI extension test proves the PR touches only `monolith/models/`, `tests/`, `docs/`; 3rd model + 2nd quant format via the documented plugin paths only |

## 1. Milestones

Estimates are engineer-weeks (ew) for one engineer working with coding agents; M1‖M2 and M8‖M9 parallelize.
Critical path: M0 → M1 → M3 → M4 → M5 → M6.

### M0 — Characterize and baseline · 1.5 ew · *partly done*

* [x] Probe suite on the M3 Pro (`probes/`, 13 probes, geometry derived from the GPU core count; reference run in
      `probes/results/`): core model, in-flight limit, atomics handoff, no preemption within a dispatch, sharing at
      dispatch granularity, launch overhead, bandwidth vs access pattern, threadgroup memory, clock SIMD-group,
      in-kernel claim protocol vs dispatch boundaries, inter-op overlap and bus saturation vs cores.
* [x] **Probe suite on the M5 Pro** (2026-09-22; 11 results files, `profiles/apple-m5-pro-20c.json`; verdicts in the
      hardware report §3): the 13 probes with `p6`/`p6b` repeated 4×, plus three new probes — `p12` streaming geometry
      (lane order × load width × loads in flight × occupancy; bus saturation vs cores; overlap with a saturating
      streamer), `p13` real FP8/NVFP4 decode GEMV (R, T, layout, geometry sweeps, CPU-checked), `p14` `matmul2d` on the
      neural accelerators from dequantized tiles. Outcomes that changed the design: intra-block lane order is a profile
      value (D8); one threadgroup per core is a default with an autotuned knob (D4); in-dispatch sharing exists but is
      unreliable and command-buffer blocking is the common case (D6); the bus saturates with 6 of 20 cores and the
      ALU-bound sibling must be encoded first (D14); the accelerator path is real from T ≈ 3–8 (§5.6, §5.8).
* [ ] Run `p12`–`p14` on the M3 Pro (they postdate its run): are "lane order decides bandwidth", "crew geometry =
      parity" and the T-cost curves Apple10-only?
* [ ] Baselines on the M3 Pro: `mlx-lm` (NVFP4 mode and affine 4-bit) and `llama.cpp` (Q4_K_M) on Qwen3.8-27B and on a
      small same-architecture model: tok/s, effective GB/s, CPU utilization, dispatches and command buffers per token.
* [ ] **Drafters.** Fetch the Apache-2.0 DSpark drafters for the target — `DimInfer/Qwen3.8-27B-Dspark-v1` (safetensors
      + GGUF Q8/BF16, trained against the Q4_K_M target) and `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` (1.3 GB,
      MLP/o_proj in NVFP4, trained on-policy against an NVFP4 target) — and `Dogacel/Qwen3-8B-DSpark` for model 2.
      Record configs (layers, block size, tapped target layers, Markov rank, dtypes) and licenses in
      `docs/research/dspark.md`; note that `RadixArk/Qwen3.8-27B-DSpark` carries an "other" license and is not used.
* [ ] **Speculative baseline on the M3 Pro:** llama.cpp `--spec-type draft-dspark` (PR #25173) with the Q8 drafter and
      the Q4_K_M target: accepted length and tok/s per workload (math, code, chat) at `--spec-draft-n-max` 2–7; DFlash's
      MLX backend (`dflash generate mlx`) as a second reference. These are the acceptance figures the verify-length rule
      must beat and the number the M6 gate is measured against.
* [ ] Reference tooling: exact NVFP4/FP8 → BF16 dequantizer; HF golden scripts (adapt
      `mirage/tests/runtime_python/models/qwen38/hf_golden.py`): full goldens for the small model; per-layer goldens
      for the 27B produced layer-streamed (54 GB of BF16 does not fit in 36 GB) or on a larger machine.
* [ ] On-screen frame-pacing check: compositor frame times while command buffers of 8 / 16 / 33 / 66 ms run back to
      back → the default `max_cb_ms`. Moved up: the M5 Pro blocks foreign work for whole command buffers more often
      than the M3 Pro, so this measurement precedes the host pump (M2).
* [ ] **Measure M4.** `./probes/remote_run.sh user@host` (or `./probes/run_all.sh` on the machine itself, ~5 min,
      Command Line Tools only) on a bare-metal M4-family Mac; commit the results files and `profiles/*.json`. M4 / M4 Pro
      are rentable as AWS EC2 Mac dedicated hosts (`mac-m4.metal` $1.23/h, `mac-m4pro.metal` $1.97/h, 24-hour minimum
      ≈ $30 / $47). Virtualized macOS runners are useless here (paravirtual GPU). The hypotheses to test and the
      outcomes that would change the design are in the hardware report §4 (H1–H10, revised after the M5 Pro): chiefly
      `p12` (which lane order streams, cores to saturate, encode order), `p6`/`p6b` ×4 (sharing), `p10`, `p13`/`p14`.

Exit: baseline table (plain and speculative), goldens, drafters on disk with recorded configs, ≥ 1 profile (two
provisional profiles exist: `profiles/`).

### M1 — The GEMV proof · 2 ew · **go/no-go #1**

The largest single-token claim is a kernel-geometry and layout claim. Prove or kill it before building on it.
**First reading from the M5 Pro** (`probes/p13_decode_gemv`, 2026-09-22): a first-cut FP8 GEMV with 16 B loads and
R-row blocks streams 275 GB/s (90 % of nominal) at the crew geometry — at *parity* with the conventional
one-block-per-SIMD-group geometry (277), with the intra-block lane order worth +16 %. The same kernel for NVFP4 is
ALU-bound: 137 GB/s of useful bytes at the crew geometry, 182 with 9 threadgroups per core (59 %) — ~275 G weights/s
either way. So the geometry claim is settled at "no worse" and M1's real problem is the **NVFP4 decode cost**.

* Standalone bench harness (C++ + MSL): `gemv_T` for NVFP4 and FP8-E4M3 in block-lane-major packs, crew geometry
  `cores × 384`, static slices; shapes 17408×5120 (gate/up), 5120×17408 (down), 10240×5120, 6144×5120, 5120×6144,
  12288×5120, 248320×5120 (lm_head); T ∈ {1, 2, 4}.
* [x] Bench harness on the native runtime (#9, `tools/bench/gemv_bench.py`) and the NVFP4 decode study (#10):
  the integer-table decode (V2) takes NVFP4 T = 1 from 51–60 % to 82 % (gate/up), 74 % (down) and 89 % (lm_head) of
  nominal on the M5 Pro; FP8 T = 1 is at 85–95 %. Full tables: `docs/research/gemv-kernel-study.md`.
* Kernel study, in this order: **NVFP4 decode** (done: V2; remaining: scale bytes folded into the payload words for
  K = 17408, a `half`-domain group dot under the numerics gate); intra-block lane
  order per chip (lane-interleaved 16 B on the M5 Pro, either on the M3 Pro); threadgroups per core ∈ {1, 2, 4, 9} as
  an autotuned knob (2–9 win 19–44 % for ALU-heavy variants on the M5 Pro); activation-stripe reuse across R rows × T
  tokens without the T = 8 register collapse seen in `p13`; scale placement (inline vs leading; E4M3 vs pre-decoded
  `half`); R ∈ {4, 8, 16} (R = 32 loses 24 % to static-slice tail quantization at 240 SIMD-groups); FP32 vs mixed
  accumulation; `safe` vs `fast` math.
* Baselines on identical shapes: MLX `quantized_matmul` (nvfp4 and affine-4; `qmv_fast`), llama.cpp `mul_mv`.

Exit gate: NVFP4 T = 1 ≥ **1.10×** MLX's kernel throughput on the same machine (M3 Pro, and the M5 Pro where MLX's
dense 4-bit `qmv` is reported at 266 GB/s), NVFP4 ≥ 80 % of nominal on the M5 Pro, FP8 shapes ≥ **100 GB/s** on the
M3 Pro; outputs within 2 ULP (BF16) of the oracle.

**Go/no-go #1, read on the M5 Pro 2026-09-24** (`docs/research/gemv-kernel-study.md` §3c, issues #9–#12):
FP8 231–291 GB/s (75–95 %) — met. NVFP4 T = 1 with the integer-table decode: 231–274 GB/s (75–89 %) — the 80 %
line is met on 5 of 7 shapes. **Against MLX's own NVFP4 `qmv` (262–286 GB/s, 85–93 %) we are at 0.85–0.96×, not
1.10×: the geometry claim does not hold on this chip; a 4-bit GEMV is a solved problem at ~92 % and the remaining
plain-decode lever is fusion + GPU autonomy, as the design's §2 ledger already sized.** Decision: proceed as the gate's
"if missed" clause says — keep the engine plan, drop the bandwidth claim, keep our kernel (it is within 10 % and
carries the fusions/row scales the program needs) and revisit the last 10 % in M5. A second finding changes M6/M9:
**T = 2–4 needs a SIMD-group-matrix kernel** — MLX's `qmm_t` path stays at 85–91 % at T = 2–4 where our shader-FMA
kernels fall to 45–51 % (FP8) and 33–51 % (NVFP4); with such a kernel a T = 4 verify pass should cost ~1.1× a T = 1
pass instead of the ×1.8–2.7 the profile's `cost_T` table records today. That kernel is the first item of M9's
accelerator work and gates M6's verify-length rule (issue #51 grows to "T ≥ 2", not "T ≥ 5"). *If missed:* keep the engine plan — fusion, GPU autonomy and
speculation stand on their own — drop the bandwidth claim from the design and adopt MLX's GEMV structure.

### M2 — Runtime core and weight packer · 3 ew · parallel with M1

* [x] Runtime core v1 (#16, #17, #18; 2026-09-24): device (IORegistry core count, GPU family), shared and
  `newBufferWithBytesNoCopy` buffers, runtime-compiled libraries, pipelines, timed dispatches; the ICB builder (all
  parameters in buffers, 32-bit offsets checked), the re-encode fallback, the host pump (bounded command buffers,
  `in_flight` ahead, drains the ring after every completion, stops on `StepState.done`), the token ring and the
  `Program` schema (`program.json` v1) with `Engine`. The two-op toy program replays 1,000 self-advancing steps from
  one encode (40 command buffers, every token in order, ICB ≡ re-encode bit-for-bit, early exit after `done`; the pump
  thread busy 7 % of wall at 13 µs steps). Still to come here: the pack loader that binds `manifest.json` slabs,
  `MTLBinaryArchive` caching, the C API.
* C++/ObjC++ runtime: device + residency set; mmap'ed pack loader (`newBufferWithBytesNoCopy`, buffers split under
  `maxBufferLength`); pipeline cache (function constants, `MTLBinaryArchive`); `program.json` loader; **ICB builder**
  (per-op parameter records in a buffer — ICBs have no `setBytes` and 32-bit bind offsets, so weights are addressed by
  64-bit GPU address from the record; barriers only on real dependencies); re-encode fallback path; host pump (a few
  command buffers in flight, each replaying one ICB *range* worth ≤ `max_cb_ms` of work — default ~16–33 ms with a
  display attached, since the measured worst case for another GPU client is a wait for one whole command buffer);
  token ring; `StepState`.
  `nanobind` bindings; small C API. References: tinygrad `runtime/graph/metal.py` (ICB), gpt-oss `context.c`
  (multi-token submission).
* `pack_weights`: safetensors (streaming, per shard) → BLM pack with format plugins (NVFP4, FP8-E4M3, BF16, INT8 for
  Q8-style drafters) in the profile's lane order, and the model's transforms (row-stacking `q|k|v` and
  `in_proj_qkv|a|b`, with the mixer's gate projection either stacked or packed as its own op — design §5.12; `gate/up`
  interleave; partial-RoPE head-dim permutation; `(1+w)` norm weights). The drafter goes through the same packer: its
  5 layers, the feature projection `Wc`, the Markov `W₁`/`W₂` (W₂ in the bias-GEMV layout) and the confidence vector.
* Contract tests (no GPU): pack ↔ checkpoint round trip bit-exact after dequantization; program schema;
  parameter records never alias; arena plan alias-free.

Exit: a two-op toy program replayed for 1,000 self-advancing steps from one encode, tokens drained from the ring with
no `waitUntilCompleted` on the hot path; the full checkpoint and a drafter pack and verify; the repo skeleton of §2 is
in place with the registries, the coverage guard stub and the contract tests running in CI.

### M3 — Kernel library v1 · 4 ew

Block bodies + kernel wrappers, each with a torch oracle and a leaf test:

| Op | Port from / cross-check with | Gate |
|---|---|---|
| `embed`, `rmsnorm_stat`, fused-norm GEMV input | MPK `rmsnorm_v2`, `docs/mpk/decode_linear.md` (γ-fold / `r[m]` scaling) | ≤ 2 ULP BF16 |
| `gemv_T` + fusions (residual epilogue, `gate|up → silu·mul`, output gates, row-stacked outputs) | M1 kernels | ≤ 2 ULP |
| `gqa_decode` (+ q/k norm, partial RoPE, KV append, sigmoid gate) | MPK `gqa_decode_sm100_v2.cuh` (online-softmax state merge); MLX `sdpa_vector.h`, llama.cpp `fa.metal` | max-abs ≤ 1e-3; repeat runs bit-identical |
| `gdn_mixer` (conv+SiLU, L2-norm, gates, delta rule, gated norm) | MPK GDN variant of `kda_fused_recurrent_v2.cuh`, `kda_short_conv_v2.cuh`, `kda_gated_norm_v2.cuh`; MLX gated-delta update | state ≤ 8 ULP FP32, output ≤ 2 ULP; fresh + continuation; T = 1 and T > 1 |
| `lm_head` + argmax / Gumbel-max / top-k / top-p | MPK `argmax_*`, `tasks/common/sampling.cuh`; gpt-oss `sample.metal`, `topk.metal` | exact argmax; distribution tests for sampling |
| `draft_attn` (block queries over injected-context KV + bidirectional block), `feature_proj`, `markov_bias` + argmax + confidence, `verify_select`, `accept_scan` | DeepSpec `modeling/dspark/qwen3/modeling.py`, `markov_head.py`, `eval/dspark/confidence_head.py`; DFlash KV injection; llama.cpp `llama_dspark_markov_bias`; MPK `mtp_verify_strict` | draft tokens identical to the DeepSpec reference in greedy mode; confidences ≤ 1e-3; select rule equal to a Python model of it; accept scan exact |

Then composites on real layer weights vs HF modules (one GDN layer, one attention layer, MLP): cos > 0.999 and bounded
max-abs (MPK's `test_layer_cores.py` bars).

Exit: all leaf + composite gates green.

*Status (2026-09-24).* The torch oracles of every op above except sampling variants and the drafter ops exist as the
layer library's `forward()` (`monolith/nn/`), and the composite bars are already green **for the oracles**
(`tests/layers/`: on the 0.8B's real weights, GDN prefill and continuation within 2.4e-4 of the HF module at scale
0.1, attention within one BF16 ULP, MLP + residual within one ULP). The kernels themselves (#19–#25) are the
remaining M3 work; each kernel test compares against these oracles fed with the pack's own aux tensors.

*Status (2026-09-24, #19/#20 + the greedy half of #23).* `gemv_T` has the residual and `silu·mul` epilogues, the
hoisted norm statistic (`STAT_OUT` partials, free) and the fused scaling (`NORM`); `embed` (raw table or the tied
slab), `rmsnorm_stat`, `norm_apply` and the two-dispatch `argmax` exist, all bound in the op registry and green
against the oracles (≤ 2 ULP; `tests/kernels/`). Measured (gemv-kernel-study.md §3d): fusing the *scaling* into an
ALU-bound GEMV costs 5–19 %, a `norm_apply` dispatch 0–3 %, so the default step program applies the norm as its
own dispatch and hoists only the statistic — the row "the norm never costs a separate dispatch" holds for the
reduction, not for the elementwise scaling. Remaining in M3: `gqa_decode` (#21), `gdn_mixer` (#22), sampling
beyond argmax (#23), the drafter ops (#24) and the composite tests on the GPU path (#25).

### M4 — Compiler and end-to-end decode · 4 ew

* IR (typed graph, symbolic `T`/context, op metadata: reads/writes, block domain, class, cost).
* `nn` module library with the three-method contract (`forward` oracle / `lower` / `load_weights`), model registry
  keyed by HF `architectures[0]`, config dataclasses (adapted from MPK `layers_v2/_base.py`, `models/_registry.py`,
  `configs/`).
* `models/qwen3_5`: structure and weight map adapted from MPK `models/qwen38/modeling.py` (drop TP sharding).
* Passes: canonicalize → fuse → select packs → partition → place barriers → memory plan → emit (`program.json`,
  kernel wrappers, pack manifest). Coverage guard: an IR op without a kernel for the target profile fails the build.
  Dynamic T: every kernel reads `T_this_step` from `StepState` (T_max = 1 + γ); per-T kernel variants are encoded
  back-to-back and predicated (design §5.7).
* Registries (models, layers, formats, ops, drafters, profiles) and the **extension test**: a CI job that fails any
  model PR touching files outside `monolith/models/`, `tests/`, `docs/`.
* `generate` CLI + Python API; tokenizer via HF `tokenizers`.

Exit: (a) small same-architecture model — 48 greedy tokens equal to the HF golden, in CI; (b) 27B-NVFP4 —
`--num-layers-override 4/8` hidden-state gates, then full-model greedy equal to the reference; (c) decode tok/s ≥ the
MLX baseline (parity); (d) host < 5 % of a core, no per-token synchronization.

*Status (2026-09-24).* IR (with states, constants and in-place `updates`), the `nn` library, the registries and
`models/qwen3_5` are in (#26–#28); the model lowers to the design's stage count (5 fused ops per layer + norm
statistics + embed/lm_head/argmax: 172 ops for the 24-layer 0.8B before the fuse pass) and packs from its module
tree (`tools/pack_weights.py --model`, 97 slabs + 135 aux tensors + RoPE tables for the 0.8B). Exit (a) holds on the
**oracle path**: the model oracle reproduces the HF golden's 48 greedy tokens and every layer's hidden state at
cos ≥ 0.9997 (`tests/models/qwen3_5/`); the GPU path needs the M3 kernels and the compiler passes (#29–#32).

### M5 — Performance pass · 3 ew · **go/no-go #2**

Per-op GPU timestamps → a per-token budget (GB streamed, ms, % of bound) → close the gap: fusion completeness,
norm-stat hoisting into producer epilogues, `lm_head` cost (4 % of traffic), barrier count, attention at 8 K / 32 K,
per-op autotune (R, block size), math modes. **Sibling overlap** (design §5.12): emit each mixer's gate projection
(`in_proj_z`; the gate half of `q_proj`) as an un-barriered sibling of the ALU-bound mixer core, both at full crew
geometry; keep it per chip only where the A/B shows a gain (expected ~2–4 % on the M3 Pro; *more* on the M5 Pro, where 6 of
20 cores saturate the bus — but only with the ALU-bound sibling encoded first, a profile rule).

Exit gate: the plain-decode success metric. *If 1.10× is missed but parity holds:* proceed to M6 — speculation does
not depend on it — and record why.

### M6 — DSpark speculative decoding · 4 ew

Design §5.8. Everything on the GPU; the host only drains tokens.

* `spec/dspark/`: the drafter as a `Drafter` module — 5 attention layers on the shared `GQAAttention`/`GatedMLP`
  library with a second KV source (the injected context), mask embeddings, the feature projection `Wc`, the Markov
  head (`W₁`, `W₂`, rank 256) and the confidence head; weight map from the public checkpoints (safetensors) and from
  the llama.cpp GGUF naming (`markov_w1/w2`, `conf_proj`, `dflash.block_size`).
* Step-program ops (M3) wired into the dynamic-T program: feature taps as stage-5 epilogues of the tapped target
  layers, feature append, draft pass at T = γ, `lm_head` at T = γ, γ Markov-bias + argmax + confidence pairs,
  `verify_select` from the confidence chain and the profile's `cost(T)` table, verify pass at T = 1 + L, accept scan
  with GDN/conv checkpoint choice and KV advance; `γ + 1` checkpoint slots.
* Correctness first: greedy speculative decode token-identical to greedy non-speculative decode on 256-token
  generations across the prompt set; rejection sampling for temperature > 0 with distribution tests; drafter block
  identical to the DeepSpec reference.
* Then measurement: accepted-length histograms per workload for each drafter (NVFP4, INT8, BF16); calibration of the
  confidence chain (per-position temperatures, STS) against measured acceptance; the verify-length rule vs fixed L;
  tokens/s vs plain decode and vs the llama.cpp `draft-dspark` baseline on the same machine; the trace shows no CPU
  synchronization inside or between rounds.
* Optional, gated on the numbers: prune the Markov bias to the top-M base logits (exactness argument required);
  an INT8 re-quantization of `W₂` at load; the accelerator verify path on Apple10 for T ≥ 5 (M9).

Exit gate: the speculative-decode success metric (≥ 1.5× our plain decode and ≥ llama.cpp's DSpark decode, greedy
token-identical). *If acceptance is the problem* (drafter trained against a different target quantization): retrain
on-policy with the NeMo AutoModel / SpecForge / DeepSpec recipes on a GPU box — a dependency, not engine work.

### M7 — In-kernel runtime re-evaluation and intra-op stealing · 1.5 ew · time-boxed, off the critical path

On the M3 Pro a dispatch boundary (1.8 µs) beats every in-kernel barrier we built (2.6–5.4 µs), and on the M5 Pro
by a wider margin (1.4 vs 2.0–2.5 / 4.3–4.8 µs; *done* 2026-09-22), so multi-op kernels are not part of the design.
This milestone (a) re-runs `p10`/`p6b` on Max-class parts and on small models, where the ratio could differ, and (b)
adds *own-slice + steal* to ops with uneven blocks (long-context attention, MoE
experts) if per-op traces show tail skew.

Exit: a short written result per chip; stealing enabled only for ops where it gains ≥ 2 %.

### M8 — Generality proof · 3 ew

* Model 2, existing ops only: **Qwen3-8B** (dense) with its public DSpark drafter (`Dogacel/Qwen3-8B-DSpark`) — no
  kernel or runtime edits allowed, the CI extension test enforces it, and the drafter contract is exercised with a
  second target–drafter pair.
* Model 3, new ops (a Qwen3.5-MoE-class model: router + expert GEMV indexed by GPU-resident expert ids) — exercises the
  new-op path and data-dependent indexing inside a static program. The survey shows this is where an overhead-free
  engine has the most headroom (today's engines reach only 36–55 % of the bound on 3B-active MoE).
* Format 2 (MXFP4 or affine INT4 groups) — exercises the format-plugin path.
* Porting guide written from the three logs (time-to-port recorded).

### M9 — M4/M5 family tuning · 3 ew · hardware-dependent

Profiles + autotune on M4 Pro/Max, M5, M5 Pro/Max (Ultra if available); MPP TensorOps block for `T > 1` on M5 —
validated on the M5 Pro by `probes/p14_tensor_ops` (dequantize a [64 × 64] tile into threadgroup memory →
`tensor_inline` → `matmul2d<…, execution_simdgroups<S>>` → cooperative-tensor accumulate; 8 tokens for 1.5× a T = 1
pass in FP8 and NVFP4, 32 tokens for 1.7–1.8×; compiles from the Command Line Tools at MSL 4.0). Remaining: tile
tuning, the cooperative right-input fill (no threadgroup staging; reference: MLX `steel/gemm/nax.h`, `quantized_nax.h`),
and the one pipelining experiment that is M5-only — dequantize tile n+1 on the shader ALUs while the neural
accelerator multiplies tile n (design §5.12); the accelerator **verify path** for T = 1 + L ≥ 5 wired into the
dynamic-T program as the predicated variant; MSL 4.1 on macOS 27 (the M5 Pro here runs 26.5.1). Exit: per-chip
results table next to each chip's bound, including tokens/s with DSpark.

### Backlog (post-v1)

Dedicated prefill path; KV quantization and long-context attention; serving/batching; vision tower; energy
measurements; Swift package / C API hardening; multi-Mac over Thunderbolt-5 RDMA (CPU-driven between dispatches).

## 2. Repository layout

Standalone (design D15): everything below builds and tests from this repo with the Command Line Tools; copied files keep
their license headers and a provenance line, and are listed in `third_party/NOTICE`. Modular (design D16, §5.14): model
names appear only under `monolith/models/`.

```
CLAUDE.md  README.md  LICENSE  third_party/NOTICE          pyproject.toml  CMakeLists.txt  .github/workflows/
docs/design/design.md                     docs/research/{apple-gpu-probes,apple-inference-systems,dspark}.md
plans/implementation-plan.md              probes/ (hardware characterization; p13/p14 = the first real kernels)
profiles/*.json                           per-chip profiles (measured; hand-derived first cut)
monolith/                                 Python package (working codename)
  core/      ir.py dtypes.py shapes.py step_state.py profile.py
  nn/        module.py (Module contract)  embedding.py norm.py linear.py attention.py gdn.py mlp.py lm_head.py sampler.py
  models/    registry.py  qwen3_5/{config,model,weights}.py   qwen3/{…}   (M8)   qwen3_5_moe/{…} (M8)
  formats/   registry.py  nvfp4/ fp8_e4m3/ bf16/ int8/       (unpack → pack, msl decode snippet, oracle)
  ops/       registry.py  gemv.py attention.py gdn.py norm.py embed.py sample.py draft.py serial.py  cost.py
  spec/      drafter.py (Drafter contract)  verify.py accept.py   dspark/{config,model,heads,select,weights}.py
  compiler/  passes/{canonicalize,fuse,select_packs,partition,barriers,memory_plan}.py  emit.py coverage.py autotune.py
  runtime/   __init__.py (nanobind module import), api.py (generate, load, profile)
  generate.py   (CLI)
kernels/                                  MSL block bodies + templates
  common/{simd,nvfp4,fp8,int8,rng,steal}.metal  gemv.metal attention.metal gdn.metal norm.metal embed.metal sample.metal draft.metal
runtime/                                  C++/ObjC++ core: device.mm packs.mm pipelines.mm icb.mm pump.mm ring.mm state.mm trace.mm
  bindings/ (nanobind)  include/monolith.h (small C API)
tools/     pack_weights.py  bench/ (GEMV harness grown from probes/p13, matmul2d from p14)  goldens/  viewer/
tests/     contract/ kernels/ layers/ models/ spec/ runtime/ perf/ extension/
```

## 3. Test strategy

| Tier | Runs on | What |
|---|---|---|
| Contract | any machine, no GPU | pack round-trip, program schema, memory plan, IR passes, registry, format plugins (MPK's `tests/v2_contract` idea: most of the compiler is testable without hardware) |
| Leaf kernels | Apple GPU | each block body vs torch oracle, ULP gates, T = 1 and T > 1, repeat-run bit-identity |
| Composite | Apple GPU | one layer of each kind on real weights vs HF modules (cos > 0.999, max-abs bound) |
| Model | Apple GPU | small-model full goldens in CI; 27B reduced-layer + full greedy match nightly |
| Runtime | Apple GPU | ICB replay ≡ re-encode, self-advancing steps, early exit after `done`, stealing exactly-once where enabled |
| Spec | Apple GPU | drafter block vs the DeepSpec torch reference (greedy tokens identical, confidences ≤ 1e-3); `verify_select` vs a Python model of the rule; 256-token greedy speculative decode identical to non-speculative; accepted-length histograms logged per workload |
| Extension | any machine | a model PR changes nothing outside `monolith/models/`, `tests/`, `docs/` (git-diff check); a registered model with a missing kernel binding fails the build (coverage guard) |
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
| Accept-scan / rollback semantics (the target side of chain verification) | `mtp_verify_strict`, `spec_decode/` | adapt |
| DSpark drafter: block drafter with KV injection, Markov head, confidence head, evaluator | DeepSpec (MIT) `deepspec/modeling/dspark/qwen3/modeling.py`, `markov_head.py`, `eval/dspark/`; DFlash (MIT) `dflash/` incl. its MLX backend | torch reference + oracle; port to `spec/dspark/` and MSL block bodies |
| DSpark in a C++ engine: GGUF tensor naming, Markov-bias kernel, verify loop; confidence scheduling | llama.cpp PR #25173 (MIT); SGLang DSpark scheduler (Apache-2.0) | reference |
| DSpark drafters for Qwen3.8-27B and Qwen3-8B | `DimInfer/Qwen3.8-27B-Dspark-v1`, `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4`, `Dogacel/Qwen3-8B-DSpark` (Apache-2.0) | weights |
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
| NVFP4 GEMV is ALU/load-bound on base/Pro chips | GB/s ≪ FP8 shapes — **measured on the M5 Pro: 59 % of nominal at best vs 90 % for FP8, the decode is the limiter** | wide loads, activation reuse, pre-decoded scales, 16-bit packed decode; accept a lower % of bound on small chips |
| Plain-decode headroom is small for a dense 27B | M5 gate missed with parity held | expected by the survey; proceed to speculation, which is where the multiple is |
| Compute-ICB driver bugs | replay ≠ re-encode, hangs | re-encode fallback (0.16 ms/token, still sync-free); keep ICB use to the documented command set |
| Our command buffers stall other GPU clients (sharing is only *usually* per dispatch) | frame-pacing check; `ImpactingInteractivity` errors | `max_cb_ms` ≤ ~16–33 ms while a display is attached; longer buffers only in a headless profile |
| Firmware behaviour differs on other chips/OS | probe-suite deltas — **measured: the M5 Pro differs from the M3 Pro in the lane order that streams, in sharing granularity and in the encode-order dependence of overlap** | profiles are measured, not assumed (lane order, threadgroups per core, sibling order, `max_cb_ms` are profile values); correctness depends only on documented Metal semantics |
| DSpark acceptance low on our W4A16 target (drafters were trained against Q4_K_M / NVFP4-W4A4 targets) or T > 1 compute-bound | M6 gate missed; accepted length ≪ the llama.cpp baseline | measure first (llama.cpp `draft-dspark` on the M3 Pro); verify-length rule per chip; INT8/NVFP4 drafter; accelerator verify path on M5; retrain on-policy with the public recipes on a GPU box |
| Drafter memory and Markov-head traffic | working set > 30 GB on a 36 GB machine; > 5 % of a round in `markov_bias` | NVFP4 / INT8 drafter; W₂ re-quantized to INT8 at load; top-M pruning of the bias if an exactness bound holds |
| Coupling to MPK creeps back in (imports, naming, build) | a mirage import or submodule appears | design D15: copy with headers + NOTICE, never depend; reviewed per PR |
| Model-specific code leaks into the engine | a model PR touches `compiler/`, `runtime/`, `kernels/` | design D16 + the CI extension test; new ops land as separate `ops/` + `kernels/` PRs |
| W4A16 reference disagrees with NVIDIA's W4A4 runtime | token drift vs Blackwell outputs | our contract is HF-on-dequantized-weights; report accuracy deltas on a small eval set |
| 36 GB headroom (21 GB weights + KV + states + OS) | memory pressure; a 24 GB machine (the M5 Pro on hand) reports a 19 GB working-set limit | no vision tower, capped context, an INT8 or NVFP4 drafter; document a 36 GB minimum; use 24 GB machines for kernels and small models only |
| Baselines improve (MLX/llama.cpp ship faster NVFP4/GDN paths) | re-measured each milestone | claims are relative, same machine, same day |
| Generated-kernel compile time / code size | build > 60 s | shared bodies, function constants, binary archives; ~12 pipelines serve all layers |

## 6. First PRs — building starts here (2026-09-23)

Each PR is small, standalone-buildable, and lands with its tests. Definition of done in brackets.

1. **Skeleton.** `pyproject.toml`, `CMakeLists.txt`, `monolith/` package with `core/`, `nn/module.py`, the registries,
   `spec/drafter.py`, `third_party/NOTICE`, `LICENSE`, CI for the contract tier and the extension test. [`pytest
   tests/contract` passes on a hosted runner; `import monolith` works; no mirage import anywhere.]
2. **Formats + dequantizer + goldens.** `formats/{nvfp4,fp8_e4m3,bf16,int8}` with exact torch oracles; HF golden
   scripts (adapted from MPK) for a small same-architecture model and per-layer goldens for the 27B (layer-streamed);
   `docs/research/dspark.md` filled with the drafter configs. [Pack ↔ checkpoint round trip bit-exact after
   dequantization; goldens checked in for the small model.]
3. **`pack_weights` + BLM packs** in both lane orders, model transforms, drafter tensors. [Round-trip tests; the 27B and
   a drafter pack on the M3 Pro.]
4. **GEMV bench harness + M1 kernels** (grown from `probes/p13`/`p14`): `gemv_T` for NVFP4/FP8/BF16/INT8, the NVFP4
   decode study, MLX/llama.cpp baseline scripts. [M1 gate table for the M3 Pro and the M5 Pro.]
5. **Runtime core** (M2): device, packs, pipelines, ICB builder + re-encode fallback, host pump with `max_cb_ms`, token
   ring, `StepState`, nanobind. [The two-op toy program replayed 1,000 steps; ICB ≡ re-encode.]
6. **Kernel library v1** (M3), one PR per op family with oracle tests; the drafter ops last.
7. **Compiler + Qwen3.8 end-to-end** (M4), then **performance pass** (M5), then **DSpark** (M6) — each behind its gate.

The MPK files copied in PRs 1–3 (model structure, weight map, module contract, registry, goldens, profiler format) keep
their Apache-2.0 headers and get provenance lines; nothing in the tree imports or names mirage.
