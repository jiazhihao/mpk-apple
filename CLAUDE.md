# CLAUDE.md

Context for anyone (human or agent) picking this repo up on another machine. Working codename: **Monolith** — a
placeholder; do not use the MPK or Mirage names for this engine.

## What this is

An LLM inference engine for Apple silicon (M3 / M4 / M5, macOS 26+, Metal 4). First target:
`nvidia/Qwen3.8-27B-NVFP4` (Qwen3.5-hybrid: 48 Gated-DeltaNet + 16 full-attention layers; NVFP4 MLP + `lm_head`, FP8
attention/GDN projections; ~17.6 GB of weights read per decoded token) with a public DSpark drafter for speculative
decoding (`docs/research/dspark.md`). **v1 is judged on batch-1 decode
latency only** — prefill/TTFT, multi-request serving and energy are explicit non-goals for v1. The engine must stay
general: new models, quantization formats and ops come in through plugins, not engine edits. Stack: C++/Objective-C++
runtime, Python front-end and compiler, generated MSL kernels.

Status (2026-09-24): design, plan, surveys and hardware characterization (M3 Pro and M5 Pro) exist, and the engine
is being built as stacked PRs (roadmap issue #1): skeleton + registries, format plugins, `pack_weights`, the GEMV
harness and M1 study, runtime core v1 (ICB + host pump + token ring + StepState), and the layer library with the
first model package (`monolith/models/qwen3_5`), the M3 kernels, and compiler v0: the 0.8B decodes end to end on the
GPU from one replayed encode and reproduces its HF goldens, prompts of any length fed in chunks of `t_max`
(`python -m monolith.generate`); since then: the fuse pass, chunked prefill through the dynamic-T program,
GPU sampling, per-op tracing and autotuning, model 2 (`monolith/models/qwen3`, Qwen3-8B NVFP4, zero engine edits),
the DSpark drafter as a `Drafter` module verified against DeepSpec's reference, the round's kernels + IR lowering
(#24), and the round inside the target's step program (#38: `python -m monolith.generate --drafter …`; greedy
speculative decode is token-identical to plain decode on the 8B and on the hybrid 0.8B, the host idle). Speed on
this chip: parity with plain decode at best on the shader GEMV path (decode-kernels.md §5) — the M6 gate waits for
the T ≥ 2 GEMM path (M9). Sampling with a drafter is exact speculative sampling (#39); the prompt-set measurement (#40, dspark.md §3) puts
the cost-aware rule at 36 ms per token vs 27 plain — a no-go on the shader path, a projected go on the T ≥ 2 GEMM
path (M9). The barrier pass and the sibling overlap (#29, #35: the gate GEMV beside the mixer core) take the 0.8B
from 6.85 to 6.58 ms per token; the ICB barrier flag orders the flagged command behind all before it (measured;
the field is `barrier_before`). #34 closed with a measured no: the attention v2 (kept as a per-profile option)
is not the long-context win, SIMD-group-matrix scoring is (M9); fast math buys 1–2 % and breaks bit-identity,
so safe stays. Format 2 (#47) is built: affine INT4 groups (`formats/int4_affine`, MLX / AWQ / GPTQ) as a plugin —
the MLX 4-bit 0.8B decodes token-identical to its oracle; the port needed a per-group bias hook, the quantized-embedding
gather, ragged lane stripes and a package-declared value adapter (mlx_lm folds `1 +` into the zero-centered norms;
porting-log.md); the porting guide (#48, `docs/porting.md`) closes M8. M9's accelerator GEMM is built (#50,
`kernels/gemm_tile.metal`): the cooperative right-input fill from the pack words streams NVFP4 at 177 GB/s and FP8
at 253 for 8 or 16 tokens — 0.9–1.1× a T = 1 shader pass, 34–49 % above `p14`'s staged tile; at 32 tokens it is
below `p14` (decode-kernels.md §6). #51 wires it into the step program: with the profile's `accelerator: on`
every T > 1 GEMV runs on the tile as the predicated variant above T = 1, and the DSpark round on the 8B goes from
35.8 to 19.9 ms per token on the prompt set (1.80× the shader path, 1.36× plain: math 1.89×, code 1.60×, text
1.24×, chat 0.99×; the step 94 % bus-bound), tokens equal to the golden (dspark.md §3) — the M6 gate is met on
math and code on this chip. Speculative decoding targets a **DSpark** drafter, not the MTP head.
An intermittent model-tier failure (wrong tokens / a hang / an empty generation, never reproducible alone) was three
out-of-bounds stores found with shader validation (#92): the GDN commit pass wrote its read-out through a 16-byte
placeholder, the tile's permute wrote a slab's K into a scratch sized by a narrower input, a drafter appended past
its context cache — fixed, and a `Program` now carries a `context_capacity` the serial ops enforce.

## Read these, in this order

1. `docs/design/design.md` — the design. §0 is the decision table (D1–D14); §7 answers "warp specialization?" (no) and
   "static megakernel?" (static yes, one kernel no).
2. `docs/research/apple-gpu-probes.md` — what was measured on real hardware and what each number implies. §1 is the
   cross-chip table (M3 Pro and M5 Pro filled, M4 empty); §3 is what the M5 Pro confirmed and changed; **§4 is the
   checklist for continuing on an M4** (hypotheses H1–H10 and the outcomes that would change the design).
3. `plans/implementation-plan.md` — milestones M0–M9 with exit gates and go/no-go points.
4. `docs/research/apple-inference-systems.md` — how MLX, llama.cpp and others work; what to reuse; headroom estimates.
5. `docs/research/dspark.md` — the speculative-decoding method we target, the public drafters for our models, their
   cost on our hardware.
6. `docs/porting.md` — adding a model, a format, an op, a drafter or a chip: the contracts as they are in the tree,
   the CI checks, the golden workflow; `docs/research/porting-log.md` is the evidence it was derived from.

## The design in six lines

* The whole generation loop is one GPU-resident **static program**, compiler-generated from the model graph: one Metal
  dispatch per fused op (~330 per step for Qwen3.8 = 5 all-to-all stages × 64 layers), encoded once into an indirect
  command buffer and replayed; all per-step state lives on the GPU; the host only keeps the queue fed.
* **Not** one long kernel: a dispatch is never preempted, and a dispatch boundary is as cheap a barrier as anything
  in-kernel. Keep dispatches sub-millisecond and command buffers to tens of milliseconds.
* No role-specialized SIMD-groups. Every kernel runs with the crew geometry `gpu_cores × 384 threads`
  (one threadgroup per core, 12 SIMD-groups × 32 lockstep lanes).
* Weights are re-laid-out at load time (block-lane-major packs) so a SIMD-group sweeps one contiguous block; the lane
  order inside the block is a per-chip profile value (lane-interleaved 16-byte words on Apple10, where lane-contiguous
  stripes cap at 70 % of the bus; a tie on Apple9).
* The lever past the memory-bandwidth bound is DSpark speculative decoding (block drafter + Markov head + confidence
  head), run entirely on the GPU with the verify length chosen per step from the confidences and the chip's measured
  cost-per-T table.
* Overlap only an ALU-bound op with a bus-bound sibling (un-barriered dispatches at full geometry, the ALU-bound one
  encoded first on Apple10); never pre-stage weights, never hand-partition cores.

## Working rules

* **Standalone repo (design D15).** Copying files or fragments from MPK/mirage, MLX, llama.cpp, tinygrad, DeepSpec,
  DFlash or gpt-oss is fine and encouraged: keep the license header, add a provenance line (repo, path, commit) and a
  `third_party/NOTICE` entry. Never `import mirage`, never add a submodule or build dependency on MPK, never name
  anything MPK/Mirage. The tree must build and test alone with the Command Line Tools.
* **Model-agnostic by construction (design D16, §5.14).** Model names appear only under `monolith/models/<name>/`.
  Models, layers, formats, ops, drafters and chip profiles are reached through registries; the compiler's coverage
  guard fails a build for an op without a kernel; a model PR that touches `compiler/`, `runtime/` or `kernels/` is
  wrong by definition (CI enforces it). New ops and drafters land as their own packages with oracle tests.

* **Measure before claiming.** Evidence tags in the docs: [M] measured by us, [S] Apple spec, [R] third-party report,
  [H] hypothesis. Microsecond-level numbers move by tens of percent between runs — report ranges, draw conclusions only
  where ranges do not overlap, use paired alternating A/B runs and min-of-N.
* **Every GPU loop must be bounded.** A running dispatch cannot be cancelled and cannot be relied on to be preempted
  (never on Apple9, only sometimes on Apple10); an unbounded spin freezes the display and can trip the watchdog. Keep any single dispatch under ~1.5 s in probes, far less in the engine.
* **Every kernel store must be inside its binding, and shader validation is the test for it.** Buffers are separate
  Metal allocations, so a store past one lands in a neighbour — StepState, a params record, an activation — and
  shows up later as a wrong token, a hang or an empty generation that never reproduces alone. After a kernel or
  emitter change run the GPU tiers under `MTL_SHADER_VALIDATION=1 MTL_SHADER_VALIDATION_REPORT_TO_STDERR=1`
  (porting.md §0); a `Program` carries its `context_capacity` and the serial ops stop at it (`error = 2`).
* Correctness may depend only on documented Metal semantics (dispatch ordering, ICB barriers, the MSL memory model).
  Threadgroup→core mapping, in-flight limits and sharing behaviour are per-chip *profile values*, measured by the probes.
* Only bare-metal Macs give meaningful numbers; virtualized macOS (hosted CI runners) exposes a paravirtual GPU.
* Code adapted from other projects keeps its license header and is listed in `third_party/NOTICE` (MPK/Mirage is
  Apache-2.0; MLX, llama.cpp, tinygrad are MIT; gpt-oss Metal is Apache-2.0).
* Numerics contract: weight-only dequantization (W4A16 / W8A16), BF16 residual stream, FP32 accumulators and recurrent
  state. Reference = the HF model run on the dequantized weights. Gates: leaf ops ≤ 2 ULP, layers cos > 0.999, greedy
  tokens equal to the golden, repeated runs bit-identical.

## Probes

`./probes/run_all.sh` builds and runs the 16 probes (~5 min, Xcode Command Line Tools only; shaders compile at runtime,
including the MPP tensor ops of `p14`) and saves `probes/results/<chip>_<cores>c_macOS<ver>_<time>.txt`.
`./probes/remote_run.sh user@host` does the same over SSH. Geometry is derived from the GPU core count (`GPU_CORES=<n>`
overrides). Commit every results file. Measured so far: an M3 Pro (2026-09-19, 13 probes) and an M5 Pro (2026-09-22,
all 16, repeats of `p6`/`p6b`/`p12`); hand-derived profiles are in `profiles/`. `p12`–`p14` have not yet run on the
M3 Pro. `./probes/build/p13_decode_gemv check` (same for `p14`) compiles every kernel variant without dispatching.

## Next steps

0. **Keep building** — what is left on this machine: the autotuner-at-install profile writer (#49), the staged
   multi-SIMD-group tile for T ≥ 32 and the K-split for the down projection (decode-kernels.md §6), the attention
   core's SIMD-group-matrix scoring for the long-context rows. Intra-op stealing (#44) is built, measured and off by
   default (decode-kernels.md §7). The 27B items (#36, #40's M3 Pro rows, #46 — the smallest MoE checkpoint in a
   format we read, `nvidia/Qwen3-30B-A3B-NVFP4`, is ~18.5 GB resident against this machine's 19.07 GB GPU working
   set), the M3 Pro / M4 rows of the A/B tables (#2, #3, #5, #7, #8, #49), #42 (a GPU box), #43 (Max-class parts)
   and #52 (macOS 27) need machines this one is not.
1. On the M3 Pro: run `p12`–`p14` (they postdate its run) to learn whether the lane-order, parity and T-cost results
   are Apple10-only. On an M4: run the suite, commit the results, fill the M4 column in the hardware report §1, walk
   H1–H10 in §4, and update the design where a hypothesis fails (D4, D5, D6, D8, D14 are the chip-sensitive decisions).
2. Plan M0 remainder: MLX / llama.cpp baselines for the target model (needs the 36 GB M3 Pro — the 24 GB M5 Pro cannot
   host it), exact NVFP4/FP8 → BF16 dequantizer, HF goldens, the on-screen frame-pacing check.
3. Plan M1 (go/no-go): the NVFP4 decode is the problem (ALU-bound at 59 % of nominal on the M5 Pro; FP8 is at 90 %);
   then the comparison against MLX `qmv` on the same machine.
