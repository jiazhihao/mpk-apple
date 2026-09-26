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
first model package (`monolith/models/qwen3_5`, oracle-verified against the HF golden of the 0.8B). Next: the M3
kernels and the compiler passes. Speculative decoding targets a **DSpark** drafter, not the checkpoint's MTP head.

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

0. **Keep building** — the roadmap issues in order (plan §6 PRs 1–5 are in): the M3 kernels (#19–#25) against the
   layer oracles in `monolith/nn/`, then the compiler passes, CLI and end-to-end gates (#29–#32). Fetch the
   Apache-2.0 DSpark drafters and run the llama.cpp `draft-dspark` baseline on the M3 Pro (plan M0).
1. On the M3 Pro: run `p12`–`p14` (they postdate its run) to learn whether the lane-order, parity and T-cost results
   are Apple10-only. On an M4: run the suite, commit the results, fill the M4 column in the hardware report §1, walk
   H1–H10 in §4, and update the design where a hypothesis fails (D4, D5, D6, D8, D14 are the chip-sensitive decisions).
2. Plan M0 remainder: MLX / llama.cpp baselines for the target model (needs the 36 GB M3 Pro — the 24 GB M5 Pro cannot
   host it), exact NVFP4/FP8 → BF16 dequantizer, HF goldens, the on-screen frame-pacing check.
3. Plan M1 (go/no-go): the NVFP4 decode is the problem (ALU-bound at 59 % of nominal on the M5 Pro; FP8 is at 90 %);
   then the comparison against MLX `qmv` on the same machine.
