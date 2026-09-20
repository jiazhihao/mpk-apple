# CLAUDE.md

Context for anyone (human or agent) picking this repo up on another machine. Working codename: **Monolith** — a
placeholder; do not use the MPK or Mirage names for this engine.

## What this is

An LLM inference engine for Apple silicon (M3 / M4 / M5, macOS 26+, Metal 4). First target:
`nvidia/Qwen3.8-27B-NVFP4` (Qwen3.5-hybrid: 48 Gated-DeltaNet + 16 full-attention layers; NVFP4 MLP + `lm_head`, FP8
attention/GDN projections, BF16 MTP head; ~17.6 GB of weights read per decoded token). **v1 is judged on batch-1 decode
latency only** — prefill/TTFT, multi-request serving and energy are explicit non-goals for v1. The engine must stay
general: new models, quantization formats and ops come in through plugins, not engine edits. Stack: C++/Objective-C++
runtime, Python front-end and compiler, generated MSL kernels.

Status (2026-09-19): design, plan, survey and hardware characterization exist; **there is no engine code yet.**

## Read these, in this order

1. `docs/design/design.md` — the design. §0 is the decision table (D1–D14); §7 answers "warp specialization?" (no) and
   "static megakernel?" (static yes, one kernel no).
2. `docs/research/apple-gpu-probes.md` — what was measured on real hardware and what each number implies. §1 is a
   cross-chip table with empty M4/M5 columns; **§3 is the checklist for continuing on an M4** (hypotheses H1–H8 and the
   outcomes that would change the design).
3. `plans/implementation-plan.md` — milestones M0–M9 with exit gates and go/no-go points.
4. `docs/research/apple-inference-systems.md` — how MLX, llama.cpp and others work; what to reuse; headroom estimates.

## The design in six lines

* The whole generation loop is one GPU-resident **static program**, compiler-generated from the model graph: one Metal
  dispatch per fused op (~330 per step for Qwen3.8 = 5 all-to-all stages × 64 layers), encoded once into an indirect
  command buffer and replayed; all per-step state lives on the GPU; the host only keeps the queue fed.
* **Not** one long kernel: a dispatch is never preempted, and a dispatch boundary is as cheap a barrier as anything
  in-kernel. Keep dispatches sub-millisecond and command buffers to tens of milliseconds.
* No role-specialized SIMD-groups. Every kernel runs with the crew geometry `gpu_cores × 384 threads`
  (one threadgroup per core, 12 SIMD-groups × 32 lockstep lanes).
* Weights are re-laid-out at load time (block-lane-major packs) so each lane streams one contiguous range per block.
* The lever past the memory-bandwidth bound is MTP speculative decoding, run entirely on the GPU.
* Overlap only an ALU-bound op with a bus-bound sibling (un-barriered dispatches at full geometry); never pre-stage
  weights, never hand-partition cores.

## Working rules

* **Measure before claiming.** Evidence tags in the docs: [M] measured by us, [S] Apple spec, [R] third-party report,
  [H] hypothesis. Microsecond-level numbers move by tens of percent between runs — report ranges, draw conclusions only
  where ranges do not overlap, use paired alternating A/B runs and min-of-N.
* **Every GPU loop must be bounded.** A running dispatch cannot be preempted or cancelled; an unbounded spin freezes the
  display and can trip the watchdog. Keep any single dispatch under ~1.5 s in probes, far less in the engine.
* Correctness may depend only on documented Metal semantics (dispatch ordering, ICB barriers, the MSL memory model).
  Threadgroup→core mapping, in-flight limits and sharing behaviour are per-chip *profile values*, measured by the probes.
* Only bare-metal Macs give meaningful numbers; virtualized macOS (hosted CI runners) exposes a paravirtual GPU.
* Code adapted from other projects keeps its license header and is listed in `third_party/NOTICE` (MPK/Mirage is
  Apache-2.0; MLX, llama.cpp, tinygrad are MIT; gpt-oss Metal is Apache-2.0).
* Numerics contract: weight-only dequantization (W4A16 / W8A16), BF16 residual stream, FP32 accumulators and recurrent
  state. Reference = the HF model run on the dequantized weights. Gates: leaf ops ≤ 2 ULP, layers cos > 0.999, greedy
  tokens equal to the golden, repeated runs bit-identical.

## Probes

`./probes/run_all.sh` builds and runs the 13 probes (~4 min, Xcode Command Line Tools only; shaders compile at runtime)
and saves `probes/results/<chip>_<cores>c_macOS<ver>_<time>.txt`. `./probes/remote_run.sh user@host` does the same over
SSH. Geometry is derived from the GPU core count (`GPU_CORES=<n>` overrides). Commit every results file. Only an
M3 Pro has been measured so far; its reference run is in `probes/results/`.

## Next steps

1. On the M4: run the suite, commit the results, fill the M4 column in the hardware report §1, walk H1–H8 in §3, and
   update the design where a hypothesis fails (D4, D5, D6, D14 are the chip-sensitive decisions).
2. Plan M0 remainder: MLX / llama.cpp baselines for the target model, exact NVFP4/FP8 → BF16 dequantizer, HF goldens.
3. Plan M1 (go/no-go): real NVFP4/FP8 GEMV kernels in the crew geometry vs MLX `qmv`.
