# Monolith *(working codename)*

A megakernel-style LLM inference engine for Apple silicon (M3 / M4 / M5, macOS 26+). First target:
[`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), batch-1 decode latency; the engine
itself is model-agnostic.

The idea, carried over from MPK: compile the *whole generation loop* — every layer, sampling, speculative
accept/rollback, stop detection — into one GPU-resident static program so that no CPU work and no CPU↔GPU
synchronization sits on the critical path. The mechanism is re-derived for Apple GPUs from measurements: a
pre-encoded, self-advancing chain of bounded whole-GPU dispatches over a homogeneous crew of SIMD-groups, streaming
weights from a block-lane-major pack — not one never-returning kernel with specialized roles.

| Document | What it is |
|---|---|
| [`docs/design/design.md`](docs/design/design.md) | The design: MPK Runtime V2 re-derived for Apple GPUs; answers on warp specialization and the static-megakernel approach |
| [`plans/implementation-plan.md`](plans/implementation-plan.md) | Milestones M0–M9 with exit gates and go/no-go points, repo layout, tests, reuse map, risks |
| [`docs/research/apple-gpu-probes.md`](docs/research/apple-gpu-probes.md) | Measured Apple-GPU execution model (M3 Pro): core mapping, in-kernel sync, no preemption within a dispatch, sharing at dispatch granularity, bandwidth vs access pattern, in-kernel barriers vs dispatch boundaries |
| [`docs/research/apple-inference-systems.md`](docs/research/apple-inference-systems.md) | How MLX, llama.cpp and others run LLMs on Apple silicon; what we reuse |
| [`probes/`](probes) | The 13 probe programs. `./probes/run_all.sh` runs them on this machine (Command Line Tools only, ~4 min) and saves `probes/results/<chip>….txt`; `./probes/remote_run.sh user@host` does the same on another bare-metal Mac. Only the M3 Pro has been measured so far |

Picking this up on another machine? Start with [`CLAUDE.md`](CLAUDE.md); the M4 checklist is §3 of the hardware report.

Status: design and plan drafted 2026-09-19. The hardware-characterization half of M0 is done for the M3 Pro
(13 probes); M4/M5 measurements, baselines, goldens and engine code are not started.
