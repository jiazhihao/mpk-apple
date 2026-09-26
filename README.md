# Monolith *(working codename)*

A megakernel-style LLM inference engine for Apple silicon (M3 / M4 / M5, macOS 26+). First target:
[`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), batch-1 decode latency, with
[DSpark](docs/research/dspark.md) speculative decoding; the engine itself is model-agnostic by construction. This is a
standalone repository: code from MPK and other projects is copied in with its license headers, never depended on.

The idea, carried over from MPK: compile the *whole generation loop* — every layer, sampling, speculative
accept/rollback, stop detection — into one GPU-resident static program so that no CPU work and no CPU↔GPU
synchronization sits on the critical path. The mechanism is re-derived for Apple GPUs from measurements: a
pre-encoded, self-advancing chain of bounded whole-GPU dispatches over a homogeneous crew of SIMD-groups, streaming
weights from a block-lane-major pack — not one never-returning kernel with specialized roles.

| Document | What it is |
|---|---|
| [`docs/design/design.md`](docs/design/design.md) | The design: MPK Runtime V2 re-derived for Apple GPUs; answers on warp specialization and the static-megakernel approach |
| [`plans/implementation-plan.md`](plans/implementation-plan.md) | Milestones M0–M9 with exit gates and go/no-go points, repo layout, tests, reuse map, risks |
| [`docs/research/apple-gpu-probes.md`](docs/research/apple-gpu-probes.md) | Measured Apple-GPU execution model (M3 Pro, M5 Pro): core mapping, in-kernel sync, preemption and sharing, bandwidth vs access pattern and lane order, in-kernel barriers vs dispatch boundaries, real FP8/NVFP4 decode kernels, the M5 `matmul2d` path |
| [`docs/research/apple-inference-systems.md`](docs/research/apple-inference-systems.md) | How MLX, llama.cpp and others run LLMs on Apple silicon; what we reuse |
| [`docs/research/dspark.md`](docs/research/dspark.md) | DSpark speculative decoding: the method, the public drafters for our targets, what a round costs on our hardware |
| [`docs/porting.md`](docs/porting.md) | The porting guide: adding a model, a format, an op, a drafter or a chip — the contracts, the registries, the CI checks, the golden workflow, with the time each port took ([`porting-log.md`](docs/research/porting-log.md)) |
| [`probes/`](probes) | The 16 probe programs. `./probes/run_all.sh` runs them on this machine (Command Line Tools only, ~5 min) and saves `probes/results/<chip>….txt`; `./probes/remote_run.sh user@host` does the same on another bare-metal Mac. Measured: M3 Pro (2026-09-19), M5 Pro (2026-09-22) |
| [`profiles/`](profiles) | Per-chip profiles: the M5 Pro's written by `tools/profile_writer.py` from the kernel harnesses, the M3 Pro's derived by hand from the probe results |

Picking this up on another machine? Start with [`CLAUDE.md`](CLAUDE.md); §4 of the hardware report is the checklist for a chip not yet measured.

Status: design and plan drafted 2026-09-19, revised 2026-09-22 (M5 Pro measurements) and 2026-09-23 (build phase,
DSpark). The hardware-characterization half of M0 is done for the M3 Pro (13 probes) and the M5 Pro (16 probes,
including the first real FP8/NVFP4 decode kernels and an M5 `matmul2d` path). Engine code starts with the PRs listed
in the plan's §6. The M3 Pro and M4 tasks were dropped from the roadmap on 2026-09-25: the M5 Pro on hand (24 GB) is
the only machine, and the 27B target's baselines and goldens wait for a machine that hosts it.
