"""Instantiate a :class:`Program` on the device and replay it: the Python face of the host pump (design §5.4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from . import _native as nt
from .program import Program


@dataclass
class StepReport:
    steps: int
    command_buffers: int
    gpu_ms: float
    wall_ms: float
    host_busy_ms: float
    done: bool
    tokens: List[int]

    @property
    def host_fraction(self) -> float:
        """CPU time of the pump over wall time — the '< 5 % of a core' metric."""
        return self.host_busy_ms / self.wall_ms if self.wall_ms else 0.0


class Engine:
    """``buffers`` lets several programs share device buffers by name (a prefill program at T = P and a decode
    program at T = 1 over the same weights, states and StepState)."""

    def __init__(self, program: Program, device: Optional[nt.Device] = None, buffers: Optional[Dict[str, nt.Buffer]] = None) -> None:
        self.program = program
        self.dev = device or nt.Device()
        self.buffers: Dict[str, nt.Buffer] = {}
        for name, spec in program.buffers.items():
            if buffers is not None and name in buffers and buffers[name].nbytes >= spec.nbytes:
                self.buffers[name] = buffers[name]           # shared (weights, states, StepState, ring, arena values)
                continue
            if buffers is not None and name in buffers and spec.role in ("state", "step_state", "ring", "weights"):
                raise ValueError(f"shared buffer {name}: {buffers[name].nbytes} bytes, program needs {spec.nbytes}")
            if spec.file is not None:
                self.buffers[name] = nt.Buffer.from_file(self.dev, spec.file, spec.file_offset, spec.nbytes)
                continue
            if spec.init is not None:
                if len(spec.init) > spec.nbytes:
                    raise ValueError(f"buffer {name}: init larger than nbytes")
                buf = nt.Buffer(self.dev, spec.nbytes)
                buf.fill(0)
                buf.write(spec.init, 0)
            else:
                buf = nt.Buffer(self.dev, spec.nbytes)
                buf.fill(0)
            self.buffers[name] = buf
        self.pipelines: Dict[str, nt.Pipeline] = {}
        for key, k in program.kernels.items():
            lib = nt.Library(self.dev, k.source, k.macros)
            self.pipelines[key] = nt.Pipeline(lib, k.function, True)
        self.ops = []
        for o in program.ops:
            d = nt.Dispatch().pipeline(self.pipelines[o.kernel]).grid(*o.grid).threadgroup(*o.threadgroup).barrier(o.barrier_after)
            for index, bname, off in o.bindings:
                d.buffer(index, self.buffers[bname], off)
            for index, length in o.threadgroup_memory:
                d.threadgroup_memory(index, length)
            self.ops.append(d)
        self.icb = nt.Icb(self.dev, self.ops)
        lay = program.layout
        ring = self.buffers[program.ring]
        if ring.nbytes < program.ring_capacity * 8:
            raise ValueError("the token ring needs 8 bytes per slot: (sequence << 32) | token")
        self.runner = nt.Runner(self.dev, self.icb, self.ops, list(self.buffers.values()), self.buffers[program.step_state],
                                lay.offset("done"), lay.offset("ring_head"), lay.offset("ring_tail"), ring, program.ring_capacity)

    def run(self, max_steps: int, *, steps_per_cb: int = 8, in_flight: int = 3, reencode: bool = False) -> StepReport:
        st = self.runner.run(max_steps, steps_per_cb, in_flight, reencode)
        if st.error:
            raise RuntimeError(st.error)
        return StepReport(st.steps_submitted, st.command_buffers, st.gpu_ms, st.wall_ms, st.host_busy_ms, st.done, self.runner.drain())

    def profile(self, steps: int = 3) -> List[List[Tuple[float, float]]]:
        """Per-dispatch GPU (start, end) ms for ``steps`` re-encoded steps (one encoder per op with timestamp counter
        samples, so the numbers carry encoder-boundary gaps the ICB replay does not have — use them for the shares
        and the per-op durations, not for the step total)."""
        q = nt.Queue(self.dev)
        return [q.profile(self.ops) for _ in range(steps)]

    def state(self) -> Dict[str, object]:
        buf = self.buffers[self.program.step_state]
        return self.program.layout.unpack(buf.read(0, self.program.layout.size))

    def read(self, name: str, nbytes: Optional[int] = None) -> bytes:
        b = self.buffers[name]
        return b.read(0, nbytes if nbytes is not None else b.nbytes)
