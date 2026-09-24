#!/usr/bin/env python3
"""Generate with the step program on the GPU (plan M4, v0): a prefill program at ``T = P`` (one step) and a decode
program at ``T = 1`` replayed from one encode, sharing weights, states and StepState; tokens come back through the
ring. The prompt must fit one step (``P ≤ t_max``, 8 by default); chunked prefill comes with the dynamic-T work.

    python -m monolith.generate --model ~/models/<ckpt> --pack <pack dir> --prompt "The capital of France is" -n 48
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compiler import compile_program
from .core.profile import Profile
from .core.step_state import StepStateLayout
from .models import resolve_model
from .nn.module import Model
from .packs.packer import PackFile


@dataclass
class Generation:
    tokens: List[int]
    prefill_ms: float
    decode_ms: float               # GPU time of the decode steps
    decode_wall_ms: float
    host_busy_ms: float
    steps: int

    @property
    def ms_per_token(self) -> float:
        return self.decode_ms / max(1, self.steps)


class Session:
    """A model + pack on a device: compiles a program per static T on demand and keeps the device buffers."""

    def __init__(self, model: Model, pack_dir: str, profile: Optional[Profile] = None, *, layout: Optional[StepStateLayout] = None,
                 eos: int = -1, ring_capacity: int = 4096) -> None:
        from .bench import profile_for_device
        from .runtime import _native as nt

        self.model, self.pack = model, PackFile(pack_dir)
        self.dev = nt.Device()
        info = self.dev.info()
        self.profile = profile or profile_for_device(info.gpu_cores, info.apple_family)
        if self.profile is None:
            raise RuntimeError(f"no profile for {info.name} ({info.gpu_cores} cores, Apple{info.apple_family}); add one under profiles/")
        self.layout = layout or StepStateLayout()
        self.eos, self.ring_capacity = eos, ring_capacity
        self.engines: Dict[int, Any] = {}
        self.buffers: Optional[Dict[str, Any]] = None

    def engine(self, t: int):
        from .runtime import Engine

        if t not in self.engines:
            prog = compile_program(self.model, self.pack, self.profile, t=t, eos=self.eos, ring_capacity=self.ring_capacity, layout=self.layout)
            eng = Engine(prog, self.dev, buffers=self.buffers)
            if self.buffers is None:
                self.buffers = dict(eng.buffers)
            else:
                self.buffers.update(eng.buffers)
            self.engines[t] = eng
        return self.engines[t]

    def reset(self) -> None:
        """Zero the states, StepState and ring for a new sequence (the weights stay mapped)."""
        for eng in self.engines.values():
            for name, spec in eng.program.buffers.items():
                if spec.role in ("state", "step_state", "ring"):
                    eng.buffers[name].fill(0)

    def generate(self, prompt_ids: List[int], max_new_tokens: int, *, steps_per_cb: int = 8, in_flight: int = 3) -> Generation:
        p = len(prompt_ids)
        if not 1 <= p <= self.layout.t_max:
            raise ValueError(f"the prompt must have 1..{self.layout.t_max} tokens in v0 (chunked prefill is not implemented), got {p}")
        self.reset()
        pre = self.engine(p)
        pre.buffers[pre.program.step_state].write(self.layout.pack({"t_this_step": p, "pending_tokens": list(prompt_ids)}), 0)
        r1 = pre.run(1, steps_per_cb=1, in_flight=1)
        tokens = list(r1.tokens)
        dec_ms = dec_wall = host = 0.0
        steps = 0
        if max_new_tokens > 1 and not r1.done:
            dec = self.engine(1)
            r2 = dec.run(max_new_tokens - 1, steps_per_cb=steps_per_cb, in_flight=in_flight)
            tokens += r2.tokens
            dec_ms, dec_wall, host, steps = r2.gpu_ms, r2.wall_ms, r2.host_busy_ms, r2.steps
        return Generation(tokens, r1.gpu_ms, dec_ms, dec_wall, host, steps)

    def read(self, name: str) -> bytes:
        eng = next(iter(self.engines.values()))
        return eng.read(name)


def load_session(model_dir: str, pack_dir: str, *, max_context: int = 4096, eos: Optional[int] = None) -> Session:
    with open(Path(model_dir) / "config.json") as f:
        arch = json.load(f)["architectures"][0]
    cls = resolve_model(arch)
    if cls is None:
        raise RuntimeError(f"no model package registered for {arch!r}")
    model = cls.from_checkpoint(model_dir, max_context=max_context)
    if eos is None:
        e = getattr(model.config, "eos_token_id", None)
        eos = e[0] if isinstance(e, list) and e else (e if isinstance(e, int) else -1)
    return Session(model, pack_dir, eos=eos)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--no-eos", action="store_true", help="ignore the model's EOS (fixed-length generation)")
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = tok.encode(a.prompt, add_special_tokens=False).ids
    t0 = time.time()
    sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1 if a.no_eos else None)
    gen = sess.generate(ids, a.max_new_tokens)
    wall = time.time() - t0
    print(tok.decode(gen.tokens))
    print(f"\n# {len(gen.tokens)} tokens; prefill {gen.prefill_ms:.1f} ms (T={len(ids)}); decode {gen.ms_per_token:.2f} ms/token GPU "
          f"({1000 / gen.ms_per_token:.1f} tok/s), wall {gen.decode_wall_ms / max(1, gen.steps):.2f} ms/token, host busy "
          f"{100 * gen.host_busy_ms / max(gen.decode_wall_ms, 1e-9):.1f} %; total wall {wall:.1f} s incl. compile", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
