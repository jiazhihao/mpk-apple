#!/usr/bin/env python3
"""Generate with the step program on the GPU (plan M4): the prompt is fed in chunks of ``t_max`` tokens through a
dynamic-T prefill program (T read from StepState per step, the last chunk shorter), then a decode program at
``T = 1`` is replayed from one encode; all programs share weights, states, StepState and the ring, and tokens come
back through the ring. With a drafter (design §5.8) the dynamic-T program carries the whole speculative round —
verify pass, accept scan, state commit, draft pass, verify-length select — and is replayed for decode as well; the
host only drains tokens.

    python -m monolith.generate --model ~/models/<ckpt> --pack <pack dir> --prompt "The capital of France is" -n 48
    python -m monolith.generate --model … --pack … --drafter ~/models/<drafter> --drafter-pack <dir> [--verify cost|threshold]
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
from .nn.sampler import GreedySampler, StochasticSampler
from .packs.packer import PackFile


@dataclass
class Generation:
    tokens: List[int]
    prefill_ms: float
    decode_ms: float               # GPU time of the decode steps
    decode_wall_ms: float
    host_busy_ms: float
    steps: int
    decode_tokens: int = 0         # tokens the decode steps produced (= steps without a drafter)
    accepted: Optional[List[int]] = None     # per decode step: drafts accepted (speculative sessions)
    committed: Optional[List[int]] = None    # per decode step: tokens committed (accepted + the bonus)

    @property
    def ms_per_token(self) -> float:
        """GPU time per decoded token; a speculative session counts every token its steps committed (the pump may
        run a few steps past the requested count), the plain one its steps."""
        n = sum(self.committed) if self.committed else self.decode_tokens
        return self.decode_ms / max(1, n)

    @property
    def tokens_per_step(self) -> float:
        return sum(self.committed) / len(self.committed) if self.committed else 1.0

    @property
    def mean_accepted(self) -> float:
        return sum(self.accepted) / len(self.accepted) if self.accepted else 0.0


class Session:
    """A model + pack on a device: compiles a program per static T on demand and keeps the device buffers."""

    def __init__(self, model: Model, pack_dir: str, profile: Optional[Profile] = None, *, layout: Optional[StepStateLayout] = None,
                 eos: int = -1, ring_capacity: int = 4096, temperature: float = 0.0, top_k: int = 0, top_p: float = 0.0,
                 min_p: float = 0.0, seed: int = 0, autotune: bool = True, drafter: Any = None, drafter_pack: Optional[str] = None,
                 verify: str = "cost", verify_threshold: Optional[float] = None) -> None:
        """``drafter`` (a ``Drafter`` built with the model's head) and its pack turn the session speculative: one
        dynamic-T program holds the round; ``verify`` / ``verify_threshold`` as in ``compile_program``."""
        from .bench import profile_for_device
        from .runtime import _native as nt

        self.model, self.pack = model, PackFile(pack_dir)
        self.dev = nt.Device()
        info = self.dev.info()
        self.profile = profile or profile_for_device(info.gpu_cores, info.apple_family)
        if self.profile is None:
            raise RuntimeError(f"no profile for {info.name} ({info.gpu_cores} cores, Apple{info.apple_family}); add one under profiles/")
        self.drafter, self.drafter_pack = drafter, (PackFile(drafter_pack) if drafter is not None else None)
        if drafter is not None and drafter_pack is None:
            raise ValueError("Session: a drafter needs its pack (drafter_pack)")
        if temperature > 0 and drafter is not None:
            raise NotImplementedError("Session: sampling with a drafter (rejection sampling, #39) is not implemented; use greedy")
        if layout is None and drafter is not None:
            layout = StepStateLayout(t_max=max(8, drafter.gamma + 1), gamma_max=max(7, drafter.gamma))
        self.layout = layout or StepStateLayout()
        self.verify, self.verify_threshold = verify, verify_threshold
        self.eos, self.ring_capacity = eos, ring_capacity
        self.seed = seed
        # temperature 0 = greedy (the argmax path); otherwise the Gumbel-max sampler with the thresholds
        model.sampler = GreedySampler(prefix="sampler.") if temperature <= 0 else StochasticSampler(temperature, top_k, top_p, min_p, seed, prefix="sampler.")
        self.engines: Dict[int, Any] = {}
        self.buffers: Optional[Dict[str, Any]] = None
        self.tuner = None
        if autotune:
            from .compiler.autotune import Autotuner

            chip = info.name.replace(" ", "-").lower()
            self.tuner = Autotuner(self.dev, info.gpu_cores, str(Path(pack_dir) / f"autotune.{chip}.json"))

    def engine(self, t: int):
        """The engine for a static ``T = t``; ``t = 0`` is the dynamic-T program (prefill, and with a drafter the
        whole speculative round, decode included)."""
        from .runtime import Engine

        if self.drafter is not None and t != 0:
            raise ValueError("Session: a speculative session runs everything in the dynamic-T program (engine(0))")
        if t not in self.engines:
            prog = compile_program(self.model, self.pack, self.profile, t=None if t == 0 else t, dynamic_t=(t == 0), eos=self.eos,
                                   ring_capacity=self.ring_capacity, layout=self.layout, tuner=self.tuner, drafter=self.drafter,
                                   drafter_pack=self.drafter_pack, verify=self.verify, verify_threshold=self.verify_threshold)
            if self.tuner is not None:
                self.tuner.save(self.dev.info().name)
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
        if p < 1:
            raise ValueError("the prompt must have at least one token")
        self.reset()
        t_max = self.layout.t_max
        chunks = [list(prompt_ids[i: i + t_max]) for i in range(0, p, t_max)]
        pre = self.engine(0)
        st = pre.buffers[pre.program.step_state]
        prefill_ms = 0.0
        tokens: List[int] = []
        for k, chunk in enumerate(chunks):
            # the host writes each chunk's tokens and length; the advance emits only after the last chunk
            state = self.layout.unpack(st.read(0, self.layout.size))
            state.update(t_this_step=len(chunk), pending_tokens=chunk, prefill_left=len(chunks) - 1 - k,
                         rng_lo=self.seed & 0xFFFFFFFF, rng_hi=(self.seed >> 32) & 0xFFFFFFFF)
            st.write(self.layout.pack(state), 0)
            r1 = pre.run(1, steps_per_cb=1, in_flight=1)
            prefill_ms += r1.gpu_ms
            tokens += r1.tokens
        dec_ms = dec_wall = host = 0.0
        steps = 0
        n_pre = len(tokens)
        if max_new_tokens > 1 and not r1.done:
            if self.drafter is None:
                dec = self.engine(1)
                r2 = dec.run(max_new_tokens - 1, steps_per_cb=steps_per_cb, in_flight=in_flight)
                tokens += r2.tokens
                dec_ms, dec_wall, host, steps = r2.gpu_ms, r2.wall_ms, r2.host_busy_ms, r2.steps
            else:
                done = False                               # every step commits ≥ 1 token: the remaining count bounds the steps
                while len(tokens) < max_new_tokens and not done:
                    need = max_new_tokens - len(tokens)
                    r2 = pre.run(need, steps_per_cb=steps_per_cb, in_flight=in_flight, max_tokens=need)
                    tokens += r2.tokens
                    dec_ms += r2.gpu_ms; dec_wall += r2.wall_ms; host += r2.host_busy_ms; steps += r2.steps
                    done = r2.done or r2.steps == 0
        gen = Generation(tokens[:max_new_tokens], prefill_ms, dec_ms, dec_wall, host, steps, decode_tokens=min(len(tokens), max_new_tokens) - n_pre)
        if self.drafter is not None:
            gen.accepted, gen.committed = self._accept_stats(pre, len(chunks))
        return gen

    def _accept_stats(self, eng, n_prefill_steps: int):
        """Per decode step (accepted drafts, committed tokens) from the program's accept log."""
        import numpy as np

        from .compiler.emit import ACCEPT_LOG

        n_steps = int(eng.state()["step"])
        log = np.frombuffer(eng.read(ACCEPT_LOG), dtype=np.uint32)[:n_steps]
        dec = [int(v) for v in log[n_prefill_steps:] if (v & 0xFFFF) != 0xFFFF]
        return [v & 0xFFFF for v in dec], [v >> 16 for v in dec]

    def read(self, name: str) -> bytes:
        eng = next(iter(self.engines.values()))
        return eng.read(name)


def load_session(model_dir: str, pack_dir: str, *, max_context: int = 4096, eos: Optional[int] = None, drafter_dir: Optional[str] = None,
                 drafter_pack: Optional[str] = None, drafter_kind: str = "dspark", **options: Any) -> Session:
    """The session for a checkpoint directory (+ optionally a drafter's: its kind names the ``Drafter`` plugin)."""
    with open(Path(model_dir) / "config.json") as f:
        arch = json.load(f)["architectures"][0]
    cls = resolve_model(arch)
    if cls is None:
        raise RuntimeError(f"no model package registered for {arch!r}")
    model = cls.from_checkpoint(model_dir, max_context=max_context)
    if eos is None:
        e = getattr(model.config, "eos_token_id", None)
        eos = e[0] if isinstance(e, list) and e else (e if isinstance(e, int) else -1)
    drafter = None
    if drafter_dir is not None:
        from .spec import DRAFTERS

        drafter = DRAFTERS.get(drafter_kind).from_checkpoint(drafter_dir, target_lm_head=model.lm_head, max_context=max_context)
    return Session(model, pack_dir, eos=eos, drafter=drafter, drafter_pack=drafter_pack, **options)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("-n", "--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--no-eos", action="store_true", help="ignore the model's EOS (fixed-length generation)")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = greedy; otherwise Gumbel-max sampling on the GPU")
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-autotune", action="store_true", help="compile with the default kernel geometry (no per-op tuning cache)")
    ap.add_argument("--drafter", default=None, help="a drafter checkpoint directory: speculative decoding (design §5.8)")
    ap.add_argument("--drafter-pack", default=None, help="the drafter's pack (tools/pack_weights.py --drafter-kind …)")
    ap.add_argument("--drafter-kind", default="dspark", help="the Drafter plugin the drafter checkpoint belongs to")
    ap.add_argument("--verify", default="cost", choices=["cost", "threshold"], help="the verify-length rule (cost needs the chip's cost table)")
    ap.add_argument("--verify-threshold", type=float, default=None, help="the confident-prefix threshold (<= 0: verify the whole block)")
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json"))
    ids = tok.encode(a.prompt, add_special_tokens=False).ids
    t0 = time.time()
    sess = load_session(a.model, a.pack, max_context=a.max_context, eos=-1 if a.no_eos else None,
                        temperature=a.temperature, top_k=a.top_k, top_p=a.top_p, min_p=a.min_p, seed=a.seed, autotune=not a.no_autotune,
                        drafter_dir=a.drafter, drafter_pack=a.drafter_pack, drafter_kind=a.drafter_kind, verify=a.verify,
                        verify_threshold=a.verify_threshold)
    gen = sess.generate(ids, a.max_new_tokens)
    wall = time.time() - t0
    print(tok.decode(gen.tokens))
    print(f"\n# {len(gen.tokens)} tokens; prefill {gen.prefill_ms:.1f} ms ({len(ids)} prompt tokens); decode {gen.ms_per_token:.2f} ms/token GPU "
          f"({1000 / gen.ms_per_token:.1f} tok/s), wall {gen.decode_wall_ms / max(1, gen.decode_tokens):.2f} ms/token, host busy "
          f"{100 * gen.host_busy_ms / max(gen.decode_wall_ms, 1e-9):.1f} %; total wall {wall:.1f} s incl. compile", file=sys.stderr)
    if gen.accepted is not None:
        hist = {}
        for c in gen.accepted:
            hist[c] = hist.get(c, 0) + 1
        print(f"# speculative: {gen.steps} decode steps, {gen.decode_tokens / max(1, gen.steps):.2f} tokens/step, mean accepted "
              f"{gen.mean_accepted:.2f}, accepted histogram {dict(sorted(hist.items()))}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
