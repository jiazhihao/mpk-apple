"""Chip profiles: the measured, per-chip values the compiler and the autotuner consume (design §5.7).

A profile is a JSON file under ``profiles/`` (hand-derived from the probe results today; written by the autotuner
later). The free-form measurement blocks are kept as ``raw``; the ``engine`` block is the normalized part this class
exposes: GPU family (the kernel-binding key), lane order of the weight pack, threadgroups per core, the encode-order
rule for sibling overlap, the command-buffer length, and the ``cost(T)`` tables the verify-length rule optimizes
against (design §5.8).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


@dataclass
class Profile:
    name: str
    chip: str
    family: str                       # "Apple9", "Apple10", …
    gpu_cores: int
    nominal_gbps: float
    lane_order: str                   # "contiguous" | "interleaved16"
    threadgroups_per_core: int = 1
    sibling_order: str = "either"     # "alu_first" | "bus_first" | "either"
    max_cb_ms: float = 16.0
    attention: str = "v1"             # the attention kernel: "v1" (block = kv head × chunk × row group) or "v2" (§5.6 v2, #34)
    accelerator: str = "off"          # "on": T > 1 GEMVs run on the tensor-ops tile (gemm_tile, #50/#51) above accelerator_min_t
    accelerator_min_t: Dict[str, int] = field(default_factory=dict)     # cost_T format key -> the smallest T the tile covers (default 2)
    cost_t: Dict[str, Dict[int, float]] = field(default_factory=dict)   # format -> {T: cost relative to T = 1}
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- kernel-binding key -------------------------------------------------------------------------------------
    @property
    def key(self) -> str:
        """Profile key used by op kernel bindings: the GPU family, lower-case (``apple10``)."""
        return self.family.lower()

    @property
    def crew_threads(self) -> int:
        return 384

    # ---- cost tables --------------------------------------------------------------------------------------------
    def cost(self, fmt: str, t: int) -> float:
        """Cost of a ``t``-token pass over weights of format ``fmt``, in units of a T = 1 pass.

        Exact at measured points, linear between them; raises ``KeyError`` for an unmeasured format and
        ``ValueError`` outside the measured range (the rule must not extrapolate).
        """
        table = self.cost_t.get(fmt)
        if not table:
            raise KeyError(f"profile {self.name}: no cost table for format {fmt!r}")
        if t in table:
            return table[t]
        ts = sorted(table)
        if t < ts[0] or t > ts[-1]:
            raise ValueError(f"profile {self.name}: T={t} outside the measured range {ts[0]}..{ts[-1]} for {fmt!r}")
        lo = max(x for x in ts if x < t)
        hi = min(x for x in ts if x > t)
        w = (t - lo) / (hi - lo)
        return table[lo] * (1 - w) + table[hi] * w

    # ---- construction -------------------------------------------------------------------------------------------
    @classmethod
    def from_dict(cls, name: str, d: Mapping[str, Any]) -> "Profile":
        eng = d.get("engine")
        if not isinstance(eng, Mapping):
            raise ValueError(f"profile {name}: missing the 'engine' block")
        for k in ("family", "lane_order"):
            if k not in eng:
                raise ValueError(f"profile {name}: engine.{k} is required")
        if eng["lane_order"] not in ("contiguous", "interleaved16"):
            raise ValueError(f"profile {name}: engine.lane_order must be 'contiguous' or 'interleaved16'")
        if eng.get("attention", "v1") not in ("v1", "v2"):
            raise ValueError(f"profile {name}: engine.attention must be 'v1' or 'v2'")
        if eng.get("accelerator", "off") not in ("on", "off"):
            raise ValueError(f"profile {name}: engine.accelerator must be 'on' or 'off'")
        min_t = {str(f): int(t) for f, t in (eng.get("accelerator_min_t") or {}).items()}
        if any(t < 1 for t in min_t.values()):
            raise ValueError(f"profile {name}: engine.accelerator_min_t entries must be >= 1")
        cost_t = {f: {int(t): float(c) for t, c in tbl.items()} for f, tbl in (eng.get("cost_T") or {}).items()}
        for f, tbl in cost_t.items():
            if tbl.get(1, 1.0) != 1.0:
                raise ValueError(f"profile {name}: cost_T[{f!r}][1] must be 1.0 (costs are relative to T = 1)")
        return cls(
            name=name,
            chip=str(d.get("chip", name)),
            family=str(eng["family"]),
            gpu_cores=int(d["gpu_cores"]),
            nominal_gbps=float(d["nominal_gbps"]),
            lane_order=str(eng["lane_order"]),
            threadgroups_per_core=int(eng.get("threadgroups_per_core", 1)),
            sibling_order=str(eng.get("sibling_order", "either")),
            max_cb_ms=float(eng.get("max_cb_ms", 16.0)),
            attention=str(eng.get("attention", "v1")),
            accelerator=str(eng.get("accelerator", "off")),
            accelerator_min_t=min_t,
            cost_t=cost_t,
            raw=dict(d),
        )


def profiles_dir() -> Path:
    """``profiles/`` at the repository root (this file lives in ``<root>/monolith/core``)."""
    return Path(__file__).resolve().parents[2] / "profiles"


def load_profile(path: str | Path) -> Profile:
    p = Path(path)
    with open(p) as f:
        return Profile.from_dict(p.stem, json.load(f))


def load_profiles(directory: Optional[str | Path] = None) -> Dict[str, Profile]:
    d = Path(directory) if directory else profiles_dir()
    return {p.stem: load_profile(p) for p in sorted(d.glob("*.json"))}
