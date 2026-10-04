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

COST_FORMAT = {"fp8_e4m3": "fp8"}   # pack format -> the profile's cost_T key (the others use their own name)


@dataclass
class Profile:
    name: str
    chip: str
    family: str                       # "Apple9", "Apple10", …
    gpu_cores: int
    nominal_gbps: float
    lane_order: str                   # "contiguous" | "interleaved16"
    scale_placement: str = "inline"   # "inline" | "block": where a pack keeps its block scales (blm.py, #101)
    threadgroups_per_core: int = 1
    sibling_order: str = "either"     # "alu_first" | "bus_first" | "either"
    max_cb_ms: float = 16.0
    attention: str = "v1"             # v1 / v2 / v3 / mma; auto chooses M5 matrix tiles at T >= 4, otherwise v3
    attention_rows: int = 4           # v1's query rows per pass over a chunk (RBMAX): more rows stream the chunk fewer times, at register cost
    attention_v2_threadgroups: int = 2  # v2's threadgroups per core (its blocks are threadgroups: two per core hide the latency of one)
    accelerator: str = "off"          # "on": T > 1 GEMVs run on the tensor-ops tile (gemm_tile, #50/#51) above accelerator_min_t
    accelerator_min_t: Dict[str, int] = field(default_factory=dict)     # cost_T format key -> the smallest T the tile covers (default 2)
    cost_t: Dict[str, Dict[int, float]] = field(default_factory=dict)   # format -> {T: cost relative to T = 1}
    gdn_mixer_fusion: Dict[str, Any] = field(default_factory=dict)      # measured fixed-eight-row shape and worker configuration
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
        if eng.get("scale_placement", "inline") not in ("inline", "block"):
            raise ValueError(f"profile {name}: engine.scale_placement must be 'inline' or 'block'")
        if eng.get("attention", "v1") not in ("v1", "v2", "v3", "mma", "auto"):
            raise ValueError(f"profile {name}: engine.attention must be 'v1', 'v2', 'v3', 'mma' or 'auto'")
        if int(eng.get("attention_rows", 4)) not in (1, 2, 4, 8, 16):
            raise ValueError(f"profile {name}: engine.attention_rows must be 1, 2, 4, 8 or 16")
        if int(eng.get("attention_v2_threadgroups", 2)) not in (1, 2, 3, 4):
            raise ValueError(f"profile {name}: engine.attention_v2_threadgroups must be 1..4")
        if eng.get("accelerator", "off") not in ("on", "off"):
            raise ValueError(f"profile {name}: engine.accelerator must be 'on' or 'off'")
        min_t = {str(f): int(t) for f, t in (eng.get("accelerator_min_t") or {}).items()}
        if any(t < 1 for t in min_t.values()):
            raise ValueError(f"profile {name}: engine.accelerator_min_t entries must be >= 1")
        cost_t = {f: {int(t): float(c) for t, c in tbl.items()} for f, tbl in (eng.get("cost_T") or {}).items()}
        fusion = dict(eng.get("gdn_mixer_fusion") or {})
        if fusion:
            required = {"shape", "workers", "sgs", "tn", "split", "compact", "gdn_sl", "barrier", "task_barrier", "scalar_sgs"}
            packed = {"fp8_layout", "q_outer", "fp8_decode", "fp8_tile_block", "restrict_weights",
                      "gemm_overrides", "perm_sgs", "direct_norm", "dual_permute"}
            if (set(fusion) not in (required, required | packed) or not isinstance(fusion.get("shape"), (list, tuple))
                    or len(fusion["shape"]) != 6):
                raise ValueError(f"profile {name}: gdn_mixer_fusion needs a six-dimension shape and the measured geometry")
            is_packed = "fp8_layout" in fusion
            geometry = (8, 32, 8, 8) if is_packed else (16, 16, 4, 4)
            if (any(type(v) is not int or v < 1 for v in fusion["shape"])
                    or type(fusion["workers"]) is not int or not 1 <= fusion["workers"] <= min(256, 4 * int(d["gpu_cores"]))
                    or tuple(fusion[k] for k in ("sgs", "tn", "gdn_sl", "scalar_sgs")) != geometry
                    or fusion["barrier"] != "simd"
                    or fusion["split"] is not True or fusion["compact"] is not True or fusion["task_barrier"] is not False):
                raise ValueError(f"profile {name}: unsupported gdn_mixer_fusion geometry")
            if is_packed:
                # Only the independently validated operand/input-transform recipe
                # is a profile option. Experimental search knobs stay in the bench.
                if (fusion["fp8_layout"] != "tile" or type(fusion["q_outer"]) is not int or fusion["q_outer"] != 0
                        or fusion["fp8_decode"] != "subtract" or fusion["fp8_tile_block"] != 8
                        or fusion["restrict_weights"] is not False or fusion["perm_sgs"] != 64
                        or fusion["direct_norm"] is not True or fusion["dual_permute"] is not True):
                    raise ValueError(f"profile {name}: unsupported gdn_mixer_fusion packing")
                overrides = fusion["gemm_overrides"]
                if not isinstance(overrides, Mapping) or set(overrides) != {"0", "1", "2", "3"}:
                    raise ValueError(f"profile {name}: gdn_mixer_fusion needs four projection geometries")
                for stage, (tn, split) in enumerate(((32, 2), (16, 8), (32, 8), (32, 4))):
                    config = overrides[str(stage)]
                    if (not isinstance(config, Mapping) or set(config) != {"tn", "ksplit", "groups", "ragged_teams"}
                            or config["tn"] != tn or config["ksplit"] != split or config["ragged_teams"] is not True
                            or type(config["groups"]) is not int or not 1 <= config["groups"] <= 6 * int(d["gpu_cores"])):
                        raise ValueError(f"profile {name}: unsupported gdn_mixer_fusion projection {stage}")
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
            scale_placement=str(eng.get("scale_placement", "inline")),
            threadgroups_per_core=int(eng.get("threadgroups_per_core", 1)),
            sibling_order=str(eng.get("sibling_order", "either")),
            max_cb_ms=float(eng.get("max_cb_ms", 16.0)),
            attention=str(eng.get("attention", "v1")),
            attention_rows=int(eng.get("attention_rows", 4)),
            attention_v2_threadgroups=int(eng.get("attention_v2_threadgroups", 2)),
            accelerator=str(eng.get("accelerator", "off")),
            accelerator_min_t=min_t,
            cost_t=cost_t,
            gdn_mixer_fusion=fusion,
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
