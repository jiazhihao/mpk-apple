"""Assembling MSL sources for the runtime compiler: kernel templates under ``kernels/`` plus the format plugins'
decode snippets and the macros that specialize them (design §5.7: block bodies + generated wrappers)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

from .formats import FORMATS
from .formats.blm import PackInfo

KERNELS_DIR = Path(__file__).resolve().parents[1] / "kernels"
PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"


def template(name: str) -> str:
    return (KERNELS_DIR / name).read_text()


def gemv_source(fmt: str) -> str:
    """The gemv_T kernel for storage format ``fmt`` (macros still to be supplied at compile time)."""
    return PRELUDE + FORMATS.get(fmt).msl_decode + "\n" + template("gemv_T.metal")


def gemv_macros(info: PackInfo, *, t: int, rg: int | None = None, out_bf16: bool = False) -> Dict[str, str]:
    """The compile-time specialization of gemv_T for one slab geometry and token count."""
    f = FORMATS.get(info.format)
    if info.k % 32:
        raise ValueError("gemv_T: K must be a multiple of 32")
    payload_words = info.payload_bytes // 16
    if info.payload_bytes % 16 or (info.k // 32) % f.weights_per_word:
        raise ValueError(f"gemv_T: a lane's stripe must be whole words for {info.format} (K={info.k})")
    scale_words = -(-info.scale_bytes // 16) if info.scale_bytes else 0
    if rg is None:
        rg = 2          # measured on the M5 Pro: RG = 2 beats 4 for FP8 (254 vs 223 GB/s) and ties for NVFP4; 1 is worse
    rg = min(rg, info.rows)
    if info.rows % rg:
        raise ValueError(f"gemv_T: RG={rg} must divide R={info.rows}")
    preconvert = t * f.weights_per_word <= 64          # T*WPW floats of registers; beyond that convert per row
    return {"K": str(info.k), "R": str(info.rows), "T": str(t), "RG": str(rg),
            "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1",
            "UNIT_WORDS": str(info.unit_bytes // 16), "PAYLOAD_WORDS": str(payload_words),
            "SCALE_WORDS": str(scale_words), "OUT_BF16": "1" if out_bf16 else "0",
            "X_PRECONVERT": "1" if preconvert else "0"}


def macro_key(macros: Mapping[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(macros.items()))
