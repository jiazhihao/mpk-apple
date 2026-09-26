"""Assembling MSL sources for the runtime compiler: kernel templates under ``kernels/`` plus the format plugins'
decode snippets and the macros that specialize them (design §5.7: block bodies + generated wrappers)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Mapping, Optional

from .formats import FORMATS
from .formats.blm import PackInfo

KERNELS_DIR = Path(__file__).resolve().parents[1] / "kernels"
PRELUDE = "#include <metal_stdlib>\nusing namespace metal;\n"


def template(name: str) -> str:
    return (KERNELS_DIR / name).read_text()


def gemv_source(fmt: str) -> str:
    """The gemv_T kernel for storage format ``fmt`` (macros still to be supplied at compile time)."""
    return PRELUDE + FORMATS.get(fmt).msl_decode + "\n" + template("gemv_T.metal")


EPILOGUES = {None: "0", "residual": "1", "silu_mul": "2"}


def gemv_macros(info: PackInfo, *, t: int, rg: int | None = None, out_bf16: bool = False, norm: bool = False,
                epilogue: Optional[str] = None, stat_out: bool = False) -> Dict[str, str]:
    """The compile-time specialization of gemv_T for one slab geometry, token count and set of fusions
    (``norm``: RMSNorm scaling on the input; ``epilogue``: ``residual`` | ``silu_mul``; ``stat_out``: per-block
    partial sums of squares of the outputs for the next norm)."""
    f = FORMATS.get(info.format)
    if info.k % 32:
        raise ValueError("gemv_T: K must be a multiple of 32")
    payload_words = info.payload_bytes // 16
    if info.payload_bytes % 16 or (info.k // 32) % f.weights_per_word:
        raise ValueError(f"gemv_T: a lane's stripe must be whole words for {info.format} (K={info.k})")
    scale_words = -(-info.scale_bytes // 16) if info.scale_bytes else 0
    if rg is None:
        # measured on the M5 Pro (gemv-kernel-study.md §3b, §3d): at T = 1, RG = 2 beats 4/8 for FP8 (-8 % at 8) and
        # ties for NVFP4; at T >= 2, RG = 8 wins by 13 % (NVFP4) to 20-34 % (FP8). With the input norm fused the
        # activation chunk is re-scaled once per row group, so the fused form always takes RG = 8
        rg = 8 if (norm or t >= 2) else 2
        if epilogue == "silu_mul":
            rg = min(rg, max(1, info.rows // 2))          # a row group never straddles the gate|up boundary
    rg = min(rg, info.rows)
    if info.rows % rg:
        raise ValueError(f"gemv_T: RG={rg} must divide R={info.rows}")
    preconvert = t * f.weights_per_word <= 64          # T*WPW floats of registers; beyond that convert per row
    if epilogue not in EPILOGUES:
        raise ValueError(f"gemv_T: unknown epilogue {epilogue!r}")
    macros = {"K": str(info.k), "R": str(info.rows), "T": str(t), "RG": str(rg),
              "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1",
              "UNIT_WORDS": str(info.unit_bytes // 16), "PAYLOAD_WORDS": str(payload_words),
              "SCALE_WORDS": str(scale_words), "OUT_BF16": "1" if out_bf16 else "0",
              "X_PRECONVERT": "1" if preconvert else "0",
              "NORM": "1" if norm else "0", "EPILOGUE": EPILOGUES[epilogue], "STAT_OUT": "1" if stat_out else "0"}
    if epilogue == "silu_mul":
        if info.rows % 2 or (info.rows // 2) % rg:
            raise ValueError(f"gemv_T silu_mul: R={info.rows} must be even and RG={rg} must divide R/2")
        macros["CHUNK"] = str(info.rows // 2)
    return macros


def gemv_params(n_rows: int, n_blocks: int, n_sg: int, t_active: int, *, out_scale: float = 1.0, eps: float = 0.0,
                stat_parts: int = 1) -> bytes:
    """The ``GemvParams`` record (buffer 4)."""
    return struct.pack("<IIIIffII", n_rows, n_blocks, n_sg, t_active, out_scale, eps, stat_parts, 0)


# ---- the other decode kernels -------------------------------------------------------------------------------

def embed_source() -> str:
    return PRELUDE + template("embed.metal")


def embed_macros(info: Optional[PackInfo] = None) -> Dict[str, str]:
    """``info`` = the BF16 slab a tied lm_head streams (gather from the pack), None = a row-major BF16 table."""
    if info is None:
        return {"EMBED_PACKED": "0"}
    if info.format != "bf16" or info.k % 256:
        raise ValueError("embed: a packed table must be a bf16 slab with K % 256 == 0")
    return {"EMBED_PACKED": "1", "R": str(info.rows), "UNIT_WORDS": str(info.unit_bytes // 16),
            "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1"}


def embed_params(k: int, t_active: int, vocab: int) -> bytes:
    return struct.pack("<IIII", k, t_active, vocab, 0)


def rmsnorm_stat_source() -> str:
    return PRELUDE + template("rmsnorm_stat.metal")


def stat_params(k: int, t_active: int) -> bytes:
    return struct.pack("<IIII", k, t_active, 0, 0)


def norm_apply_source() -> str:
    return PRELUDE + template("norm_apply.metal")


def norm_apply_params(k: int, t_active: int, stat_parts: int, eps: float) -> bytes:
    return struct.pack("<IIIf", k, t_active, stat_parts, eps)


def argmax_source() -> str:
    return PRELUDE + template("argmax.metal")


ARGMAX_SPAN = 256


def argmax_params(vocab: int, t_active: int, n_sg: int) -> bytes:
    return struct.pack("<IIII", vocab, t_active, n_sg, -(-vocab // ARGMAX_SPAN))


def macro_key(macros: Mapping[str, str]) -> str:
    return " ".join(f"{k}={v}" for k, v in sorted(macros.items()))
