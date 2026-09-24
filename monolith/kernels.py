"""Assembling MSL sources for the runtime compiler: kernel templates under ``kernels/`` plus the format plugins'
decode snippets and the macros that specialize them (design §5.7: block bodies + generated wrappers)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

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


# ---- attention ----------------------------------------------------------------------------------------------------

def gqa_source() -> str:
    return PRELUDE + template("gqa_decode.metal")


def gqa_macros(head_dim: int, *, chunk: int = 64, rb_max: int = 4) -> Dict[str, str]:
    """Measured on the M5 Pro (docs/research/decode-kernels.md §1): RBMAX = 4 query rows per pass is 13× faster than
    8 (register spills above 4 rows) and CH = 64 keys per chunk is the best chunk from 1 K to 32 K of context."""
    if head_dim % 32 or chunk % 32 or rb_max < 1:
        raise ValueError("gqa_decode: head_dim and chunk must be multiples of 32")
    return {"D": str(head_dim), "CH": f"{chunk}u", "RBMAX": f"{rb_max}u"}


def gqa_params(*, heads: int, kv_heads: int, t_active: int, position: int, n_sg: int, q_off: int, gate_off: int, k_off: int,
               v_off: int, in_stride: int, out_stride: int, ctx_max: int, eps: float, scaling: float, has_gate: bool,
               n_chunks_max: int, rows_max: int) -> bytes:
    """The ``GqaParams`` record (buffer 9 of gqa_decode, 4 of gqa_merge)."""
    return struct.pack("<IIIIIIIIIIIIffIIIIII", heads, kv_heads, t_active, position, n_sg, q_off, gate_off, k_off, v_off,
                       in_stride, out_stride, ctx_max, eps, scaling, 1 if has_gate else 0, n_chunks_max, rows_max, 0, 0, 0)


def gqa_workspace(kv_heads: int, n_chunks_max: int, rows_max: int, head_dim: int) -> Tuple[int, int]:
    """Bytes of the ``part_o`` and ``part_md`` workspaces."""
    n = kv_heads * n_chunks_max * rows_max
    return n * head_dim * 4, n * 2 * 4


# ---- Gated DeltaNet ----------------------------------------------------------------------------------------------

def gdn_source() -> str:
    return PRELUDE + template("gdn_mixer.metal")


def gdn_macros(dk: int, dv: int, *, conv_width: int, t: int, slice_cols: int = 8, slices_per_block: int = 4,
               tokens_per_pass: Optional[int] = None) -> Dict[str, str]:
    """Measured defaults (docs/research/decode-kernels.md §2): 8-column state slices (the register budget: 16 spills)
    and 4 slices per block (fewer, longer blocks amortize the per-block conv/norm prologue; the Hv·DV/32 blocks
    still fill the crew for Hv ≥ 16)."""
    if dk % 32 or dv % 32 or dv % (slice_cols * slices_per_block) or conv_width < 2:
        raise ValueError("gdn_mixer: dk, dv must be multiples of 32, slice_cols·slices_per_block must divide dv, conv_width >= 2")
    tp = min(t, 4) if tokens_per_pass is None else tokens_per_pass
    return {"DK": str(dk), "DV": str(dv), "CW": f"{conv_width}u", "SL": f"{slice_cols}u", "SPB": f"{slices_per_block}u", "TP": f"{tp}u"}


def gdn_workspace(t_max: int, hv: int, dv: int) -> int:
    """Bytes of the ``o_part`` workspace (FP32 read-out before the gated norm)."""
    return t_max * hv * dv * 4


def gdn_params(*, hv: int, hk: int, t_active: int, q_off: int, k_off: int, v_off: int, z_off: int, a_off: int, b_off: int,
               in_stride: int, ab_stride: int, ab_separate: bool, out_stride: int, n_sg: int, key_dim: int, eps: float) -> bytes:
    """The ``GdnParams`` record (buffer 9)."""
    return struct.pack("<IIIIIIIIIIIIIIIIffff", hv, hk, t_active, q_off, k_off, v_off, z_off, a_off, b_off, in_stride, ab_stride,
                       1 if ab_separate else 0, out_stride, n_sg, key_dim, 0, eps, 0.0, 0.0, 0.0)


# ---- the step's advance --------------------------------------------------------------------------------------------

def advance_source(step_state_msl: str) -> str:
    return PRELUDE + step_state_msl + "\n" + template("advance.metal")


def advance_params(t_active: int, ring_cap: int, eos: int) -> bytes:
    return struct.pack("<IIiI", t_active, ring_cap, eos, 0)
