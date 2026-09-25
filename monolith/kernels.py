"""Assembling MSL sources for the runtime compiler: kernel templates under ``kernels/`` plus the format plugins'
decode snippets and the macros that specialize them (design §5.7: block bodies + generated wrappers)."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

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
                epilogue: Optional[str] = None, stat_out: bool = False, round_before_residual: bool = False) -> Dict[str, str]:
    """The compile-time specialization of gemv_T for one slab geometry, token count and set of fusions
    (``norm``: RMSNorm scaling on the input; ``epilogue``: ``residual`` | ``silu_mul``; ``stat_out``: per-block
    partial sums of squares of the outputs for the next norm; ``round_before_residual``: the product is rounded to
    BF16 before the residual add — a separate BF16 linear followed by a BF16 add, the Markov head's semantics)."""
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
    if round_before_residual:
        if epilogue != "residual":
            raise ValueError("gemv_T: round_before_residual needs the residual epilogue")
        macros["EPILOGUE_ROUND"] = "1"
    return macros


def gemv_params(n_rows: int, n_blocks: int, n_sg: int, t_active: int, *, out_scale: float = 1.0, eps: float = 0.0,
                stat_parts: int = 1) -> bytes:
    """The ``GemvParams`` record (buffer 4)."""
    return struct.pack("<IIIIffII", n_rows, n_blocks, n_sg, t_active, out_scale, eps, stat_parts, 0)


# ---- the other decode kernels -------------------------------------------------------------------------------

def embed_source() -> str:
    return PRELUDE + template("embed.metal")


def embed_macros(info: Optional[PackInfo] = None, *, ids: Optional[str] = None) -> Dict[str, str]:
    """``info`` = the BF16 slab a tied lm_head streams (gather from the pack), None = a row-major BF16 table.
    ``ids="block"``: a draft block — row 0 reads the token at ``tokens[0]`` (the anchor), the other rows the mask id."""
    if info is None:
        macros = {"EMBED_PACKED": "0"}
    else:
        if info.format != "bf16" or info.k % 256:
            raise ValueError("embed: a packed table must be a bf16 slab with K % 256 == 0")
        macros = {"EMBED_PACKED": "1", "R": str(info.rows), "UNIT_WORDS": str(info.unit_bytes // 16),
                  "LANE_ORDER": "0" if info.lane_order == "contiguous" else "1"}
    if ids not in (None, "block"):
        raise ValueError(f"embed: unknown ids mode {ids!r}")
    if ids == "block":
        macros["EMBED_IDS"] = "1"
    return macros


def embed_params(k: int, t_active: int, vocab: int, mask_id: int = 0) -> bytes:
    return struct.pack("<IIII", k, t_active, vocab, mask_id)


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
               tokens_per_pass: Optional[int] = None, slots: int = 1, commit: bool = False) -> Dict[str, str]:
    """Measured defaults (docs/research/decode-kernels.md §2): 8-column state slices (the register budget: 16 spills)
    and 4 slices per block (fewer, longer blocks amortize the per-block conv/norm prologue; the Hv·DV/32 blocks
    still fill the crew for Hv ≥ 16). ``slots=2``: the states double-buffered by step parity (needs STEP_STATE);
    ``commit``: the commit pass of a speculative program (T = n_inject, rewrites the slot the step's pass wrote)."""
    if dk % 32 or dv % 32 or dv % (slice_cols * slices_per_block) or conv_width < 2:
        raise ValueError("gdn_mixer: dk, dv must be multiples of 32, slice_cols·slices_per_block must divide dv, conv_width >= 2")
    if slots not in (1, 2) or (commit and slots != 2):
        raise ValueError("gdn_mixer: slots must be 1 or 2; the commit pass needs 2 slots")
    tp = min(t, 4) if tokens_per_pass is None else tokens_per_pass
    m = {"DK": str(dk), "DV": str(dv), "CW": f"{conv_width}u", "SL": f"{slice_cols}u", "SPB": f"{slices_per_block}u", "TP": f"{tp}u"}
    if slots == 2:
        m["SLOTS"] = "2u"
    if commit:
        m["COMMIT"] = "1"
    return m


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


# ---- stochastic sampling ---------------------------------------------------------------------------------------

SAMPLE_HIST_KEYS = 65536


def sample_source() -> str:
    """argmax's helpers + the sampling kernels (one library: argmax_partial/final, sample_hist/select/gumbel)."""
    return PRELUDE + template("argmax.metal") + "\n" + template("sample.metal")


def sample_params(*, vocab: int, t_active: int, n_sg: int, top_k: int = 0, temperature: float = 1.0, top_p: float = 0.0,
                  min_p: float = 0.0, seed: int = 0, step: int = 0) -> bytes:
    """The ``SampleParams`` record (buffer 3 of the sampling kernels); ``top_k``/``top_p``/``min_p`` of 0 disable."""
    flags = (1 if top_k > 0 else 0) | (2 if 0.0 < top_p < 1.0 else 0) | (4 if min_p > 0.0 else 0)
    if temperature <= 0.0:
        raise ValueError("sample_params: temperature must be positive (use the argmax path for greedy)")
    return struct.pack("<IIIIIfffIIII", vocab, t_active, n_sg, -(-vocab // ARGMAX_SPAN), top_k, temperature, top_p, min_p,
                       seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, step, flags)


def sample_workspace(t_max: int, n_sg: int) -> Tuple[int, int, int]:
    """Bytes of the histogram, tau and partial buffers."""
    return t_max * SAMPLE_HIST_KEYS * 4, t_max * 4, t_max * n_sg * 4


# ---- the DSpark round (design §5.8; issue #24) ------------------------------------------------------------------

def spec_ops_source(step_state_msl: str) -> str:
    """tap_concat, confidence, verify_select and accept_scan (one library; the StepState struct prepended)."""
    return PRELUDE + step_state_msl + "\n" + template("spec_ops.metal")


def tap_concat_macros(n_src: int) -> Dict[str, str]:
    if not 1 <= n_src <= 8:
        raise ValueError("tap_concat: 1 to 8 sources")
    return {"N_SRC": str(n_src)}


def concat_params(k: int, t_active: int) -> bytes:
    """The ``ConcatParams`` record: columns per source (BF16, K % 8 == 0) and the row count."""
    if k % 8:
        raise ValueError("tap_concat: each source's width must be a multiple of 8")
    return struct.pack("<IIII", k, t_active, 0, 0)


def conf_params(gamma: int, hidden: int, rank: int, sts: Optional[Sequence[float]] = None) -> bytes:
    """The ``ConfParams`` record; ``sts[k]`` = the position's calibration temperature (1 = uncalibrated)."""
    t = [float(x) for x in (sts or [])]
    if len(t) > 16 or any(x <= 0 for x in t):
        raise ValueError("conf_params: up to 16 positive STS temperatures")
    t = t + [1.0] * (16 - len(t))
    return struct.pack("<IIII16f", gamma, hidden, rank, 0, *t)


CONF_LOG_WIDTH = 16


def select_params(gamma: int, threshold: float, t_max: int, mode: int = 0, cost: Optional[Sequence[float]] = None, log_cap: int = 0) -> bytes:
    """The ``SelectParams`` record: mode 0 = the confident-prefix rule (``threshold``), 1 = the cost-aware rule with
    ``cost[l]`` = the relative cost of a (1 + l)-token target pass for l = 0 … γ (≤ 16 entries; cost[0] = 1),
    2 = a fixed verify length (``threshold`` = L). ``log_cap`` > 0 logs the block's confidences per step."""
    c = list(cost or [])
    if mode == 1 and (len(c) < 1 or len(c) > 16 or abs(c[0] - 1.0) > 1e-6 or any(x <= 0 for x in c)):
        raise ValueError("select_params: the cost rule needs 1..16 positive costs relative to cost[0] = 1")
    c = c + [1.0] * (16 - len(c))
    return struct.pack("<IfII16fIIII", gamma, threshold, t_max, mode, *c, log_cap, 0, 0, 0)


ACCEPT_LOG_CAP = 65536


def accept_params(ring_cap: int, eos: int, log_cap: int = 0) -> bytes:
    return struct.pack("<IiII", ring_cap, eos, log_cap, 0)


def draft_attn_params(*, heads: int, kv_heads: int, gamma: int, ctx_len: int, n_new: int, n_sg: int, q_off: int, k_off: int, v_off: int,
                      in_stride: int, kvp_stride: int, out_stride: int, ctx_max: int, eps: float, scaling: float, n_chunks_max: int) -> bytes:
    """The ``GqaParams`` record for the DRAFT variant of gqa_decode: ``t_active`` = γ, ``position`` = the drafter's
    context length, ``pad0`` = the new context positions, ``pad1`` = the row stride of the features' k/v projection
    (buffer 11); with STEP_STATE the kernel reads position / n_new from ``drafter_ctx_len`` / ``n_inject`` instead."""
    return struct.pack("<IIIIIIIIIIIIffIIIIII", heads, kv_heads, gamma, ctx_len, n_sg, q_off, 0, k_off, v_off, in_stride, out_stride, ctx_max,
                       eps, scaling, 0, n_chunks_max, heads // kv_heads * gamma, n_new, kvp_stride, 0)
