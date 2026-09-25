"""``compile_program`` / ``emit_program``: the lowered graph → a runtime :class:`Program` (plan M4, v0).

v0 keeps every decision simple and correct: one device buffer per graph value (no aliasing except the row views the
IR declares), a barrier on every op unless the barrier pass proves it independent of the ops before it, the norm as
``rmsnorm_stat`` → ``norm_apply`` → plain GEMV (the fuse pass
that hoists the statistic into the producer's epilogue comes next), kernels specialized to the program's static ``T``
(a prefill program at ``T = P`` and a decode program at ``T = 1`` share their buffers by name), weights and constants
mapped straight from the pack file in page-aligned windows (no copy). The op handlers below are the only place an op
kind meets a kernel source; a new op kind adds a handler and a kernel, nothing else.

Row counts. A value's leading dimension is the ``T`` symbol (the step's tokens, ``StepState.t_this_step``), the
``N_INJ`` symbol (the rows the drafter injects, ``StepState.n_inject``) or a static number (a draft block of γ rows).
In a dynamic-T program every kernel reads its row count from the field the symbol names (``T_SRC``); a static value
compiles to that count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .. import kernels
from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, Op, OpClass, Value
from ..core.profile import Profile
from ..core.shapes import N_INJ, Sym, T, bind, numel, step_bindings
from ..core.step_state import StepStateLayout
from ..formats.blm import PackInfo
from ..nn.module import Model
from ..packs.packer import ALIGN, PackFile
from ..runtime.program import BufferSpec, KernelSpec, OpSpec, Program
from .barriers import place_barriers
from .coverage import check_coverage
from .passes import DEFAULT_PASSES

WINDOW_BYTES = 2 << 30          # pack windows: ICB bind offsets are 32-bit (design §5.1)
ROW_SOURCE = {T: 0, N_INJ: 1}   # the StepState field a symbolic row count reads (T_SRC): t_this_step / n_inject
STATIC_ROWS = 2
ACCEPT_LOG = "accept_log"       # the speculative program's per-step (committed << 16 | verify_len << 8 | accepted) log buffer
CONF_LOG = "conf_log"           # … and its per-step confidences (16 floats per step)
COST_FORMAT = {"fp8_e4m3": "fp8"}   # pack format -> the profile's cost_T key
FALLBACK_THRESHOLD = 0.5            # the confident-prefix threshold when no cost table exists (verifying the whole block costs ×5 at T = 8)


@dataclass
class _Ctx:
    program: Program
    packs: List[PackFile]
    layout: StepStateLayout
    t: int
    n_sg: int
    tg: int
    values: Dict[str, Value] = field(default_factory=dict)
    windows: Dict[str, Tuple[str, int]] = field(default_factory=dict)      # pack entry name -> (buffer, offset)
    row_scales: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    stat_parts: Dict[str, int] = field(default_factory=dict)              # statistic value -> partial sums per token
    dynamic_t: bool = False                                               # T from StepState (prefill chunks); else static
    speculative: bool = False                                             # the round is in the program: per-T GEMV variants
    attention: str = "v1"                                                 # the attention kernel (profile / override)
    tuner: Any = None                                                     # compiler.autotune.Autotuner or None
    shared: Dict[str, str] = field(default_factory=dict)                  # shared scratch name -> buffer (sized to the largest request)
    eos: int = -1
    ring_capacity: int = 4096
    counter: int = 0
    norm_scratch: Dict[Tuple[str, str, int], str] = field(default_factory=dict)   # (x, stat, rows) -> the normalized scratch

    # ---- helpers -----------------------------------------------------------------------------------------------
    def slab_info(self, name: str) -> PackInfo:
        for pk in self.packs:
            if name in pk.slabs:
                return pk.slab_info(name)
        raise KeyError(f"emit: no pack holds the slab {name!r}")

    def kernel(self, key: str, source: str, function: str, macros: Dict[str, str]) -> str:
        macros = dict(macros)
        if self.dynamic_t:
            macros["STEP_STATE"] = "1"
        if "struct StepState" not in source:
            source = source.replace(kernels.PRELUDE, kernels.PRELUDE + self.layout.to_msl() + "\n", 1)
        k = f"{function}|{key}|{kernels.macro_key(macros)}"
        if k not in self.program.kernels:
            self.program.kernels[k] = KernelSpec(source, function, macros)
        return k

    def params(self, name: str, data: bytes) -> str:
        bname = f"params.T{self.t}.{name}.{self.counter}"          # per-T: programs share buffers by name, params must not
        self.counter += 1
        self.program.buffers[bname] = BufferSpec(len(data), data, "params")
        return bname

    def scratch(self, name: str, nbytes: int, shared: bool = False) -> str:
        """A workspace buffer; ``shared`` = one buffer per name for the whole program (the partials an op hands to
        its follow-up dispatch: every layer's attention or GDN core reuses it, the barrier pass orders the reuse)."""
        if shared:
            bname = self.shared.get(name)
            if bname is None:
                bname = f"ws.T{self.t}.{name}.shared"
                self.shared[name] = bname
                self.program.buffers[bname] = BufferSpec(max(nbytes, 16), None, "arena")
            elif self.program.buffers[bname].nbytes < nbytes:
                self.program.buffers[bname].nbytes = nbytes
            return bname
        bname = f"ws.T{self.t}.{name}.{self.counter}"
        self.counter += 1
        self.program.buffers[bname] = BufferSpec(max(nbytes, 16), None, "arena")
        return bname

    def crew_grid(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        return (-(-(self.n_sg * 32) // self.tg), 1, 1), (self.tg, 1, 1)

    def geometry(self, mode: str, n_blocks: int) -> Tuple[int, Tuple[int, int, int], Tuple[int, int, int]]:
        """(n_sg, grid, threadgroup) for an autotuned geometry mode."""
        if mode == "block":
            return n_blocks, (-(-(n_blocks * 32) // 64), 1, 1), (64, 1, 1)
        n_sg = self.n_sg * (2 if mode == "crew2" else 1)
        return n_sg, (-(-(n_sg * 32) // self.tg), 1, 1), (self.tg, 1, 1)

    def add(self, kernel: str, bindings: List[Tuple[int, str, int]], grid, tg, name: str, *, writes: Optional[Sequence[int]] = None,
            **meta: Any) -> None:
        """Append a dispatch; ``writes`` = the binding indices the kernel writes (the barrier pass reads them; an
        op without the record is taken to write everything it binds)."""
        if self.dynamic_t and name not in ("advance", "accept_scan", "verify_select") and not any(b[0] == 15 for b in bindings):
            bindings = list(bindings) + [(15, self.program.step_state, 0)]
        if writes is not None:
            meta["writes"] = sorted(int(i) for i in writes)
        self.program.ops.append(OpSpec(kernel, bindings, tuple(grid), tuple(tg), True, [], name, dict(meta)))

    def shape(self, v: Value) -> Tuple[int, ...]:
        return bind(v.shape, step_bindings(self.t))

    def rows_of(self, op: Op) -> Tuple[int, int]:
        """``(rows compiled, T source)`` of an op from its first output's leading dimension: the ``T`` symbol →
        ``t_this_step`` (0), ``N_INJ`` → ``n_inject`` (1), a number → that many rows, static (2)."""
        d = op.outputs[0].shape[0]
        if isinstance(d, Sym):
            if d not in ROW_SOURCE:
                raise ValueError(f"emit: {op!r} has an unknown row symbol {d}")
            return self.t, ROW_SOURCE[d]
        return int(d), STATIC_ROWS

    def t_macros(self, t_c: int, t_src: int) -> Dict[str, str]:
        """The row-source macros of a dynamic-T program (a static program takes the row count from its params)."""
        if not self.dynamic_t:
            return {}
        m = {"T_SRC": str(t_src)}
        if t_src == STATIC_ROWS:
            m["T_STATIC_ROWS"] = f"{t_c}u"
        return m

    def buf(self, v: Value) -> Tuple[str, int]:
        """The ``(buffer, byte offset)`` a value binds to: a view → rows of its base; an input named after a StepState
        field (``tokens`` = ``pending_tokens``, ``anchor`` …) → that field; a pack entry → its window."""
        if v.view_of is not None:
            base_name, row = v.view_of
            b, off = self.buf(self.values[base_name])
            return b, off + row * numel(v.shape[1:], step_bindings(self.t)) * v.dtype.itemsize
        if v.is_input:
            fld = "pending_tokens" if v.name == "tokens" else v.name
            if fld in self.layout.offsets:
                return self.program.step_state, self.layout.offset(fld)
            return v.name, 0
        if v.is_weight or v.is_const:
            return self.windows[v.name]
        return v.name, 0


def _value_bytes(v: Value, t: int) -> int:
    return numel(v.shape, step_bindings(t)) * v.dtype.itemsize


def _pack_windows(ctx: _Ctx) -> None:
    """Group each pack's slabs, row-scale tables and aux entries into ≤ 2 GiB file-backed windows (the target's pack
    first, then a drafter's; entry names are distinct across them)."""
    n_win = 0
    for pk in ctx.packs:
        path = pk.dir / pk.manifest["pack"]
        entries: List[Tuple[int, int, str, str]] = []                       # (offset, nbytes, kind, name)
        for s in pk.manifest["slabs"]:
            entries.append((s["offset"], s["nbytes"], "slab", s["name"]))
            entries.append((s["row_scales_offset"], 4 * s["n"], "rs", s["name"]))
        for a in pk.manifest["aux"]:
            entries.append((a["offset"], a["nbytes"], "aux", a["name"]))
        entries.sort()
        start, end, members = None, 0, []
        windows: List[Tuple[int, int, list]] = []
        for off, nb, kind, name in entries:
            if start is None or off + nb - start > WINDOW_BYTES:
                if start is not None:
                    windows.append((start, end, members))
                start, end, members = off - off % ALIGN, off + nb, []      # mapped windows start and end on page boundaries
            end = max(end, off + nb)
            members.append((off, kind, name))
        if start is not None:
            windows.append((start, end, members))
        file_bytes = int(pk.manifest["nbytes"])                              # the packer pads the file to ALIGN
        for start, end, members in windows:
            bname = f"pack.{n_win}"
            n_win += 1
            end = min(-(-end // ALIGN) * ALIGN, file_bytes)
            ctx.program.buffers[bname] = BufferSpec(end - start, None, "weights", str(path), start)
            for off, kind, name in members:
                table = ctx.row_scales if kind == "rs" else ctx.windows
                if name in table:
                    raise ValueError(f"emit: the packs both hold an entry named {name!r}")
                table[name] = (bname, off - start)


# ---- op handlers ---------------------------------------------------------------------------------------------------

def _embed(ctx: _Ctx, op: Op) -> None:
    tokens, table = op.inputs
    h = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    info = ctx.slab_info(table.name)
    macros = dict(kernels.embed_macros(info, ids=op.attrs.get("ids")), **ctx.t_macros(t_c, t_src))
    k = ctx.kernel(f"embed|{info.format}", kernels.embed_source(info.format), "embed", macros)
    prm = ctx.params("embed", kernels.embed_params(info.k, t_c, info.n, mask_id=int(op.attrs.get("mask_id", 0))))
    ctx.add(k, [(0, *ctx.buf(tokens)), (1, *ctx.windows[table.name]), (2, *ctx.buf(h)), (3, prm, 0)], (t_c, 1, 1), (32, 1, 1), op.kind, writes=[2])


def _rmsnorm_stat(ctx: _Ctx, op: Op) -> None:
    h, = op.inputs
    stat = op.outputs[0]
    if op.attrs.get("hoisted"):
        return                                        # the producer GEMV writes the partials (fuse_norm_stat)
    t_c, t_src = ctx.rows_of(op)
    k = ctx.kernel("rmsnorm_stat", kernels.rmsnorm_stat_source(), "rmsnorm_stat", ctx.t_macros(t_c, t_src))
    prm = ctx.params("stat", kernels.stat_params(ctx.shape(h)[1], t_c))
    ctx.add(k, [(0, *ctx.buf(h)), (1, *ctx.buf(stat)), (2, prm, 0)], (t_c, 1, 1), (32, 1, 1), op.kind, writes=[1])


def _norm_apply(ctx: _Ctx, h: Value, stat: Value, nw: Value, eps: float, out: Tuple[str, int], t_c: int, t_src: int,
                name: str = "norm_apply") -> None:
    k = ctx.kernel("norm_apply", kernels.norm_apply_source(), "norm_apply", ctx.t_macros(t_c, t_src))
    kdim = ctx.shape(h)[1]
    prm = ctx.params("norm_apply", kernels.norm_apply_params(kdim, t_c, ctx.stat_parts.get(stat.name, 1), eps))
    ctx.add(k, [(0, *ctx.buf(h)), (1, *ctx.buf(stat)), (2, *ctx.windows[nw.name]), (3, *out), (4, prm, 0)], (t_c, 1, 1), (32, 1, 1), name, writes=[3])


def _norm_apply_op(ctx: _Ctx, op: Op) -> None:
    """An explicit ``norm_apply`` op: the normalized activation as a graph value (shared by several consumers)."""
    h, stat, nw = op.inputs
    t_c, t_src = ctx.rows_of(op)
    _norm_apply(ctx, h, stat, nw, float(op.attrs.get("eps", 1e-6)), ctx.buf(op.outputs[0]), t_c, t_src, op.kind)


def t_variants(t_max: int) -> List[int]:
    """The predicated per-T variants of a GEMV in a speculative program: the powers of two up to ``t_max`` and
    ``t_max`` itself; variant ``T_v`` runs for ``T_v/2 < T ≤ T_v`` (the M1 study measured the kernel at these T)."""
    out = []
    v = 1
    while v < t_max:
        out.append(v)
        v *= 2
    return out + [t_max]


def _gemv(ctx: _Ctx, op: Op) -> None:
    """One GEMV op → one dispatch, or in a speculative program with a symbolic row count (the target's verify pass,
    the drafter's injection) one predicated dispatch per T variant: every variant is compiled for its own T (code
    shape and autotuned geometry) and returns at once unless the step's T falls in its range, so the ALU work of the
    ALU-bound formats follows the actual T instead of ``t_max`` (design §5.7)."""
    ins = list(op.inputs)
    x, w = ins[0], ins[1]
    rest = ins[2:]
    stat = nw = residual = None
    if op.attrs.get("norm"):
        stat, nw = rest[0], rest[1]
        rest = rest[2:]
    if op.attrs.get("epilogue") == "residual":
        residual = rest[0]
    y = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    info = ctx.slab_info(w.name)
    # a row range of the slab: whole blocks only (a mixer's gate rows, design §5.12)
    rr = op.attrs.get("row_range")
    if rr is not None:
        start, count = int(rr[0]), int(rr[1])
        if start % info.rows or start + count > info.n or (count % info.rows and start + count != info.n):
            raise ValueError(f"gemv {w.name}: row range {rr} must start on a block of {info.rows} rows and end on one or at the slab's end ({info.n})")
        block0, n_blocks, n_rows = start // info.rows, -(-count // info.rows), count
        nbytes = int(info.nbytes) * n_blocks // info.n_blocks
    else:
        block0, n_blocks, n_rows, nbytes = 0, info.n_blocks, info.n, int(info.nbytes)
    variants = t_variants(t_c) if (ctx.speculative and ctx.dynamic_t and t_src != STATIC_ROWS and t_c > 1) else [t_c]
    epilogue = op.attrs.get("epilogue")
    choices = [ctx.tuner.tune_gemv(info, tv, epilogue, stat is not None) if ctx.tuner is not None else None for tv in variants]
    fuse_norm = bool(choices[0]) and all(c is not None and c.fuse_norm for c in choices)
    eps = float(op.attrs.get("eps", 1e-6))
    x_binding = ctx.buf(x)
    if stat is not None and not fuse_norm:
        key = (x.name, stat.name, t_c)
        xn = ctx.norm_scratch.get(key)                    # one normalized copy per (input, statistic): siblings share it
        if xn is None:
            xn = ctx.scratch(f"{y.name}.xn", t_c * ctx.shape(x)[1] * 2)
            _norm_apply(ctx, x, stat, nw, eps, (xn, 0), t_c, t_src)
            ctx.norm_scratch[key] = xn
        x_binding = (xn, 0)
    stat_out = op.attrs.get("stat_value")
    if stat_out is not None:
        ctx.stat_parts[stat_out] = n_blocks
    ctx.counter += 1
    vgroup = ctx.counter if len(variants) > 1 else None
    lo = 0
    for tv, choice in zip(variants, choices):
        rg = int(choice.macros["RG"]) if choice else None
        macros = dict(kernels.gemv_macros(info, t=tv, rg=rg, epilogue=epilogue, out_bf16=True, stat_out=stat_out is not None,
                                          norm=fuse_norm, round_before_residual=bool(op.attrs.get("round_residual"))), **ctx.t_macros(tv, t_src))
        if len(variants) > 1:
            macros["T_LO"], macros["T_HI"] = str(lo), str(tv)
        k = ctx.kernel(f"gemv_T|{info.format}", kernels.gemv_source(info.format), "gemv_T", macros)
        n_sg, grid, tg = ctx.geometry(choice.grid_mode if choice else "crew", n_blocks)
        prm = ctx.params("gemv", kernels.gemv_params(n_rows, n_blocks, n_sg, tv, eps=eps, block0=block0,
                                                     stat_parts=ctx.stat_parts.get(stat.name, 1) if stat is not None else 1))
        bindings = [(0, *ctx.windows[w.name]), (1, *ctx.row_scales[w.name]), (2, *x_binding), (3, *ctx.buf(y)), (4, prm, 0)]
        writes = [3]
        if fuse_norm:
            bindings += [(5, *ctx.buf(stat)), (6, *ctx.windows[nw.name])]
        if residual is not None:
            bindings.append((7, *ctx.buf(residual)))
        if stat_out is not None:
            bindings.append((8, stat_out, 0))
            writes.append(8)
        ctx.add(k, bindings, grid, tg, f"{op.kind}:{w.name}", writes=writes, kind=op.kind, bytes=nbytes, format=info.format, n=n_rows, k=info.k,
                rg=int(macros["RG"]), geometry=choice.grid_mode if choice else "crew", fused_norm=fuse_norm, t_variant=tv,
                t_range=[lo, tv] if len(variants) > 1 else None, variant_group=vgroup, sibling=bool(op.attrs.get("sibling")),
                row_range=[block0 * info.rows, n_rows] if rr is not None else None)
        lo = tv


def _gqa_src(ctx: _Ctx, v2: bool = False) -> str:
    return kernels.PRELUDE + ctx.layout.to_msl() + "\n" + kernels.template("gqa_common.metal") + "\n" + kernels.template(
        "gqa_decode_v2.metal" if v2 else "gqa_decode.metal")


def _gqa_v2(ctx: _Ctx, heads: int, kv: int) -> bool:
    """v2 when the profile (or the override) asks and the block's rows fit its query cache."""
    return ctx.attention == "v2" and (heads // kv) * ctx.t <= 32


def _gqa_geometry(ctx: _Ctx, a: Dict[str, Any], ctx_max: int, v2: bool):
    """(macros, n_sg field, chunk for the workspace)."""
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    rep = heads // kv
    if v2:
        return dict(kernels.gqa_v2_macros(d, rmax=rep * ctx.t, rg=4), STEP_STATE="1"), ctx.n_sg // 12, kernels.GQA_V2_CHUNK_MIN
    chunk = int(a.get("chunk", 64))
    return dict(kernels.gqa_macros(d, chunk=chunk), STEP_STATE="1"), ctx.n_sg, chunk


def _gqa(ctx: _Ctx, op: Op) -> None:
    """The attention core: partials per (kv head, chunk, row) into the op's two output values (a shared workspace
    across the layers; the barrier pass orders its reuse)."""
    proj, kc, vc, cos, sin, qn, kn = op.inputs
    part_o, part_md = op.outputs
    a = op.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    segs = {name: (off, n) for name, off, n in a["segments"]}
    ctx_max = ctx.shape(kc)[0]
    v2 = _gqa_v2(ctx, heads, kv)
    macros, n_sg, chunk = _gqa_geometry(ctx, a, ctx_max, v2)
    kd = ctx.kernel("gqa", _gqa_src(ctx, v2), "gqa_decode_v2" if v2 else "gqa_decode", macros)
    rep = heads // kv
    n_chunks_max, rows_max = -(-ctx_max // chunk), rep * ctx.t
    po, pm = kernels.gqa_workspace(kv, n_chunks_max, rows_max, d)
    if _value_bytes(part_o, ctx.t) < po or _value_bytes(part_md, ctx.t) < pm:
        raise ValueError(f"gqa_decode: the partial values are too small for {kv} kv heads × {n_chunks_max} chunks × {rows_max} rows")
    prm = ctx.params("gqa", kernels.gqa_params(
        heads=heads, kv_heads=kv, t_active=ctx.t, position=0, n_sg=n_sg, q_off=segs["q"][0], gate_off=0, k_off=segs["k"][0],
        v_off=segs["v"][0], in_stride=ctx.shape(proj)[1], out_stride=heads * d, ctx_max=ctx_max, eps=float(a["eps"]),
        scaling=float(a["scaling"]), has_gate=False, n_chunks_max=n_chunks_max, rows_max=rows_max))
    st = ctx.program.step_state
    grid, tg = ctx.crew_grid()
    ctx.add(kd, [(0, *ctx.buf(proj)), (1, *ctx.buf(kc)), (2, *ctx.buf(vc)), (3, *ctx.windows[cos.name]), (4, *ctx.windows[sin.name]),
                 (5, *ctx.windows[qn.name]), (6, *ctx.windows[kn.name]), (7, *ctx.buf(part_o)), (8, *ctx.buf(part_md)), (9, prm, 0), (15, st, 0)],
            grid, tg, op.kind, writes=[1, 2, 7, 8], attention="v2" if v2 else "v1")


def _gqa_merge(ctx: _Ctx, op: Op) -> None:
    """The fold over the chunks, times σ(gate) when the gate projection value is given (its own dispatch, so the
    gate GEMV can run beside the core)."""
    part_o, part_md = op.inputs[0], op.inputs[1]
    gate = op.inputs[2] if len(op.inputs) > 2 else None
    out = op.outputs[0]
    a = op.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    v2 = _gqa_v2(ctx, heads, kv)
    rep = heads // kv
    core = part_o.producer
    ctx_max = ctx.shape(core.inputs[1])[0] if core is not None else 0
    macros, n_sg, chunk = _gqa_geometry(ctx, dict(a, chunk=core.attrs.get("chunk", 64) if core is not None else 64), ctx_max, v2)
    km = ctx.kernel("gqa", _gqa_src(ctx, v2), "gqa_merge_v2" if v2 else "gqa_merge", macros)
    n_chunks_max = ctx.shape(part_o)[1] // (kv * rep * d)
    prm = ctx.params("gqa_merge", kernels.gqa_params(
        heads=heads, kv_heads=kv, t_active=ctx.t, position=0, n_sg=n_sg, q_off=0, gate_off=0, k_off=0, v_off=0,
        in_stride=heads * d, out_stride=heads * d, ctx_max=n_chunks_max * chunk, eps=1e-6, scaling=1.0, has_gate=gate is not None,
        n_chunks_max=n_chunks_max, rows_max=rep * ctx.t))
    st = ctx.program.step_state
    gb = ctx.buf(gate) if gate is not None else ctx.buf(part_o)
    ctx.add(km, [(0, *ctx.buf(part_o)), (1, *ctx.buf(part_md)), (2, *gb), (3, *ctx.buf(out)), (4, prm, 0), (15, st, 0)],
            (ctx.t * heads, 1, 1), (32, 1, 1), op.kind, writes=[3])


def _draft_attn(ctx: _Ctx, op: Op) -> None:
    """The drafter's block attention: gqa_decode's DRAFT variant + gqa_merge (design §5.8). The context length and the
    number of injected positions come from StepState (``drafter_ctx_len``, ``n_inject``); γ is static."""
    proj, kvp, kc, vc, cos, sin, qn, kn = op.inputs
    out = op.outputs[0]
    a = op.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    gamma = ctx.shape(proj)[0]
    if gamma > ctx.layout.gamma_max:
        raise ValueError(f"draft_attn: a block of {gamma} rows exceeds the layout's gamma_max {ctx.layout.gamma_max}")
    ctx_max = ctx.shape(kc)[0]
    chunk = 64
    macros = dict(kernels.gqa_macros(d, chunk=chunk), DRAFT="1", STEP_STATE="1")
    src = _gqa_src(ctx)
    kd, km = ctx.kernel("gqa", src, "gqa_decode", macros), ctx.kernel("gqa", src, "gqa_merge", macros)
    rep = heads // kv
    n_chunks_max, rows_max = -(-ctx_max // chunk), rep * gamma
    po, pm = kernels.gqa_workspace(kv, n_chunks_max, rows_max, d)
    part_o, part_md = ctx.scratch("draft_attn.part_o", po, shared=True), ctx.scratch("draft_attn.part_md", pm, shared=True)
    prm = ctx.params("draft_attn", kernels.draft_attn_params(
        heads=heads, kv_heads=kv, gamma=gamma, ctx_len=0, n_new=0, n_sg=ctx.n_sg, q_off=0, k_off=heads * d, v_off=(heads + kv) * d,
        in_stride=ctx.shape(proj)[1], kvp_stride=ctx.shape(kvp)[1], out_stride=heads * d, ctx_max=ctx_max, eps=float(a["eps"]),
        scaling=float(a["scaling"]), n_chunks_max=n_chunks_max))
    st = ctx.program.step_state
    grid, tg = ctx.crew_grid()
    ctx.add(kd, [(0, *ctx.buf(proj)), (1, *ctx.buf(kc)), (2, *ctx.buf(vc)), (3, *ctx.windows[cos.name]), (4, *ctx.windows[sin.name]),
                 (5, *ctx.windows[qn.name]), (6, *ctx.windows[kn.name]), (7, part_o, 0), (8, part_md, 0), (9, prm, 0), (11, *ctx.buf(kvp)),
                 (15, st, 0)], grid, tg, op.kind, writes=[1, 2, 7, 8])
    ctx.add(km, [(0, part_o, 0), (1, part_md, 0), (2, *ctx.buf(proj)), (3, *ctx.buf(out)), (4, prm, 0), (15, st, 0)],
            (gamma * heads, 1, 1), (32, 1, 1), "gqa_merge", writes=[3])


def _gdn_macros(ctx: _Ctx, a: Dict[str, Any], commit: bool) -> Dict[str, str]:
    hv, hk, dk, dv, cw = a["v_heads"], a["k_heads"], a["dk"], a["dv"], a["conv_width"]
    gch = ctx.tuner.tune_gdn(hv, hk, dk, dv, cw, ctx.t) if ctx.tuner is not None else None
    return dict(kernels.gdn_macros(dk, dv, conv_width=cw, t=ctx.t, slice_cols=int(str(gch.macros["SL"]).rstrip("u")) if gch else 8,
                                   slices_per_block=int(str(gch.macros["SPB"]).rstrip("u")) if gch else 4, slots=2, commit=commit),
                STEP_STATE="1")


def _gdn(ctx: _Ctx, op: Op) -> None:
    """The GDN core (the FP32 read-out into the op's output value) or, for ``gdn_commit``, the commit pass. The
    states live in two slots by step parity, so the kernel always reads StepState (like the attention's position)."""
    a = op.attrs
    commit = op.kind == "gdn_commit"
    n_proj = len(a["proj_segments"]) and (1 + max(idx for idx, _, _ in a["proj_segments"].values()))
    projs = op.inputs[:n_proj]
    cs, rs, conv_w, a_log, dt_bias = op.inputs[n_proj:]
    o_part = op.outputs[0]
    hv, hk, dk, dv = a["v_heads"], a["k_heads"], a["dk"], a["dv"]
    ps = a["proj_segments"]                                           # local -> (value index, column offset, columns)
    kd = hk * dk
    ab_separate = ps["in_proj_a"][0] != ps["in_proj_qkv"][0]
    if ctx.shape(cs)[0] != 2 or ctx.shape(rs)[0] != 2:
        raise ValueError(f"gdn_mixer: the states need two slots (StateEntry.checkpoints = 2), got {ctx.shape(cs)} / {ctx.shape(rs)}")
    macros = _gdn_macros(ctx, a, commit)
    kmix = ctx.kernel("gdn", kernels.gdn_source(), "gdn_mixer", macros)
    main, abv = projs[ps["in_proj_qkv"][0]], projs[ps["in_proj_a"][0]]
    prm = ctx.params("gdn", kernels.gdn_params(
        hv=hv, hk=hk, t_active=ctx.t, q_off=ps["in_proj_qkv"][1], k_off=ps["in_proj_qkv"][1] + kd, v_off=ps["in_proj_qkv"][1] + 2 * kd,
        z_off=0, a_off=ps["in_proj_a"][1], b_off=ps["in_proj_b"][1], in_stride=ctx.shape(main)[1],
        ab_stride=ctx.shape(abv)[1], ab_separate=ab_separate, out_stride=hv * dv, n_sg=ctx.n_sg, key_dim=kd, eps=float(a["eps"])))
    st = ctx.program.step_state
    grid, tg = ctx.crew_grid()
    ctx.add(kmix, [(0, *ctx.buf(main)), (1, *ctx.buf(abv)), (2, *ctx.buf(cs)), (3, *ctx.buf(rs)), (4, *ctx.windows[conv_w.name]),
                   (5, *ctx.windows[a_log.name]), (6, *ctx.windows[dt_bias.name]), (7, *ctx.buf(o_part)), (9, prm, 0), (15, st, 0)],
            grid, tg, op.kind, writes=[2, 3, 7])


def _gdn_norm(ctx: _Ctx, op: Op) -> None:
    """The gated RMSNorm over the read-out: ``z`` comes from its own value (the gate GEMV, the core's sibling)."""
    o_part, z, norm_w = op.inputs
    out = op.outputs[0]
    a = op.attrs
    hv, dv = a["v_heads"], a["dv"]
    attrs = dict(a, k_heads=1, dk=32, conv_width=2)                  # the norm kernel only needs DV (and the shared macros)
    core = op.inputs[0].producer
    macros = _gdn_macros(ctx, core.attrs if core is not None else attrs, False)
    knorm = ctx.kernel("gdn", kernels.gdn_source(), "gdn_norm", macros)
    prm = ctx.params("gdn_norm", kernels.gdn_params(
        hv=hv, hk=1, t_active=ctx.t, q_off=0, k_off=0, v_off=0, z_off=0, a_off=0, b_off=0, in_stride=ctx.shape(z)[1], ab_stride=ctx.shape(z)[1],
        ab_separate=False, out_stride=hv * dv, n_sg=ctx.n_sg, key_dim=0, eps=float(a["eps"])))
    st = ctx.program.step_state
    ctx.add(knorm, [(0, *ctx.buf(o_part)), (1, *ctx.buf(z)), (2, *ctx.windows[norm_w.name]), (3, *ctx.buf(out)), (4, prm, 0), (15, st, 0)],
            (ctx.t * hv, 1, 1), (32, 1, 1), op.kind, writes=[3])


def _argmax(ctx: _Ctx, op: Op) -> None:
    logits, = op.inputs
    token = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    vocab = ctx.shape(logits)[1]
    src = kernels.argmax_source()
    m = ctx.t_macros(t_c, t_src)
    kp, kf = ctx.kernel("argmax", src, "argmax_partial", m), ctx.kernel("argmax", src, "argmax_final", m)
    pv, pi = ctx.scratch("argmax.val", t_c * ctx.n_sg * 4, shared=True), ctx.scratch("argmax.idx", t_c * ctx.n_sg * 4, shared=True)
    prm = ctx.params("argmax", kernels.argmax_params(vocab, t_c, ctx.n_sg))
    grid, tg = ctx.crew_grid()
    ctx.add(kp, [(0, *ctx.buf(logits)), (1, pv, 0), (2, pi, 0), (3, prm, 0)], grid, tg, op.kind, writes=[1, 2])
    ctx.add(kf, [(0, pv, 0), (1, pi, 0), (2, *ctx.buf(token)), (3, prm, 0)], (t_c, 1, 1), (32, 1, 1), "argmax_final", writes=[2])


def _sample(ctx: _Ctx, op: Op) -> None:
    logits, = op.inputs
    token = op.outputs[0]
    vocab = ctx.shape(logits)[1]
    a = op.attrs
    src = kernels.sample_source()
    macros = {"STEP_STATE": "1"}                                       # seed and step always come from StepState
    kh, ks, kg, kf = (ctx.kernel("sample", src, f, macros) for f in ("sample_hist", "sample_select", "sample_gumbel", "argmax_final"))
    hb, tb, pb = kernels.sample_workspace(ctx.t, ctx.n_sg)
    hist, tau = ctx.scratch("sample.hist", hb), ctx.scratch("sample.tau", tb)
    pv, pi = ctx.scratch("sample.val", pb), ctx.scratch("sample.idx", pb)
    prm = ctx.params("sample", kernels.sample_params(vocab=vocab, t_active=ctx.t, n_sg=ctx.n_sg, top_k=int(a.get("top_k", 0)),
                                                     temperature=float(a.get("temperature", 1.0)), top_p=float(a.get("top_p", 0.0)),
                                                     min_p=float(a.get("min_p", 0.0)), seed=int(a.get("seed", 0)), step=0))
    st = ctx.program.step_state
    grid, tg = ctx.crew_grid()
    ctx.add(kh, [(0, *ctx.buf(logits)), (1, hist, 0), (3, prm, 0), (15, st, 0)], grid, tg, op.kind, writes=[1])
    ctx.add(ks, [(1, hist, 0), (2, tau, 0), (3, prm, 0), (15, st, 0)], (ctx.t, 1, 1), (32, 1, 1), "sample_select", writes=[2])
    ctx.add(kg, [(0, *ctx.buf(logits)), (2, tau, 0), (3, prm, 0), (4, pv, 0), (5, pi, 0), (15, st, 0)], grid, tg, "sample_gumbel", writes=[4, 5])
    ctx.add(kf, [(0, pv, 0), (1, pi, 0), (2, *ctx.buf(token)), (3, prm, 0), (15, st, 0)], (ctx.t, 1, 1), (32, 1, 1), "argmax_final", writes=[2])


# ---- the DSpark round (design §5.8; issue #24) ------------------------------------------------------------------

def _spec_ops(ctx: _Ctx) -> str:
    return kernels.spec_ops_source(ctx.layout.to_msl())


def _tap_concat(ctx: _Ctx, op: Op) -> None:
    taps = list(op.inputs)
    x = op.outputs[0]
    t_c, t_src = ctx.rows_of(op)
    k_each = ctx.shape(taps[0])[1]
    if any(ctx.shape(v)[1] != k_each for v in taps):
        raise ValueError("tap_concat: every tap must have the same width")
    macros = dict(kernels.tap_concat_macros(len(taps)), **ctx.t_macros(t_c, t_src))
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "tap_concat", macros)
    prm = ctx.params("tap_concat", kernels.concat_params(k_each, t_c))
    bindings = [(i, *ctx.buf(taps[min(i, len(taps) - 1)])) for i in range(8)]     # unused slots bound to a valid buffer
    bindings += [(8, *ctx.buf(x)), (9, prm, 0)]
    ctx.add(k, bindings, (t_c * len(taps), 1, 1), (32, 1, 1), op.kind, writes=[8])


def _confidence(ctx: _Ctx, op: Op) -> None:
    hidden, emb, w, b = op.inputs
    conf = op.outputs[0]
    gamma, hid = ctx.shape(hidden)
    rank = int(op.attrs.get("rank", 0))
    if rank and ctx.shape(emb) != (gamma, rank):
        raise ValueError(f"confidence: emb must be [{gamma}, {rank}], got {ctx.shape(emb)}")
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "confidence", {})
    prm = ctx.params("confidence", kernels.conf_params(gamma, hid, rank, sts=op.attrs.get("sts")))
    ctx.add(k, [(0, *ctx.buf(hidden)), (1, *ctx.buf(emb)), (2, *ctx.windows[w.name]), (3, *ctx.windows[b.name]), (4, *ctx.buf(conf)), (5, prm, 0)],
            (gamma, 1, 1), (32, 1, 1), op.kind, writes=[4])


def _verify_select(ctx: _Ctx, op: Op) -> None:
    drafts = op.inputs[0]
    conf = op.inputs[1] if len(op.inputs) > 1 else None
    gamma = int(op.attrs["gamma"])
    if gamma > ctx.layout.gamma_max or gamma + 1 > ctx.layout.t_max:
        raise ValueError(f"verify_select: gamma {gamma} exceeds the layout (gamma_max {ctx.layout.gamma_max}, t_max {ctx.layout.t_max})")
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "verify_select", {})
    cost = op.attrs.get("cost") if conf is not None else None
    fixed = op.attrs.get("fixed")
    if fixed is not None:
        mode, thr = 2, float(int(fixed))
    elif cost:
        mode, thr = 1, float(op.attrs.get("threshold", 0.0))
    else:
        mode, thr = 0, (float(op.attrs.get("threshold", 0.0)) if conf is not None else 0.0)
    prm = ctx.params("verify_select", kernels.select_params(gamma, thr, ctx.layout.t_max, mode=mode, cost=cost, log_cap=kernels.ACCEPT_LOG_CAP))
    ctx.program.buffers.setdefault(CONF_LOG, BufferSpec(kernels.ACCEPT_LOG_CAP * kernels.CONF_LOG_WIDTH * 4, None, "arena"))
    cb = ctx.buf(conf) if conf is not None else (ctx.scratch("verify_select.conf", gamma * 4), 0)
    ctx.add(k, [(0, *ctx.buf(drafts)), (1, *cb), (2, ctx.program.step_state, 0), (3, prm, 0), (4, CONF_LOG, 0)], (1, 1, 1), (32, 1, 1), op.kind,
            writes=[2, 4])


def _accept_scan(ctx: _Ctx, op: Op) -> None:
    token, = op.inputs
    k = ctx.kernel("spec_ops", _spec_ops(ctx), "accept_scan", {})
    prm = ctx.params("accept_scan", kernels.accept_params(ctx.ring_capacity, ctx.eos, kernels.ACCEPT_LOG_CAP))
    ctx.program.buffers.setdefault(ACCEPT_LOG, BufferSpec(kernels.ACCEPT_LOG_CAP * 4, None, "arena"))
    ctx.add(k, [(0, *ctx.buf(token)), (1, ctx.program.step_state, 0), (2, ctx.program.ring, 0), (3, prm, 0), (4, ACCEPT_LOG, 0)],
            (1, 1, 1), (32, 1, 1), op.kind, writes=[1, 2, 4])


HANDLERS = {"embed": _embed, "rmsnorm_stat": _rmsnorm_stat, "norm_apply": _norm_apply_op, "gemv": _gemv, "lm_head": _gemv,
            "gqa_decode": _gqa, "gqa_merge": _gqa_merge, "gdn_mixer": _gdn, "gdn_commit": _gdn, "gdn_norm": _gdn_norm, "argmax": _argmax,
            "sample": _sample,
            "tap_concat": _tap_concat, "draft_attn": _draft_attn, "confidence": _confidence, "verify_select": _verify_select,
            "accept_scan": _accept_scan}


def emit_program(g: Graph, *, pack: Union[PackFile, Sequence[PackFile]], profile: Profile, t: Optional[int] = None, dynamic_t: bool = False,
                 layout: Optional[StepStateLayout] = None, eos: int = -1, ring_capacity: int = 4096, tg: int = 384, tuner: Any = None,
                 tail: Optional[str] = "advance", token: Optional[Value] = None, speculative: bool = False, barriers: str = "minimal",
                 attention: Optional[str] = None) -> Program:
    """Check coverage on ``profile`` and emit the step program for a lowered (and passed) graph: for a static
    ``T = t`` (kernels specialized, T from params), or with ``dynamic_t`` for any T ≤ ``t_max`` read from StepState
    at run time (kernels compiled at ``t_max``). ``pack`` is the pack (or the packs: the target's, then a drafter's)
    the graph's weights and constants come from. ``tail="advance"`` appends the step's advance on ``token`` (the
    sampled tokens); ``None`` leaves the closing op to the graph (a program with the DSpark round emits its own
    ``accept_scan``). ``barriers``: ``"minimal"`` keeps an ICB barrier only where a dependency needs one (the
    barrier pass), ``"all"`` after every op (v0; the A/B baseline). The profile's ``sibling_order`` decides whether a
    mixer's gate GEMV is encoded after its core (``alu_first``, ``either``) or before it (``bus_first``)."""
    layout = layout or StepStateLayout()
    packs = [pack] if isinstance(pack, PackFile) else list(pack)
    if dynamic_t:
        t = layout.t_max
    if t is None or t < 1 or t > layout.t_max:
        raise ValueError(f"emit_program: T = {t} must be within 1..t_max = {layout.t_max}")
    check_coverage(g, profile)
    program = Program(kernels={}, buffers={}, ops=[], ring_capacity=ring_capacity, layout=layout)
    ctx = _Ctx(program, packs, layout, t, 12 * profile.gpu_cores * profile.threadgroups_per_core, tg, values=g.values, dynamic_t=dynamic_t,
               speculative=speculative, attention=attention or profile.attention, tuner=tuner, eos=eos, ring_capacity=ring_capacity)
    _pack_windows(ctx)
    hoisted = {op.attrs["stat_value"]: ctx.slab_info(op.inputs[1].name).n_blocks for op in g.ops if op.kind == "gemv" and op.attrs.get("stat_value")}
    for v in g.values.values():
        if v.is_view:
            continue                                                  # rows of its base's buffer
        if v.is_state:
            program.buffers[v.name] = BufferSpec(_value_bytes(v, t), None, "state")
        elif v.is_input:
            fld = "pending_tokens" if v.name == "tokens" else v.name
            if fld not in layout.offsets:                             # not StepState-backed: a host-written arena buffer
                program.buffers[v.name] = BufferSpec(max(_value_bytes(v, t), 16), None, "arena")
        elif not v.is_source:
            nbytes = _value_bytes(v, t)
            if v.name in hoisted:
                nbytes = t * hoisted[v.name] * 4              # a hoisted statistic holds n_blocks partials per token
            program.buffers[v.name] = BufferSpec(max(nbytes, 16), None, "arena")
        elif v.is_weight or v.is_const:
            if v.name not in ctx.windows:
                raise KeyError(f"emit_program: the pack has no entry for {v.name!r}")
    program.buffers[program.step_state] = BufferSpec(layout.size, layout.pack({"t_this_step": t}), "step_state")
    program.buffers[program.ring] = BufferSpec(ring_capacity * 8, None, "ring")
    order = list(g.ops)
    if profile.sibling_order == "bus_first":
        order = _bus_first(order)
    for op in order:
        HANDLERS[op.kind](ctx, op)
    if tail == "advance":
        if token is None:
            raise ValueError("emit_program: the advance needs the sampled token value")
        adv = ctx.kernel("advance", kernels.advance_source(layout.to_msl()), "advance", {})
        prm = ctx.params("advance", kernels.advance_params(t, ring_capacity, eos))
        ctx.add(adv, [(0, *ctx.buf(token)), (1, program.step_state, 0), (2, program.ring, 0), (3, prm, 0)], (1, 1, 1), (32, 1, 1), "advance",
                writes=[1, 2])
    elif tail is not None:
        raise ValueError(f"emit_program: unknown tail {tail!r}")
    place_barriers(program, barriers)
    return program


def _bus_first(ops: List[Op]) -> List[Op]:
    """Move each sibling gate GEMV in front of the mixer core it was emitted after (profiles where the bus-bound
    dispatch must be encoded first for the pair to overlap)."""
    out = list(ops)
    for i, op in enumerate(out):
        if op.attrs.get("sibling") and i > 0 and out[i - 1].kind in ("gqa_decode", "gdn_mixer"):
            out[i - 1], out[i] = out[i], out[i - 1]
    return out


def verify_costs(profile: Profile, pack: PackFile, gamma: int, t_max: int) -> Optional[List[float]]:
    """``cost[l]`` = the profile's relative cost of a (1 + l)-token pass over the pack's dominant weight format,
    l = 0 … min(γ, t_max − 1); None when the profile has no table for that format (the M3 Pro's, until p13 runs)."""
    by_fmt: Dict[str, int] = {}
    for s in pack.manifest["slabs"]:
        by_fmt[s["format"]] = by_fmt.get(s["format"], 0) + int(s["nbytes"])
    if not by_fmt:
        return None
    fmt = max(by_fmt, key=lambda f: by_fmt[f])
    key = COST_FORMAT.get(fmt, fmt)
    try:
        return [profile.cost(key, 1 + l) for l in range(min(gamma, t_max - 1) + 1)]
    except (KeyError, ValueError):
        return None


def lower_round(g: Graph, model: Model, drafter: Any, token: Value, profile: Profile, *, cost: Optional[Sequence[float]] = None,
                threshold: Optional[float] = None, fixed: Optional[int] = None) -> Value:
    """Append the speculative round (design §5.8) to a lowered target step: the accept scan on the sampled tokens,
    the state commit passes of the mixers that declare one (``commit_kind``), the drafter's draft pass on the
    target's tapped residual streams and the verify-length select. Returns the select's value."""
    from ..spec import DraftContext

    acc = g.value("accepted", (1,), DType.U32)
    g.op("accept_scan", [token], [acc], domain=BlockDomain("span", 1), klass=OpClass.SERIAL)
    for op in list(g.ops):
        ck = op.attrs.get("commit_kind")
        if ck:
            attrs = {k: v for k, v in op.attrs.items() if k != "commit_kind"}
            g.op(ck, list(op.inputs), [g.value(f"{op.outputs[0].name}.commit", (1,), DType.U32)], domain=op.domain, klass=op.klass, **attrs)
    taps = []
    for i in drafter.tap_layers():
        if i not in model.tap_values:
            raise ValueError(f"lower_round: the drafter taps layer {i}, which the model does not expose ({sorted(model.tap_values)})")
        taps.append(model.tap_values[i])
    block = drafter.lower_draft(g, DraftContext(taps, None))
    return drafter.lower_select(g, block, profile, cost=cost, threshold=threshold, fixed=fixed)


def compile_program(model: Model, pack: PackFile, profile: Profile, *, t: Optional[int] = None, eos: int = -1, ring_capacity: int = 4096,
                    layout: Optional[StepStateLayout] = None, tg: int = 384, passes=DEFAULT_PASSES, dynamic_t: bool = False,
                    tuner: Any = None, drafter: Any = None, drafter_pack: Optional[PackFile] = None, verify: str = "cost",
                    verify_threshold: Optional[float] = None, verify_length: Optional[int] = None, barriers: str = "minimal",
                    attention: Optional[str] = None) -> Program:
    """Lower ``model``, run the ``passes`` and emit its step program (see :func:`emit_program`). With a ``drafter``
    (and its pack) the dynamic-T program carries the speculative round instead of the advance: ``verify`` = ``"cost"``
    (the cost-aware verify-length rule when the profile has a cost table for the pack's dominant format, otherwise the
    threshold rule), ``"threshold"`` (``verify_threshold``; None = 0.5, ≤ 0 = verify the whole block) or ``"fixed"``
    (``verify_length`` drafts every step — the measurement's baseline)."""
    layout = layout or StepStateLayout()
    g = Graph("step")
    token = model.lower(g)
    if drafter is None:
        g.check()
        for p in passes:
            p(g)
        return emit_program(g, pack=pack, profile=profile, t=t, dynamic_t=dynamic_t, layout=layout, eos=eos, ring_capacity=ring_capacity, tg=tg,
                            tuner=tuner, tail="advance", token=token, barriers=barriers, attention=attention)
    if not dynamic_t:
        raise ValueError("compile_program: the speculative round needs the dynamic-T program (dynamic_t=True)")
    if drafter_pack is None:
        raise ValueError("compile_program: a drafter needs its pack")
    if drafter.gamma > layout.gamma_max or drafter.gamma + 1 > layout.t_max:
        raise ValueError(f"compile_program: a block of {drafter.gamma} needs gamma_max >= {drafter.gamma} and t_max >= {drafter.gamma + 1} "
                         f"(layout: {layout.gamma_max}, {layout.t_max})")
    if verify not in ("cost", "threshold", "fixed"):
        raise ValueError(f"compile_program: verify must be 'cost', 'threshold' or 'fixed', got {verify!r}")
    fixed = None
    if verify == "fixed":
        if verify_length is None or not 0 <= verify_length <= drafter.gamma:
            raise ValueError(f"compile_program: verify='fixed' needs verify_length in 0..{drafter.gamma}")
        fixed = int(verify_length)
    cost = verify_costs(profile, pack, drafter.gamma, layout.t_max) if verify == "cost" else None
    if cost is None and fixed is None and verify_threshold is None:
        verify_threshold = FALLBACK_THRESHOLD           # no cost table (or the threshold rule asked for without a threshold)
    lower_round(g, model, drafter, token, profile, cost=cost, threshold=verify_threshold, fixed=fixed)
    g.check()
    for p in passes:
        p(g)
    return emit_program(g, pack=[pack, drafter_pack], profile=profile, dynamic_t=True, layout=layout, eos=eos, ring_capacity=ring_capacity,
                        tg=tg, tuner=tuner, tail=None, speculative=True, barriers=barriers, attention=attention)
