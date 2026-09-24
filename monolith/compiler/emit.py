"""``compile_program``: the lowered graph → a runtime :class:`Program` (plan M4, v0).

v0 keeps every decision simple and correct: one device buffer per graph value (no aliasing), a barrier after every
op, the norm as ``rmsnorm_stat`` → ``norm_apply`` → plain GEMV (the fuse pass that hoists the statistic into the
producer's epilogue comes next), kernels specialized to the program's static ``T`` (a prefill program at ``T = P``
and a decode program at ``T = 1`` share their buffers by name), weights and constants mapped straight from the
pack file in page-aligned windows (no copy). The op handlers below are the only place an op kind meets a kernel
source; a new op kind adds a handler and a kernel, nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .. import kernels
from ..core.ir import Graph, Op, Value
from ..core.profile import Profile
from ..core.shapes import T, bind, numel
from ..core.step_state import StepStateLayout
from ..nn.module import Model
from ..packs.packer import PackFile
from ..runtime.program import BufferSpec, KernelSpec, OpSpec, Program
from .coverage import check_coverage
from .passes import DEFAULT_PASSES

WINDOW_BYTES = 2 << 30          # pack windows: ICB bind offsets are 32-bit (design §5.1)


@dataclass
class _Ctx:
    program: Program
    pack: PackFile
    layout: StepStateLayout
    t: int
    n_sg: int
    tg: int
    windows: Dict[str, Tuple[str, int]] = field(default_factory=dict)      # pack entry name -> (buffer, offset)
    row_scales: Dict[str, Tuple[str, int]] = field(default_factory=dict)
    stat_parts: Dict[str, int] = field(default_factory=dict)              # statistic value -> partial sums per token
    dynamic_t: bool = False                                               # T from StepState (prefill chunks); else static
    tuner: Any = None                                                     # compiler.autotune.Autotuner or None
    counter: int = 0

    # ---- helpers -----------------------------------------------------------------------------------------------
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

    def scratch(self, name: str, nbytes: int) -> str:
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

    def add(self, kernel: str, bindings: List[Tuple[int, str, int]], grid, tg, name: str, **meta: Any) -> None:
        if self.dynamic_t and name != "advance" and not any(b[0] == 15 for b in bindings):
            bindings = list(bindings) + [(15, self.program.step_state, 0)]
        self.program.ops.append(OpSpec(kernel, bindings, tuple(grid), tuple(tg), True, [], name, dict(meta)))

    def shape(self, v: Value) -> Tuple[int, ...]:
        return bind(v.shape, {T: self.t})


def _value_bytes(v: Value, t: int) -> int:
    return numel(v.shape, {T: t}) * v.dtype.itemsize


def _pack_windows(ctx: _Ctx, path: str) -> None:
    """Group the pack's slabs, row-scale tables and aux entries into ≤ 2 GiB file-backed windows."""
    entries: List[Tuple[int, int, str, str]] = []                       # (offset, nbytes, kind, name)
    for s in ctx.pack.manifest["slabs"]:
        entries.append((s["offset"], s["nbytes"], "slab", s["name"]))
        entries.append((s["row_scales_offset"], 4 * s["n"], "rs", s["name"]))
    for a in ctx.pack.manifest["aux"]:
        entries.append((a["offset"], a["nbytes"], "aux", a["name"]))
    entries.sort()
    start, end, members = None, 0, []
    windows: List[Tuple[int, int, list]] = []
    for off, nb, kind, name in entries:
        if start is None or off + nb - start > WINDOW_BYTES:
            if start is not None:
                windows.append((start, end, members))
            start, end, members = off, off + nb, []
        end = max(end, off + nb)
        members.append((off, kind, name))
    if start is not None:
        windows.append((start, end, members))
    for i, (start, end, members) in enumerate(windows):
        bname = f"pack.{i}"
        ctx.program.buffers[bname] = BufferSpec(end - start, None, "weights", str(path), start)
        for off, kind, name in members:
            (ctx.row_scales if kind == "rs" else ctx.windows)[name] = (bname, off - start)


# ---- op handlers ---------------------------------------------------------------------------------------------------

def _embed(ctx: _Ctx, op: Op) -> None:
    tokens, table = op.inputs
    h = op.outputs[0]
    info = ctx.pack.slab_info(table.name)
    k = ctx.kernel("embed", kernels.embed_source(), "embed", kernels.embed_macros(info))
    prm = ctx.params("embed", kernels.embed_params(info.k, ctx.t, info.n))
    tok_binding = (ctx.program.step_state, ctx.layout.offset("pending_tokens")) if tokens.is_input else (tokens.name, 0)
    ctx.add(k, [(0, tok_binding[0], tok_binding[1]), (1, *ctx.windows[table.name]), (2, h.name, 0), (3, prm, 0)], (ctx.t, 1, 1), (32, 1, 1), op.kind)


def _rmsnorm_stat(ctx: _Ctx, op: Op) -> None:
    h, = op.inputs
    stat = op.outputs[0]
    if op.attrs.get("hoisted"):
        return                                        # the producer GEMV writes the partials (fuse_norm_stat)
    k = ctx.kernel("rmsnorm_stat", kernels.rmsnorm_stat_source(), "rmsnorm_stat", {})
    prm = ctx.params("stat", kernels.stat_params(ctx.shape(h)[1], ctx.t))
    ctx.add(k, [(0, h.name, 0), (1, stat.name, 0), (2, prm, 0)], (ctx.t, 1, 1), (32, 1, 1), op.kind)


def _norm_apply(ctx: _Ctx, h: Value, stat: Value, nw: Value, eps: float, out_name: str) -> str:
    k = ctx.kernel("norm_apply", kernels.norm_apply_source(), "norm_apply", {})
    kdim = ctx.shape(h)[1]
    xn = ctx.scratch(out_name, ctx.t * kdim * 2)
    prm = ctx.params("norm_apply", kernels.norm_apply_params(kdim, ctx.t, ctx.stat_parts.get(stat.name, 1), eps))
    ctx.add(k, [(0, h.name, 0), (1, stat.name, 0), (2, *ctx.windows[nw.name]), (3, xn, 0), (4, prm, 0)], (ctx.t, 1, 1), (32, 1, 1), "norm_apply")
    return xn


def _gemv(ctx: _Ctx, op: Op) -> None:
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
    info = ctx.pack.slab_info(w.name)
    choice = ctx.tuner.tune_gemv(info, ctx.t, op.attrs.get("epilogue"), stat is not None) if ctx.tuner is not None else None
    fuse_norm = bool(choice and choice.fuse_norm)
    rg = int(choice.macros["RG"]) if choice else None
    x_binding = (x.name, 0)
    if stat is not None and not fuse_norm:
        x_binding = (_norm_apply(ctx, x, stat, nw, float(op.attrs.get("eps", 1e-6)), f"{y.name}.xn"), 0)
    stat_out = op.attrs.get("stat_value")
    macros = kernels.gemv_macros(info, t=ctx.t, rg=rg, epilogue=op.attrs.get("epilogue"), out_bf16=True, stat_out=stat_out is not None,
                                 norm=fuse_norm)
    k = ctx.kernel(f"gemv_T|{info.format}", kernels.gemv_source(info.format), "gemv_T", macros)
    n_sg, grid, tg = ctx.geometry(choice.grid_mode if choice else "crew", info.n_blocks)
    prm = ctx.params("gemv", kernels.gemv_params(info.n, info.n_blocks, n_sg, ctx.t, eps=float(op.attrs.get("eps", 1e-6)),
                                                 stat_parts=ctx.stat_parts.get(stat.name, 1) if stat is not None else 1))
    bindings = [(0, *ctx.windows[w.name]), (1, *ctx.row_scales[w.name]), (2, *x_binding), (3, y.name, 0), (4, prm, 0)]
    if fuse_norm:
        bindings += [(5, stat.name, 0), (6, *ctx.windows[nw.name])]
    if residual is not None:
        bindings.append((7, residual.name, 0))
    if stat_out is not None:
        bindings.append((8, stat_out, 0))
        ctx.stat_parts[stat_out] = info.n_blocks
    ctx.add(k, bindings, grid, tg, f"{op.kind}:{w.name}", kind=op.kind, bytes=int(info.nbytes), format=info.format, n=info.n, k=info.k,
            rg=int(macros["RG"]), geometry=choice.grid_mode if choice else "crew", fused_norm=fuse_norm)


def _gqa(ctx: _Ctx, op: Op) -> None:
    proj, kc, vc, cos, sin, qn, kn = op.inputs
    out = op.outputs[0]
    a = op.attrs
    d, heads, kv = a["head_dim"], a["heads"], a["kv_heads"]
    segs = {name: (off, n) for name, off, n in a["segments"]}
    ctx_max = ctx.shape(kc)[0]
    chunk = 64
    macros = dict(kernels.gqa_macros(d, chunk=chunk), STEP_STATE="1")     # position always comes from StepState
    src = kernels.PRELUDE + ctx.layout.to_msl() + "\n" + kernels.template("gqa_decode.metal")
    kd, km = ctx.kernel("gqa", src, "gqa_decode", macros), ctx.kernel("gqa", src, "gqa_merge", macros)
    rep = heads // kv
    n_chunks_max, rows_max = -(-ctx_max // chunk), rep * ctx.t
    po, pm = kernels.gqa_workspace(kv, n_chunks_max, rows_max, d)
    part_o, part_md = ctx.scratch("gqa.part_o", po), ctx.scratch("gqa.part_md", pm)
    prm = ctx.params("gqa", kernels.gqa_params(
        heads=heads, kv_heads=kv, t_active=ctx.t, position=0, n_sg=ctx.n_sg, q_off=segs["q"][0],
        gate_off=segs["gate"][0] if "gate" in segs else 0, k_off=segs["k"][0], v_off=segs["v"][0], in_stride=ctx.shape(proj)[1],
        out_stride=heads * d, ctx_max=ctx_max, eps=float(a["eps"]), scaling=float(a["scaling"]), has_gate=bool(a.get("gate")),
        n_chunks_max=n_chunks_max, rows_max=rows_max))
    st = ctx.program.step_state
    grid, tg = ctx.crew_grid()
    ctx.add(kd, [(0, proj.name, 0), (1, kc.name, 0), (2, vc.name, 0), (3, *ctx.windows[cos.name]), (4, *ctx.windows[sin.name]),
                 (5, *ctx.windows[qn.name]), (6, *ctx.windows[kn.name]), (7, part_o, 0), (8, part_md, 0), (9, prm, 0), (15, st, 0)],
            grid, tg, op.kind)
    ctx.add(km, [(0, part_o, 0), (1, part_md, 0), (2, proj.name, 0), (3, out.name, 0), (4, prm, 0), (15, st, 0)],
            (ctx.t * heads, 1, 1), (32, 1, 1), "gqa_merge")


def _gdn(ctx: _Ctx, op: Op) -> None:
    a = op.attrs
    n_proj = len(a["proj_segments"]) and (1 + max(idx for idx, _, _ in a["proj_segments"].values()))
    projs = op.inputs[:n_proj]
    cs, rs, conv_w, a_log, dt_bias, norm_w = op.inputs[n_proj:]
    out = op.outputs[0]
    hv, hk, dk, dv, cw = a["v_heads"], a["k_heads"], a["dk"], a["dv"], a["conv_width"]
    ps = a["proj_segments"]                                           # local -> (value index, column offset, columns)
    kd = hk * dk
    ab_separate = ps["in_proj_a"][0] != ps["in_proj_qkv"][0]
    gch = ctx.tuner.tune_gdn(hv, hk, dk, dv, cw, ctx.t) if ctx.tuner is not None else None
    macros = kernels.gdn_macros(dk, dv, conv_width=cw, t=ctx.t, slice_cols=int(str(gch.macros["SL"]).rstrip("u")) if gch else 8,
                                slices_per_block=int(str(gch.macros["SPB"]).rstrip("u")) if gch else 4)
    src = kernels.gdn_source()
    kmix, knorm = ctx.kernel("gdn", src, "gdn_mixer", macros), ctx.kernel("gdn", src, "gdn_norm", macros)
    o_part = ctx.scratch("gdn.o_part", kernels.gdn_workspace(ctx.t, hv, dv))
    main, abv = projs[ps["in_proj_qkv"][0]], projs[ps["in_proj_a"][0]]
    prm = ctx.params("gdn", kernels.gdn_params(
        hv=hv, hk=hk, t_active=ctx.t, q_off=ps["in_proj_qkv"][1], k_off=ps["in_proj_qkv"][1] + kd, v_off=ps["in_proj_qkv"][1] + 2 * kd,
        z_off=ps["in_proj_z"][1], a_off=ps["in_proj_a"][1], b_off=ps["in_proj_b"][1], in_stride=ctx.shape(main)[1],
        ab_stride=ctx.shape(abv)[1], ab_separate=ab_separate, out_stride=hv * dv, n_sg=ctx.n_sg, key_dim=kd, eps=float(a["eps"])))
    grid, tg = ctx.crew_grid()
    ctx.add(kmix, [(0, main.name, 0), (1, abv.name, 0), (2, cs.name, 0), (3, rs.name, 0), (4, *ctx.windows[conv_w.name]),
                   (5, *ctx.windows[a_log.name]), (6, *ctx.windows[dt_bias.name]), (7, o_part, 0), (9, prm, 0)], grid, tg, op.kind)
    ctx.add(knorm, [(0, o_part, 0), (1, main.name, 0), (2, *ctx.windows[norm_w.name]), (3, out.name, 0), (4, prm, 0)],
            (ctx.t * hv, 1, 1), (32, 1, 1), "gdn_norm")


def _argmax(ctx: _Ctx, op: Op) -> None:
    logits, = op.inputs
    token = op.outputs[0]
    vocab = ctx.shape(logits)[1]
    src = kernels.argmax_source()
    kp, kf = ctx.kernel("argmax", src, "argmax_partial", {}), ctx.kernel("argmax", src, "argmax_final", {})
    pv, pi = ctx.scratch("argmax.val", ctx.t * ctx.n_sg * 4), ctx.scratch("argmax.idx", ctx.t * ctx.n_sg * 4)
    prm = ctx.params("argmax", kernels.argmax_params(vocab, ctx.t, ctx.n_sg))
    grid, tg = ctx.crew_grid()
    ctx.add(kp, [(0, logits.name, 0), (1, pv, 0), (2, pi, 0), (3, prm, 0)], grid, tg, op.kind)
    ctx.add(kf, [(0, pv, 0), (1, pi, 0), (2, token.name, 0), (3, prm, 0)], (ctx.t, 1, 1), (32, 1, 1), "argmax_final")


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
    ctx.add(kh, [(0, logits.name, 0), (1, hist, 0), (3, prm, 0), (15, st, 0)], grid, tg, op.kind)
    ctx.add(ks, [(1, hist, 0), (2, tau, 0), (3, prm, 0), (15, st, 0)], (ctx.t, 1, 1), (32, 1, 1), "sample_select")
    ctx.add(kg, [(0, logits.name, 0), (2, tau, 0), (3, prm, 0), (4, pv, 0), (5, pi, 0), (15, st, 0)], grid, tg, "sample_gumbel")
    ctx.add(kf, [(0, pv, 0), (1, pi, 0), (2, token.name, 0), (3, prm, 0), (15, st, 0)], (ctx.t, 1, 1), (32, 1, 1), "argmax_final")


HANDLERS = {"embed": _embed, "rmsnorm_stat": _rmsnorm_stat, "gemv": _gemv, "lm_head": _gemv, "gqa_decode": _gqa,
            "gdn_mixer": _gdn, "argmax": _argmax, "sample": _sample}


def compile_program(model: Model, pack: PackFile, profile: Profile, *, t: Optional[int] = None, eos: int = -1, ring_capacity: int = 4096,
                    layout: Optional[StepStateLayout] = None, tg: int = 384, passes=DEFAULT_PASSES, dynamic_t: bool = False,
                    tuner: Any = None) -> Program:
    """Lower ``model``, run the ``passes``, check coverage on ``profile`` and emit the step program: for a static
    ``T = t`` (kernels specialized, T from params), or with ``dynamic_t`` for any T ≤ ``t_max`` read from
    ``StepState.t_this_step`` at run time (the prefill-chunk program; kernels compiled at ``t_max``)."""
    layout = layout or StepStateLayout()
    if dynamic_t:
        t = layout.t_max
    if t is None or t < 1 or t > layout.t_max:
        raise ValueError(f"compile_program: T = {t} must be within 1..t_max = {layout.t_max}")
    g = Graph("step")
    token = model.lower(g)
    g.check()
    for p in passes:
        p(g)
    check_coverage(g, profile)
    program = Program(kernels={}, buffers={}, ops=[], ring_capacity=ring_capacity, layout=layout)
    ctx = _Ctx(program, pack, layout, t, 12 * profile.gpu_cores * profile.threadgroups_per_core, tg, dynamic_t=dynamic_t, tuner=tuner)
    _pack_windows(ctx, pack.dir / pack.manifest["pack"])
    hoisted = {op.attrs["stat_value"]: pack.slab_info(op.inputs[1].name).n_blocks for op in g.ops if op.kind == "gemv" and op.attrs.get("stat_value")}
    for v in g.values.values():
        if v.is_state:
            program.buffers[v.name] = BufferSpec(_value_bytes(v, t), None, "state")
        elif not v.is_source:
            nbytes = _value_bytes(v, t)
            if v.name in hoisted:
                nbytes = t * hoisted[v.name] * 4              # a hoisted statistic holds n_blocks partials per token
            program.buffers[v.name] = BufferSpec(max(nbytes, 16), None, "arena")
        elif v.is_weight or v.is_const:
            if v.name not in ctx.windows:
                raise KeyError(f"compile_program: the pack has no entry for {v.name!r}")
    program.buffers[program.step_state] = BufferSpec(layout.size, layout.pack({"t_this_step": t}), "step_state")
    program.buffers[program.ring] = BufferSpec(ring_capacity * 8, None, "ring")
    for op in g.ops:
        HANDLERS[op.kind](ctx, op)
    adv = ctx.kernel("advance", kernels.advance_source(layout.to_msl()), "advance", {})
    prm = ctx.params("advance", kernels.advance_params(t, ring_capacity, eos))
    ctx.add(adv, [(0, token.name, 0), (1, program.step_state, 0), (2, program.ring, 0), (3, prm, 0)], (1, 1, 1), (32, 1, 1), "advance")
    return program
