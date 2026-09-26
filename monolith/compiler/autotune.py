"""Per-op autotune (plan M5, #34; design §5.7 "autotune hooks"): for every distinct GEMV shape and GDN mixer
geometry of a program, time a small set of kernel variants on the device with synthetic data of the same shape and
format, keep the fastest, and cache the choices per (chip, pack layout) so a program compiles with them.

GEMV variants: rows per activation-reuse group ``RG``, the geometry (the crew: 12 SIMD-groups per core with 1 or 2
threadgroups per core, or one block per SIMD-group in 64-thread threadgroups) and, for a norm-fed GEMV, whether
the RMSNorm scaling is fused into the load (``NORM=1``, one dispatch fewer) or applied by ``norm_apply``. GDN
variants: state-slice columns and slices per block. Timings are min-of-N after a warm-up; a variant must beat the
default by more than the noise margin to replace it.
"""

from __future__ import annotations

import itertools
import json
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .. import kernels
from ..bench import pack_spec, random_spec
from ..formats import FORMATS, PackLayout
from ..formats.blm import PackInfo
from ..formats.fp import f32_to_bf16

NOISE_MARGIN = 0.03           # a variant replaces the default only if faster by more than this fraction


def gemv_key(info: PackInfo, t: int, epilogue: Optional[str], norm_fed: bool) -> str:
    return f"gemv|{info.format}|{info.n}x{info.k}|R{info.rows}|{info.lane_order}|T{t}|{epilogue or 'plain'}|{'norm' if norm_fed else 'raw'}"


def gdn_key(hv: int, hk: int, dk: int, dv: int, conv_width: int, t: int) -> str:
    return f"gdn|{hk}x{hv}|{dk}x{dv}|cw{conv_width}|T{t}"


@dataclass
class Choice:
    macros: Dict[str, str]
    grid_mode: str                 # "crew" | "crew2" | "block"
    fuse_norm: bool = False
    ms: float = 0.0
    default_ms: float = 0.0


class Autotuner:
    def __init__(self, device, gpu_cores: int, cache_path: Optional[str] = None, *, reps: int = 5, warmup_ms: float = 30.0) -> None:
        from ..runtime import _native as nt

        self.nt, self.dev, self.cores = nt, device, gpu_cores
        self.reps, self.warmup_ms = reps, warmup_ms
        self.cache_path = Path(cache_path) if cache_path else None
        self.choices: Dict[str, Dict[str, Any]] = {}
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path) as f:
                self.choices = json.load(f).get("choices", {})
        self.queue = nt.Queue(device)

    # ---- timing ------------------------------------------------------------------------------------------------
    def _time(self, dispatches) -> float:
        warm = 0.0
        while warm < self.warmup_ms:
            r = self.queue.run(dispatches)
            if r.error:
                raise RuntimeError(r.error)
            warm += max(r.gpu_ms, 0.01)
        best = None
        for _ in range(self.reps):
            r = self.queue.run(dispatches)
            best = r.gpu_ms if best is None else min(best, r.gpu_ms)
        return best

    def grid(self, mode: str, n_blocks: int) -> Tuple[int, int, int]:
        """(n_sg, grid threadgroups, threadgroup size) for a geometry mode."""
        if mode == "block":
            n_sg = n_blocks
            return n_sg, -(-(n_sg * 32) // 64), 64
        n_sg = 12 * self.cores * (2 if mode == "crew2" else 1)
        return n_sg, -(-(n_sg * 32) // 384), 384

    # ---- GEMV ----------------------------------------------------------------------------------------------------
    def tune_gemv(self, info: PackInfo, t: int, epilogue: Optional[str], norm_fed: bool, *, force: bool = False) -> Choice:
        key = gemv_key(info, t, epilogue, norm_fed)
        if key in self.choices and not force:
            c = self.choices[key]
            return Choice(dict(c["macros"]), c["grid_mode"], c.get("fuse_norm", False), c.get("ms", 0.0), c.get("default_ms", 0.0))
        nt = self.nt
        rng = np.random.default_rng(0)
        spec = random_spec(info.format, info.n, info.k, rng)
        data, pinfo, row_scales = pack_spec(spec, PackLayout(rows=info.rows, lane_order=info.lane_order))
        copies = max(1, int((256 << 20) // max(len(data), 1)))           # ≥ 256 MB streamed per timing
        wbuf = nt.Buffer(self.dev, len(data) * copies)
        for c in range(copies):
            wbuf.write(data, c * len(data))
        xb = f32_to_bf16(rng.uniform(-1, 1, size=(t, info.k)).astype(np.float32))
        xbuf, rsbuf = nt.Buffer(self.dev, xb.tobytes()), nt.Buffer(self.dev, row_scales.tobytes())
        ybuf = nt.Buffer(self.dev, t * info.n * 2)
        stat = nt.Buffer(self.dev, (t * 4 * 64))
        stat.write(np.full(t * 64, float(info.k) / 64, np.float32).tobytes(), 0)
        nw = nt.Buffer(self.dev, np.ones(info.k, np.float32).tobytes())
        res = nt.Buffer(self.dev, t * info.n * 2)
        xn = nt.Buffer(self.dev, t * info.k * 2)
        variants: List[Tuple[str, Dict[str, Any]]] = []
        rgs = [r for r in (2, 4, 8) if r <= info.rows and info.rows % r == 0]
        for rg, mode in itertools.product(rgs, ("crew", "crew2", "block")):
            variants.append(("apply", {"rg": rg, "mode": mode, "fuse": False}))
            if norm_fed:
                variants.append(("fused", {"rg": rg, "mode": mode, "fuse": True}))
        results = []
        default_ms = None
        for _, v in variants:
            try:
                macros = kernels.gemv_macros(pinfo, t=t, rg=v["rg"], epilogue=epilogue, norm=v["fuse"], out_bf16=True)
            except ValueError:
                continue
            n_sg, grid, tg = self.grid(v["mode"], pinfo.n_blocks)
            pso = nt.Pipeline(nt.Library(self.dev, kernels.gemv_source(info.format), macros), "gemv_T")
            prm = kernels.gemv_params(info.n, pinfo.n_blocks, n_sg, t, eps=1e-6, stat_parts=64)
            ds = []
            if norm_fed and not v["fuse"]:
                apso = nt.Pipeline(nt.Library(self.dev, kernels.norm_apply_source(), {}), "norm_apply")
                aprm = kernels.norm_apply_params(info.k, t, 64, 1e-6)
            for c in range(copies):
                if norm_fed and not v["fuse"]:
                    ds.append(nt.Dispatch().pipeline(apso).buffer(0, xbuf).buffer(1, stat).buffer(2, nw).buffer(3, xn).bytes(4, aprm).grid(t).threadgroup(32))
                d = (nt.Dispatch().pipeline(pso).buffer(0, wbuf, c * len(data)).buffer(1, rsbuf).buffer(2, xn if (norm_fed and not v["fuse"]) else xbuf)
                     .buffer(3, ybuf).bytes(4, prm).grid(grid).threadgroup(tg))
                if v["fuse"]:
                    d.buffer(5, stat).buffer(6, nw)
                if epilogue == "residual":
                    d.buffer(7, res)
                ds.append(d)
            ms = self._time(ds) / copies
            is_default = v["rg"] == int(kernels.gemv_macros(pinfo, t=t, epilogue=epilogue, out_bf16=True)["RG"]) and v["mode"] == "crew" and not v["fuse"]
            if is_default:
                default_ms = ms
            results.append((ms, v, macros))
        results.sort(key=lambda r: r[0])
        best_ms, bv, bmacros = results[0]
        if default_ms is not None and best_ms > default_ms * (1 - NOISE_MARGIN):
            bv = {"rg": int(kernels.gemv_macros(pinfo, t=t, epilogue=epilogue, out_bf16=True)["RG"]), "mode": "crew", "fuse": False}
            bmacros = kernels.gemv_macros(pinfo, t=t, epilogue=epilogue, out_bf16=True)
            best_ms = default_ms
        choice = Choice({"RG": bmacros["RG"]}, bv["mode"], bv["fuse"], best_ms, default_ms or best_ms)
        self.choices[key] = {"macros": choice.macros, "grid_mode": choice.grid_mode, "fuse_norm": choice.fuse_norm, "ms": choice.ms,
                             "default_ms": choice.default_ms, "variants": [(round(ms, 4), v) for ms, v, _ in results]}
        return choice

    # ---- the tensor-ops tile (#51) -------------------------------------------------------------------------------
    def tune_gemm(self, info: PackInfo, tm: int, epilogue: Optional[str], *, force: bool = False) -> Choice:
        """The tile's geometry: one or two threadgroups per core (the sweep in decode-kernels.md §6 found either,
        by format) or the K-split (one tile per threadgroup of 2 or 4 SIMD-groups, for the shapes whose tiles cannot
        occupy the crew), timed on synthetic data like the GEMV variants."""
        key = f"gemm2|{info.format}|{info.n}x{info.k}|R{info.rows}|{info.lane_order}|TM{tm}|{epilogue or 'plain'}"
        if key in self.choices and not force:
            c = self.choices[key]
            return Choice(dict(c["macros"]), c["grid_mode"], False, c.get("ms", 0.0), c.get("default_ms", 0.0))
        nt = self.nt
        rng = np.random.default_rng(0)
        spec = random_spec(info.format, info.n, info.k, rng)
        data, pinfo, row_scales = pack_spec(spec, PackLayout(rows=info.rows, lane_order=info.lane_order))
        copies = max(1, int((256 << 20) // max(len(data), 1)))
        wbuf = nt.Buffer(self.dev, len(data) * copies)
        for c in range(copies):
            wbuf.write(data, c * len(data))
        macros = kernels.gemm_macros(pinfo, tm=tm, out_bf16=True, epilogue=epilogue)
        tn, tk = int(macros["TN"].rstrip("u")), int(macros["TK"].rstrip("u"))
        lib = nt.Library(self.dev, kernels.gemm_source(info.format), dict(macros, **kernels.x_permute_macros(False)), language_version=kernels.MSL_TENSOR_OPS)
        pso, ppso = nt.Pipeline(lib, "gemm_tile"), nt.Pipeline(lib, "x_permute")
        psos = {"crew": pso, "crew2": pso}
        for s in (2, 4):
            try:
                km = kernels.gemm_macros(pinfo, tm=tm, out_bf16=True, epilogue=epilogue, ksplit=s)
            except ValueError:
                continue                                               # the slab's K tiles do not split that way
            psos[f"ksplit{s}"] = nt.Pipeline(nt.Library(self.dev, kernels.gemm_source(info.format), km, language_version=kernels.MSL_TENSOR_OPS), "gemm_tile")
        xb = f32_to_bf16(rng.uniform(-1, 1, size=(tm, info.k)).astype(np.float32))
        xbuf, rsbuf = nt.Buffer(self.dev, xb.tobytes()), nt.Buffer(self.dev, row_scales.tobytes())
        xp = nt.Buffer(self.dev, tm * info.k * 2)
        n_out = info.n // 2 if epilogue == "silu_mul" else info.n
        ybuf = nt.Buffer(self.dev, tm * n_out * 2)
        res = nt.Buffer(self.dev, tm * info.n * 2)
        wpw = int(FORMATS.get(info.format).weights_per_word)
        n_tiles = -(-info.n // tn)
        results = []
        for mode, mpso in psos.items():
            n_sg, n_tg, tg = kernels.gemm_geometry(mode, n_tiles, self.cores, min(384, mpso.max_threads_per_threadgroup))
            grid = (n_tg, 1, 1)
            prm = kernels.gemm_params(info.n, n_tiles, n_sg, tm, n_blocks=pinfo.n_blocks)
            pprm = kernels.x_permute_params(info.k, tm, tm, wpw, tk)
            ds = [nt.Dispatch().pipeline(ppso).buffer(0, xbuf).buffer(3, xp).bytes(4, pprm).grid(tm * kernels.GEMM_PERM_SG).threadgroup(32).barrier()]
            for c in range(copies):
                d = (nt.Dispatch().pipeline(mpso).buffer(0, wbuf, c * len(data)).buffer(1, rsbuf).buffer(2, xp).buffer(3, ybuf).bytes(4, prm)
                     .grid(*grid).threadgroup(tg, 1, 1))
                if epilogue == "residual":
                    d.buffer(7, res)
                ds.append(d)
            results.append((self._time(ds) / copies, mode))
        results.sort(key=lambda r: r[0])
        best_ms, best_mode = results[0]
        default_ms = [ms for ms, m in results if m == "crew"][0]
        if best_ms > default_ms * (1 - NOISE_MARGIN):
            best_ms, best_mode = default_ms, "crew"
        choice = Choice({}, best_mode, False, best_ms, default_ms)
        self.choices[key] = {"macros": {}, "grid_mode": best_mode, "ms": best_ms, "default_ms": default_ms, "variants": [(round(ms, 4), m) for ms, m in results]}
        return choice

    # ---- GDN -----------------------------------------------------------------------------------------------------
    def tune_gdn(self, hv: int, hk: int, dk: int, dv: int, conv_width: int, t: int, *, force: bool = False) -> Choice:
        key = gdn_key(hv, hk, dk, dv, conv_width, t)
        if key in self.choices and not force:
            c = self.choices[key]
            return Choice(dict(c["macros"]), "crew", False, c.get("ms", 0.0), c.get("default_ms", 0.0))
        nt = self.nt
        rng = np.random.default_rng(0)
        kd, vd = hk * dk, hv * dv
        conv_dim = 2 * kd + vd
        n1 = conv_dim + vd + 2 * hv
        proj = nt.Buffer(self.dev, f32_to_bf16(rng.standard_normal((t, n1)).astype(np.float32)).tobytes())
        conv_state = nt.Buffer(self.dev, f32_to_bf16(rng.standard_normal((conv_dim, conv_width - 1)).astype(np.float32)).tobytes())
        rec_state = nt.Buffer(self.dev, (rng.standard_normal((hv, dk, dv)) * 0.1).astype(np.float32).tobytes())
        aux = [nt.Buffer(self.dev, f32_to_bf16(rng.standard_normal((conv_dim, conv_width)).astype(np.float32) * 0.3).tobytes()),
               nt.Buffer(self.dev, (-np.exp(rng.uniform(-2, 1, hv))).astype(np.float32).tobytes()),
               nt.Buffer(self.dev, rng.standard_normal(hv).astype(np.float32).tobytes()),
               nt.Buffer(self.dev, np.ones(dv, np.float32).tobytes())]
        o_part = nt.Buffer(self.dev, kernels.gdn_workspace(t, hv, dv))
        out = nt.Buffer(self.dev, t * vd * 2)
        n_sg, grid, tg = self.grid("crew", 0)
        params = kernels.gdn_params(hv=hv, hk=hk, t_active=t, q_off=0, k_off=kd, v_off=2 * kd, z_off=conv_dim, a_off=conv_dim + vd,
                                    b_off=conv_dim + vd + hv, in_stride=n1, ab_stride=n1, ab_separate=False, out_stride=vd, n_sg=n_sg,
                                    key_dim=kd, eps=1e-6)
        default = kernels.gdn_macros(dk, dv, conv_width=conv_width, t=t)
        results = []
        for sl, spb in ((8, 4), (8, 2), (8, 1), (4, 4), (4, 2), (16, 2)):
            if dv % (sl * spb):
                continue
            macros = kernels.gdn_macros(dk, dv, conv_width=conv_width, t=t, slice_cols=sl, slices_per_block=spb)
            lib = nt.Library(self.dev, kernels.gdn_source(), macros)
            pso, pn = nt.Pipeline(lib, "gdn_mixer"), nt.Pipeline(lib, "gdn_norm")
            d = (nt.Dispatch().pipeline(pso).buffer(0, proj).buffer(1, proj).buffer(2, conv_state).buffer(3, rec_state).buffer(4, aux[0])
                 .buffer(5, aux[1]).buffer(6, aux[2]).buffer(7, o_part).bytes(9, params).grid(grid).threadgroup(tg).barrier())
            d2 = nt.Dispatch().pipeline(pn).buffer(0, o_part).buffer(1, proj).buffer(2, aux[3]).buffer(3, out).bytes(4, params).grid(t * hv).threadgroup(32)
            ms = self._time([d, d2])
            results.append((ms, macros))
        results.sort(key=lambda r: r[0])
        default_ms = next(ms for ms, m in results if m["SL"] == default["SL"] and m["SPB"] == default["SPB"])
        best_ms, bm = results[0]
        if best_ms > default_ms * (1 - NOISE_MARGIN):
            best_ms, bm = default_ms, default
        choice = Choice({"SL": bm["SL"], "SPB": bm["SPB"]}, "crew", False, best_ms, default_ms)
        self.choices[key] = {"macros": choice.macros, "grid_mode": "crew", "ms": best_ms, "default_ms": default_ms,
                             "variants": [(round(ms, 4), {"SL": m["SL"], "SPB": m["SPB"]}) for ms, m in results]}
        return choice

    def save(self, chip: str) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump({"chip": chip, "cores": self.cores, "written": time.strftime("%Y-%m-%d %H:%M"), "choices": self.choices}, f, indent=1)
