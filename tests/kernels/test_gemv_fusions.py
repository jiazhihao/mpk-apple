"""gemv_T fusions (design §5.1, §5.6; issues #19/#20) against numpy oracles that mirror the nn layer semantics:
RMSNorm scaling on the input (statistic from rmsnorm_stat or from a producer's per-block partials), residual-add
epilogue, silu·mul over chunk-interleaved gate|up rows, and the chained producer → consumer norm hoist."""

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle, pack_spec, random_spec
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt

K = 1024
EPS = 1e-6


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def rbf(x):
    """Round to BF16 and back (the kernel's single rounding)."""
    return bf16_to_f32(f32_to_bf16(np.asarray(x, dtype=np.float32)))


class Gemv:
    """One packed matrix and the dispatch plumbing for its fusion variants."""

    def __init__(self, dev, fmt, n, rows, t, lane_order="interleaved16", seed=3):
        self.dev, self.n, self.t = dev, n, t
        rng = np.random.default_rng(seed)
        self.spec = random_spec(fmt, n, K, rng)
        self.data, self.info, self.row_scales = pack_spec(self.spec, PackLayout(rows=rows, lane_order=lane_order))
        self.w = FORMATS.get(fmt).dequantize(self.spec).astype(np.float64)       # includes the per-tensor scale
        self.fmt = fmt
        self.wbuf, self.rsbuf = nt.Buffer(dev, self.data), nt.Buffer(dev, self.row_scales.tobytes())
        self.n_sg = 12 * dev.info().gpu_cores

    def run(self, x_bf16, *, norm=None, epilogue=None, residual=None, stat_out=False, t_active=None, out_bf16=None):
        """``x_bf16`` uint16 [T, K]; ``norm`` = (stat float32 array, parts, norm_w float32 [K]); returns
        ``(y, stat_out)`` with y float32 [T, N] (or [T, N/2] for silu_mul)."""
        t, n = self.t, self.n
        if out_bf16 is None:
            out_bf16 = epilogue is not None
        macros = kernels.gemv_macros(self.info, t=t, norm=norm is not None, epilogue=epilogue, stat_out=stat_out, out_bf16=out_bf16)
        pso = nt.Pipeline(nt.Library(self.dev, kernels.gemv_source(self.fmt), macros), "gemv_T")
        n_out = n // 2 if epilogue == "silu_mul" else n
        y = nt.Buffer(self.dev, t * n_out * 4); y.fill(0)
        parts = norm[1] if norm else 1
        t_act = t if t_active is None else t_active
        d = (nt.Dispatch().pipeline(pso).buffer(0, self.wbuf).buffer(1, self.rsbuf).buffer(2, nt.Buffer(self.dev, x_bf16.tobytes()))
             .buffer(3, y).bytes(4, kernels.gemv_params(n, self.info.n_blocks, self.n_sg, t_act, eps=EPS, stat_parts=parts))
             .grid(-(-(self.n_sg * 32) // 384)).threadgroup(384))
        if norm:
            d.buffer(5, nt.Buffer(self.dev, np.asarray(norm[0], np.float32).tobytes())).buffer(6, nt.Buffer(self.dev, np.asarray(norm[2], np.float32).tobytes()))
        if epilogue == "residual":
            d.buffer(7, nt.Buffer(self.dev, residual.tobytes()))
        so = None
        if stat_out:
            so = nt.Buffer(self.dev, t * self.info.n_blocks * 4); so.fill(0)
            d.buffer(8, so)
        r = nt.Queue(self.dev).run([d])
        assert not r.error, r.error
        raw = y.read(0, t * n_out * (2 if out_bf16 else 4))
        out = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(t, n_out) if out_bf16 else np.frombuffer(raw, dtype=np.float32).reshape(t, n_out)
        so_arr = np.frombuffer(so.read(0, t * self.info.n_blocks * 4), dtype=np.float32).reshape(t, self.info.n_blocks) if stat_out else None
        return out, so_arr


def _norm_inputs(rng, t, scale=1.0):
    h = f32_to_bf16(rng.standard_normal((t, K)).astype(np.float32) * scale)
    hf = bf16_to_f32(h)
    nw = (1.0 + rng.standard_normal(K).astype(np.float32) * 0.1).astype(np.float32)
    stat = (hf.astype(np.float64) ** 2).sum(-1).astype(np.float32)
    r = 1.0 / np.sqrt(stat.astype(np.float64) / K + EPS)
    x_ref = rbf(hf * r[:, None] * nw[None, :])                           # the reference norm's BF16 output
    return h, nw, stat, x_ref


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
@pytest.mark.parametrize("t", [1, 4])
def test_norm_input_matches_reference_norm(dev, fmt, t):
    g = Gemv(dev, fmt, 100, 16, t)
    rng = np.random.default_rng(5)
    h, nw, stat, x_ref = _norm_inputs(rng, t)
    y, _ = g.run(h, norm=(stat, 1, nw))
    ref = (x_ref.astype(np.float64) @ g.w.T).astype(np.float32)
    chk = check_against_oracle(y, ref)
    assert chk.ok(), chk
    # T*WPW > 64 takes the word path (NVFP4 T=4 above already does); the preconvert path is BF16/FP8 T ≤ 8


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
def test_residual_epilogue(dev, fmt):
    t, n = 3, 96
    g = Gemv(dev, fmt, n, 16, t)
    rng = np.random.default_rng(7)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(t, K)).astype(np.float32))
    res = f32_to_bf16(rng.standard_normal((t, n)).astype(np.float32))
    y, _ = g.run(x, epilogue="residual", residual=res)
    ref = rbf(bf16_to_f32(x).astype(np.float64) @ g.w.T + bf16_to_f32(res))
    chk = check_against_oracle(y, ref)
    assert chk.ok() and chk.max_ulp_elementwise <= 1, chk


@pytest.mark.parametrize("fmt,rows", [("nvfp4", 16), ("fp8_e4m3", 16), ("bf16", 8), ("int8", 16), ("int4_affine", 16)])
@pytest.mark.parametrize("t", [1, 2])
def test_silu_mul_epilogue(dev, fmt, rows, t):
    n = 8 * rows                                                          # 8 blocks, chunk = rows/2 outputs each
    g = Gemv(dev, fmt, n, rows, t)
    rng = np.random.default_rng(9)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(t, K)).astype(np.float32))
    y, _ = g.run(x, epilogue="silu_mul")
    full = bf16_to_f32(x).astype(np.float64) @ g.w.T                      # [t, n] in pack row order
    c = rows // 2
    gate = np.concatenate([full[:, b * rows: b * rows + c] for b in range(n // rows)], axis=1)
    up = np.concatenate([full[:, b * rows + c: (b + 1) * rows] for b in range(n // rows)], axis=1)
    ref = rbf(gate / (1 + np.exp(-gate)) * up)
    assert y.shape == (t, n // 2)
    chk = check_against_oracle(y, ref)
    assert chk.ok(), chk


def test_hoisted_stat_chain(dev):
    """Producer (residual epilogue + STAT_OUT) → consumer (NORM from the producer's per-block partials): the two
    dispatches of one layer boundary, equal to the layer oracle (norm of the BF16 residual stream, then GEMV)."""
    t = 4
    prod = Gemv(dev, "fp8_e4m3", K, 16, t, seed=11)                       # N = K so the output feeds the next GEMV
    cons = Gemv(dev, "nvfp4", 128, 16, t, seed=12)
    rng = np.random.default_rng(13)
    x = f32_to_bf16(rng.uniform(-1, 1, size=(t, K)).astype(np.float32))
    res = f32_to_bf16(rng.standard_normal((t, K)).astype(np.float32))
    h_new, partials = prod.run(x, epilogue="residual", residual=res, stat_out=True)
    h_ref = rbf(bf16_to_f32(x).astype(np.float64) @ prod.w.T + bf16_to_f32(res))
    assert check_against_oracle(h_new, h_ref).ok()
    assert partials.shape == (t, prod.info.n_blocks)
    ssq_ref = (h_ref.astype(np.float64) ** 2).sum(-1)
    assert np.allclose(partials.sum(-1), ssq_ref, rtol=1e-5)
    nw = (1.0 + rng.standard_normal(K).astype(np.float32) * 0.1).astype(np.float32)
    y, _ = cons.run(f32_to_bf16(h_new), norm=(partials.reshape(-1), prod.info.n_blocks, nw))
    r = 1.0 / np.sqrt(ssq_ref / K + EPS)
    x2 = rbf(h_ref * r[:, None] * nw[None, :])
    ref = (x2.astype(np.float64) @ cons.w.T).astype(np.float32)
    chk = check_against_oracle(y, ref)
    assert chk.ok(), chk


def test_dynamic_t_with_fusions(dev):
    t = 4
    g = Gemv(dev, "bf16", 64, 16, t)
    rng = np.random.default_rng(17)
    h, nw, stat, x_ref = _norm_inputs(rng, t)
    res = f32_to_bf16(rng.standard_normal((t, 64)).astype(np.float32))
    y, so = g.run(h, norm=(stat, 1, nw), epilogue="residual", residual=res, stat_out=True, t_active=2)
    ref = rbf(x_ref.astype(np.float64) @ g.w.T + bf16_to_f32(res))
    assert check_against_oracle(y[:2], ref[:2]).ok()
    assert np.all(y[2:] == 0) and np.all(so[2:] == 0)                    # untouched beyond t_active
    assert np.allclose(so[:2].sum(-1), (ref[:2].astype(np.float64) ** 2).sum(-1), rtol=1e-5)


def test_autotuner_picks_and_caches(dev, tmp_path):
    """The autotuner times the GEMV and GDN variants of a shape, returns a valid choice (never slower than the
    default by more than the noise margin) and round-trips its cache."""
    from monolith.compiler.autotune import Autotuner, Choice
    from monolith.formats import FORMATS, PackLayout

    cache = tmp_path / "autotune.json"
    tuner = Autotuner(dev, dev.info().gpu_cores, str(cache), reps=2, warmup_ms=5.0)
    rng = np.random.default_rng(1)
    _, info, _ = pack_spec(random_spec("bf16", 512, K, rng), PackLayout(rows=16))
    c = tuner.tune_gemv(info, 1, "residual", False)
    assert isinstance(c, Choice) and c.ms > 0 and c.ms <= c.default_ms * 1.001 and c.grid_mode in ("crew", "crew2", "block")
    cn = tuner.tune_gemv(info, 1, None, True)
    assert cn.ms > 0 and isinstance(cn.fuse_norm, bool)
    g = tuner.tune_gdn(4, 4, 128, 128, 4, 1)
    assert g.ms > 0 and g.macros["SL"] in ("4u", "8u", "16u")
    m = tuner.tune_gemm(info, 8, "residual")                                # the tensor-ops tile: one or two threadgroups per core
    assert m.ms > 0 and m.grid_mode in ("crew", "crew2") and m.ms <= m.default_ms * 1.001
    tuner.save("test-chip")
    again = Autotuner(dev, dev.info().gpu_cores, str(cache))
    assert again.tune_gemv(info, 1, "residual", False).macros == c.macros and len(again.choices) == 4
    assert again.tune_gemm(info, 8, "residual").grid_mode == m.grid_mode


def test_row_range_equals_the_slice_of_the_whole(dev):
    """A dispatch over blocks [block0, block0 + n) of a slab writes the same outputs as the whole-slab dispatch's slice,
    with the residual epilogue, the input norm and the STAT_OUT partials relative to the range (design §5.12: a
    mixer's gate rows as their own dispatch)."""
    rng = np.random.default_rng(17)
    fmt, n, rows, t = "nvfp4", 512, 16, 2
    gm = Gemv(dev, fmt, n, rows, t, seed=17)
    x = f32_to_bf16(rng.standard_normal((t, K)).astype(np.float32))
    residual = f32_to_bf16(rng.standard_normal((t, n)).astype(np.float32))
    stat = np.array([float((bf16_to_f32(x[i]).astype(np.float64) ** 2).sum()) for i in range(t)], np.float32)
    nw = rbf(1.0 + rng.standard_normal(K) * 0.1)
    full, full_stat = gm.run(x, norm=(stat, 1, nw), epilogue="residual", residual=residual, stat_out=True)
    start, count = 128, 256                                                    # blocks 8 .. 24 of 32
    b0, nb = start // rows, count // rows
    macros = kernels.gemv_macros(gm.info, t=t, norm=True, epilogue="residual", stat_out=True, out_bf16=True)
    pso = nt.Pipeline(nt.Library(dev, kernels.gemv_source(fmt), macros), "gemv_T")
    y = nt.Buffer(dev, t * count * 2); y.fill(0)
    so = nt.Buffer(dev, t * nb * 4); so.fill(0)
    d = (nt.Dispatch().pipeline(pso).buffer(0, gm.wbuf).buffer(1, gm.rsbuf).buffer(2, nt.Buffer(dev, x.tobytes())).buffer(3, y)
         .bytes(4, kernels.gemv_params(count, nb, gm.n_sg, t, eps=EPS, stat_parts=1, block0=b0))
         .buffer(5, nt.Buffer(dev, stat.tobytes())).buffer(6, nt.Buffer(dev, nw.astype(np.float32).tobytes()))
         .buffer(7, nt.Buffer(dev, np.ascontiguousarray(residual[:, start: start + count]).tobytes())).buffer(8, so)
         .grid(-(-(gm.n_sg * 32) // 384)).threadgroup(384))
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    got = bf16_to_f32(np.frombuffer(y.read(0, t * count * 2), dtype=np.uint16)).reshape(t, count)
    assert np.array_equal(got, full[:, start: start + count])
    parts = np.frombuffer(so.read(0, t * nb * 4), dtype=np.float32).reshape(t, nb)
    assert np.array_equal(parts, full_stat[:, b0: b0 + nb])

