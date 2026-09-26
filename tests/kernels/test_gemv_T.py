"""gemv_T on every format, both lane orders, R in {4, 16}, T in {1, 2, 4, 8}: ≤ 2 ULP BF16 vs the exact oracle."""
import struct

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import check_against_oracle, pack_spec, random_spec
from monolith.formats import FORMATS, PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.runtime import _native as nt

K = 1024


@pytest.fixture(scope="module")
def dev():
    return nt.Device()


def _run(dev, fmt, n, rows, t, lane_order, t_active=None, one_block_per_sg=False, extra_macros=None, K=K, placement="inline"):
    rng = np.random.default_rng(3)
    spec = random_spec(fmt, n, K, rng)
    data, info, row_scales = pack_spec(spec, PackLayout(rows=rows, lane_order=lane_order, scale_placement=placement))
    x = rng.uniform(-1, 1, size=(t, K)).astype(np.float32)
    xb = f32_to_bf16(x)
    macros = kernels.gemv_macros(info, t=t)
    macros.update(extra_macros or {})
    pso = nt.Pipeline(nt.Library(dev, kernels.gemv_source(fmt), macros), "gemv_T")
    n_sg = info.n_blocks if one_block_per_sg else 12 * dev.info().gpu_cores
    tg = 64 if one_block_per_sg else 384
    y = nt.Buffer(dev, t * n * 4); y.fill(0)
    t_act = t if t_active is None else t_active
    d = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, data)).buffer(1, nt.Buffer(dev, row_scales.tobytes()))
         .buffer(2, nt.Buffer(dev, xb.tobytes())).buffer(3, y).bytes(4, struct.pack("<IIIIfIII", n, info.n_blocks, n_sg, t_act, 1.0, 0, 0, 0))
         .grid(-(-(n_sg * 32) // tg)).threadgroup(tg))
    r = nt.Queue(dev).run([d])
    assert not r.error, r.error
    out = np.frombuffer(y.read(0, t * n * 4), dtype=np.float32).reshape(t, n)
    w = FORMATS.get(fmt).dequantize(spec)                            # the per-tensor scale is the row-scale table
    ref = (bf16_to_f32(xb).astype(np.float64) @ w.astype(np.float64).T).astype(np.float32)
    return out, ref, t_act


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
@pytest.mark.parametrize("lane_order", ["contiguous", "interleaved16"])
@pytest.mark.parametrize("rows,t", [(16, 1), (4, 1), (16, 2), (8, 4), (8, 8)])
def test_gemv_matches_oracle(dev, fmt, lane_order, rows, t):
    out, ref, _ = _run(dev, fmt, 100, rows, t, lane_order)        # 100 rows: partial last block
    chk = check_against_oracle(out, ref)
    assert chk.ok(), chk


@pytest.mark.parametrize("fmt", ["int4_affine", "nvfp4"])
@pytest.mark.parametrize("t", [1, 4])                                  # both activation paths (X_PRECONVERT at T = 1)
def test_gemv_ragged_stripe(dev, fmt, t):
    """K = 3584: a lane's stripe is 112 columns — 3.5 words of 32 nibbles (the last one partial, its tail masked),
    the scale bytes in the tail word, and (int4_affine) stripes starting 0/48/32/16 columns into a group of 64."""
    out, ref, _ = _run(dev, fmt, 40, 8, t, "interleaved16", K=3584)
    chk = check_against_oracle(out, ref)
    assert chk.ok(), chk


def test_gemv_conventional_geometry_and_dynamic_t(dev):
    out, ref, _ = _run(dev, "fp8_e4m3", 96, 16, 2, "interleaved16", one_block_per_sg=True)
    assert check_against_oracle(out, ref).ok()
    out, ref, t_act = _run(dev, "nvfp4", 64, 16, 4, "interleaved16", t_active=2)
    assert check_against_oracle(out[:2], ref[:2]).ok() and t_act == 2
    assert np.all(out[2:] == 0)                                    # tokens beyond t_active are not written


@pytest.mark.parametrize("variant", ["0", "1", "2", "3"])
@pytest.mark.parametrize("rows,t", [(16, 1), (4, 1), (8, 4)])
def test_nvfp4_decode_variants_are_exact(dev, variant, rows, t):
    out, ref, _ = _run(dev, "nvfp4", 100, rows, t, "interleaved16", extra_macros={"NVFP4_DECODE": variant})
    chk = check_against_oracle(out, ref)
    assert chk.ok() and chk.max_rel_err < 1e-6, chk         # the decodes are bit-exact; only accumulation order differs


@pytest.mark.parametrize("fmt,K", [("nvfp4", 4096), ("nvfp4", 5120), ("nvfp4", 12288), ("int8", 4096), ("int8", 5120), ("int4_affine", 4096), ("nvfp4", 3584)])
@pytest.mark.parametrize("lane_order", ["interleaved16", "contiguous"])
@pytest.mark.parametrize("rows,t", [(16, 1), (8, 4)])
def test_gemv_block_scale_placement(dev, fmt, K, lane_order, rows, t):
    """The block's scales in their own region (#101): a lane's run starting mid-word (K = 5120: 10 bytes per lane,
    two words for some lanes), the K = 12288 run of 24 bytes, INT8's halves and INT4's float pairs, both lane orders;
    the ragged K = 3584 stays inline (its tail half-word holds the scales)."""
    out, ref, _ = _run(dev, fmt, 100, rows, t, lane_order, K=K, placement="block")
    chk = check_against_oracle(out, ref)
    assert chk.ok(), chk
    inline, _, _ = _run(dev, fmt, 100, rows, t, lane_order, K=K, placement="inline")
    assert np.array_equal(out, inline), "the same codes and scales in the same order must give the same bits whatever the placement"
