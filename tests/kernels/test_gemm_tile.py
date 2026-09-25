"""The accelerator GEMM tile (plan M9, #50): the cooperative-tensor register layout the fill assumes, read back from
the tensor-ops API for every thread and element; then gemm_tile against the format oracle on every format, both lane
orders, TM = 8 / 16 / 32 with T_act < TM, BF16 and float outputs, and the x_permute column order. The reference holds
the dequantized weights in BF16 — the accelerator's operand dtype and the HF reference model's parameter dtype (exact
for NVFP4, FP8 and BF16; INT8 and affine INT4 products are rounded once, unlike the FP32-dequant GEMV path)."""

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


def _lib(dev, fmt, macros):
    return nt.Library(dev, kernels.gemm_source(fmt), macros, language_version=kernels.MSL_TENSOR_OPS)


@pytest.mark.parametrize("tm,tn,tk", [(8, 64, 64), (16, 64, 64), (32, 64, 64), (8, 32, 128), (16, 16, 256), (32, 16, 256)])
def test_cooperative_layout_matches_the_fill_formula(dev, tm, tn, tk):
    """Thread ``lane`` holds runs of 4 consecutive inner coordinates at 4·(bit0 + 2·bit3) + 16·jump for outer rows
    (bits 1, 2, 4) + 8·slot. The right operand (inner = k, outer = n) orders its elements q, slot (8), jump; the
    destination (inner = n, outer = m) orders them q, slot (2), jump, 16-row block (valid iff m < TM)."""
    spec = random_spec("bf16", 64, K, np.random.default_rng(0))
    _, info, _ = pack_spec(spec, PackLayout(rows=16))
    macros = kernels.gemm_macros(info, tm=tm, tn=tn, tk=tk)
    pso = nt.Pipeline(_lib(dev, "bf16", macros), "coop_layout")
    stride = 4 + 3 * 1024
    out = nt.Buffer(dev, 32 * stride * 4)
    out.fill(0)
    r = nt.Queue(dev).run([nt.Dispatch().pipeline(pso).buffer(0, out).grid(1).threadgroup(32)])
    assert not r.error, r.error
    a = np.frombuffer(out.read(0, 32 * stride * 4), dtype=np.int32).reshape(32, stride)
    nb_c, ns_b, nj_c = max(tm, 16) // 16, tn // 8, tn // 16
    for lane in range(32):
        cap_b, cap_c = int(a[lane, 0]), int(a[lane, 1])
        assert cap_b == 128 and cap_c == 8 * nj_c * nb_c
        c0b = 4 * ((lane & 1) + 2 * ((lane >> 3) & 1))
        c1b = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1)
        rec = a[lane, 4:4 + 3 * cap_b].reshape(cap_b, 3)
        for i in range(cap_b):
            q, s, jump = i & 3, (i >> 2) % ns_b, (i >> 2) // ns_b
            assert tuple(rec[i]) == (1, c0b + 16 * jump + q, c1b + 8 * s), (lane, i, rec[i])
        rec = a[lane, 4 + 3 * cap_b:4 + 3 * cap_b + 3 * cap_c].reshape(cap_c, 3)
        for i in range(cap_c):
            q, s2, jump, blk = i & 3, (i >> 2) & 1, ((i >> 2) >> 1) % nj_c, ((i >> 2) >> 1) // nj_c
            n, m = c0b + 16 * jump + q, 16 * blk + c1b + 8 * s2
            assert tuple(rec[i]) == (1 if m < tm else 0, n, m), (lane, i, rec[i])


def test_x_permute_column_order():
    kl = K // 32
    perm = kernels.x_permute_columns(K, 32, tk=64)
    assert sorted(perm.tolist()) == list(range(K))
    # slot (bit0, bit3, jump, q) of a tile <-> pack column (TK/4)·(bit0 + 2·bit3) + 4·jump + q; pack column 32 = lane 1's word 0
    assert perm[:4].tolist() == [0, 1, 2, 3] and perm[4] == 16 and perm[8] == kl and perm[16] == 4 and perm[64] == 2 * kl
    perm = kernels.x_permute_columns(K, 32, tk=256)                     # the default tile: quarters of 64 columns
    assert sorted(perm.tolist()) == list(range(K))
    assert perm[4] == 2 * kl and perm[8] == 4 * kl and perm[16] == 4 and perm[64] == 16 and perm[256] == 8 * kl


def _run(dev, fmt, n, k, tm, t_act, lane_order, out_bf16=False, rows=16, tn=None, tk=None):
    rng = np.random.default_rng(5)
    spec = random_spec(fmt, n, k, rng)
    data, info, row_scales = pack_spec(spec, PackLayout(rows=rows, lane_order=lane_order))
    f = FORMATS.get(fmt)
    x = rng.uniform(-1, 1, size=(t_act, k)).astype(np.float32)
    xb = f32_to_bf16(x)
    macros = kernels.gemm_macros(info, tm=tm, out_bf16=out_bf16, tn=tn, tk=tk)
    tn, tk = int(macros["TN"].rstrip("u")), int(macros["TK"].rstrip("u"))
    lib = _lib(dev, fmt, macros)
    pso, ppso = nt.Pipeline(lib, "gemm_tile"), nt.Pipeline(lib, "x_permute")
    xp = nt.Buffer(dev, tm * k * 2)
    y = nt.Buffer(dev, tm * n * (2 if out_bf16 else 4)); y.fill(0)
    tg = min(384, pso.max_threads_per_threadgroup)
    n_sg = (tg // 32) * dev.info().gpu_cores
    d0 = (nt.Dispatch().pipeline(ppso).buffer(0, nt.Buffer(dev, xb.tobytes())).buffer(1, xp)
          .bytes(2, kernels.x_permute_params(k, t_act, tm, int(f.weights_per_word), tk)).grid(-(-(tm * k) // 256)).threadgroup(256).barrier())
    d1 = (nt.Dispatch().pipeline(pso).buffer(0, nt.Buffer(dev, data)).buffer(1, nt.Buffer(dev, row_scales.tobytes())).buffer(2, xp).buffer(3, y)
          .bytes(4, kernels.gemm_params(n, kernels.gemm_tiles(n, tn), n_sg, t_act)).grid(-(-(n_sg * 32) // tg)).threadgroup(tg))
    r = nt.Queue(dev).run([d0, d1])
    assert not r.error, r.error
    if out_bf16:
        out = bf16_to_f32(np.frombuffer(y.read(0, tm * n * 2), dtype=np.uint16).reshape(tm, n))
    else:
        out = np.frombuffer(y.read(0, tm * n * 4), dtype=np.float32).reshape(tm, n)
    xperm = bf16_to_f32(np.frombuffer(xp.read(0, tm * k * 2), dtype=np.uint16).reshape(tm, k))
    assert np.array_equal(xperm[:t_act], bf16_to_f32(xb)[:, kernels.x_permute_columns(k, int(f.weights_per_word), tk)]) and np.all(xperm[t_act:] == 0)
    rs = row_scales.astype(np.float64)[:, None]                            # the per-tensor scale, applied in FP32 at the epilogue
    w = bf16_to_f32(f32_to_bf16((f.dequantize(spec) / rs).astype(np.float32))) * rs   # the BF16 operand the accelerator multiplies
    ref = (bf16_to_f32(xb).astype(np.float64) @ w.T).astype(np.float32)
    return out, ref


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
@pytest.mark.parametrize("lane_order", ["contiguous", "interleaved16"])
@pytest.mark.parametrize("tm,t_act", [(8, 8), (8, 3), (16, 16), (32, 21)])
def test_gemm_tile_matches_oracle(dev, fmt, lane_order, tm, t_act):
    out, ref = _run(dev, fmt, 200, K, tm, t_act, lane_order)           # 200 rows: a partial last tile
    chk = check_against_oracle(out[:t_act], ref)
    assert chk.ok(), chk
    assert np.all(out[t_act:] == 0)


def test_gemm_tile_bf16_output_and_k2048(dev):
    out, ref = _run(dev, "nvfp4", 128, 2048, 16, 16, "interleaved16", out_bf16=True)
    chk = check_against_oracle(out, ref)
    assert chk.ok_rounded(), chk                                         # one BF16 rounding of the FP32 result
    out, ref = _run(dev, "int4_affine", 128, 2048, 8, 8, "interleaved16")
    assert check_against_oracle(out[:8], ref).ok()


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8", "int4_affine"])
@pytest.mark.parametrize("tn,tk", [(64, 64), (32, 128)])
def test_gemm_tile_other_tile_shapes(dev, fmt, tn, tk):
    """The non-default tiles (the default cases above run 16 × 256 up to 16 tokens and 32 × 128 at 32)."""
    out, ref = _run(dev, fmt, 200, 2048, 16, 11, "interleaved16", tn=tn, tk=tk)
    chk = check_against_oracle(out[:11], ref)
    assert chk.ok(), chk
