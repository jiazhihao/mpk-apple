import numpy as np
import pytest

from monolith.formats import FORMATS, PackLayout
from monolith.formats.blm import LANES
from monolith.formats.fp import e2m1_to_f32, e4m3_to_f32, unpack_nibbles

rng = np.random.default_rng(0)
K = 1024   # every stripe holds whole scale groups: K/32 = 32 columns per lane


def _w(n=40, k=K):
    return (rng.standard_normal((n, k)) * 0.02).astype(np.float32)


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8"])
def test_quantize_dequantize_is_close_and_exact_on_requantize(fmt):
    f = FORMATS.get(fmt)
    w = _w()
    spec = f.quantize(w)
    wq = f.dequantize(spec)
    tol = {"nvfp4": 0.35, "fp8_e4m3": 0.07, "bf16": 0.004, "int8": 0.01}[fmt]           # relative RMS error bounds
    assert np.sqrt(np.mean((wq - w) ** 2)) / np.sqrt(np.mean(w ** 2)) < tol
    assert np.array_equal(f.dequantize(f.quantize(wq)), wq)                                # idempotent on its own grid


def test_nvfp4_dequant_matches_reference_formula():
    f = FORMATS.get("nvfp4")
    spec = f.quantize(_w(8, 64))
    codes = unpack_nibbles(spec.tensors["weight"])                                       # low nibble first
    ref = np.empty((8, 64), dtype=np.float32)
    for n in range(8):
        for k in range(64):
            ref[n, k] = e2m1_to_f32(codes[n, k]) * e4m3_to_f32(spec.tensors["weight_scale"][n, k // 16]) * np.float32(spec.params["weight_scale_2"])
    assert np.array_equal(f.dequantize(spec), ref)
    got = f.unpack({"weight": spec.tensors["weight"], "weight_scale": spec.tensors["weight_scale"],
                    "weight_scale_2": np.float32(spec.params["weight_scale_2"])}, shape=(8, 64))
    assert np.array_equal(f.dequantize(got), ref)


@pytest.mark.parametrize("fmt", ["nvfp4", "fp8_e4m3", "bf16", "int8"])
@pytest.mark.parametrize("lane_order", ["contiguous", "interleaved16"])
@pytest.mark.parametrize("rows", [4, 16])
def test_pack_roundtrip_and_geometry(fmt, lane_order, rows):
    f = FORMATS.get(fmt)
    spec = f.quantize(_w(n=37))                       # 37 rows: the last block is partial
    layout = PackLayout(rows=rows, lane_order=lane_order)
    data, info = f.pack(spec, layout)
    assert info.n_blocks == -(-37 // rows) and len(data) == info.nbytes and info.unit_bytes % 16 == 0
    expected_unit = {"nvfp4": 16 + 2, "fp8_e4m3": 32, "bf16": 64, "int8": 32 + 2}[fmt]  # K/32 = 32 columns per lane
    assert info.unit_bytes == (expected_unit + 15) // 16 * 16
    back = f.unpack_pack(data, info)
    assert np.array_equal(f.dequantize(back), f.dequantize(spec))                        # bit-exact round trip
    # the kernel's index: word j of (row, lane) sits at unit_offset(...)
    payload = spec.tensors["weight"].view(np.uint8).reshape(37, -1)
    per_lane = payload.shape[1] // LANES
    row, lane, blk = 5, 17, 5 // rows
    unit = np.frombuffer(data, dtype=np.uint8)[info.unit_offset(blk, row % rows, lane): info.unit_offset(blk, row % rows, lane) + 16]
    assert np.array_equal(unit[: min(16, per_lane)], payload[row, lane * per_lane: lane * per_lane + min(16, per_lane)])


def test_pack_rejects_bad_stripes():
    f = FORMATS.get("nvfp4")
    with pytest.raises(ValueError):
        f.pack(f.quantize(_w(4, 64)), PackLayout())   # 64/32 = 2 columns per lane: no whole scale group


def test_unit_sizes_for_the_target_shapes():
    f4, f8 = FORMATS.get("nvfp4"), FORMATS.get("fp8_e4m3")
    _, i4 = f4.pack(f4.quantize(np.zeros((16, 5120), np.float32)), PackLayout(rows=16))
    _, i8 = f8.pack(f8.quantize(np.zeros((16, 5120), np.float32)), PackLayout(rows=16))
    assert (i4.payload_bytes, i4.scale_bytes, i4.unit_bytes) == (80, 10, 96) and i8.unit_bytes == 160   # p13's units
