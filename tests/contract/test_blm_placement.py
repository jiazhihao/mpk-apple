"""The block scale placement (#101): a lane-row unit of whole payload words, the block's scales in their own region —
the pack streams the weights' bytes; the round trip, the offsets the kernels compute, the macros."""

import numpy as np
import pytest

from monolith import kernels
from monolith.bench import pack_spec, random_spec
from monolith.formats import FORMATS, PackLayout
from monolith.formats.blm import LANES, unpack_blm


@pytest.mark.parametrize("fmt,k", [("nvfp4", 4096), ("nvfp4", 5120), ("nvfp4", 12288), ("nvfp4", 3584), ("int8", 4096), ("int8", 5120),
                                   ("int4_affine", 4096), ("int4_affine", 2048), ("fp8_e4m3", 4096), ("bf16", 4096)])
@pytest.mark.parametrize("lane_order", ["interleaved16", "contiguous"])
def test_block_placement_round_trip_and_bytes(fmt, k, lane_order):
    rng = np.random.default_rng(1)
    spec = random_spec(fmt, 40, k, rng)                                          # a partial last block
    f = FORMATS.get(fmt)
    d_in, i_in, _ = pack_spec(spec, PackLayout(rows=16, lane_order=lane_order, scale_placement="inline"))
    d_bl, i_bl, _ = pack_spec(spec, PackLayout(rows=16, lane_order=lane_order, scale_placement="block"))
    assert i_bl.n_blocks == i_in.n_blocks == 3 and len(d_bl) == i_bl.nbytes and len(d_in) == i_in.nbytes
    p_in, s_in = unpack_blm(d_in, i_in)
    p_bl, s_bl = unpack_blm(d_bl, i_bl)
    assert np.array_equal(p_in, p_bl) and ((s_in is None and s_bl is None) or np.array_equal(s_in, s_bl))
    assert np.array_equal(f.dequantize(f.unpack_pack(d_bl, i_bl)), f.dequantize(spec))
    p16 = -(-i_in.payload_bytes // 16) * 16
    s = i_in.scale_bytes
    two_words = s and max(-(-((lane * s) % 16 + s) // 16) for lane in range(LANES)) > 1
    if s and (p16 + s >= i_in.unit_bytes or two_words):
        assert i_bl.scale_placement == "inline" and d_bl == d_in          # nothing to save (INT4's 64 + 16, a ragged tail half-word), or two scale words per lane
    elif s:
        assert i_bl.scale_placement == "block" and i_bl.unit_bytes == -(-i_bl.payload_bytes // 16) * 16 and i_bl.nbytes < i_in.nbytes
        # the lane's scales sit where the kernel's SCALE_WORD / SCALE_SOFF arithmetic says
        buf = np.frombuffer(d_bl, dtype=np.uint8)
        for b, r, lane in ((0, 0, 0), (1, 5, 31), (2, 7, 11)):
            row = b * 16 + r
            if row < 40:
                off = i_bl.scale_offset(b, r, lane)
                assert np.array_equal(buf[off: off + i_bl.scale_bytes], s_bl[row, lane])
        s = i_bl.scale_bytes
        assert i_bl.scale_words == max(-(-((lane * s) % 16 + s) // 16) for lane in range(LANES))
        m = kernels.unit_geometry(i_bl)
        assert m["SCALE_PLACEMENT"] == "1" and m["SCALE_RUN"] == f"{s}u" and m["SCALE_WORDS"] == str(i_bl.scale_words)
        assert m["SCALE_UNIT_BYTES"] == f"{f.scale_unit_bytes}u" and s % f.scale_unit_bytes == 0
    else:
        assert i_bl.scale_placement == "inline" and d_bl == d_in                  # no scales: nothing moves


def test_block_placement_streams_the_weights_bytes():
    """NVFP4 at K = 4096: 36,864 bytes per block of 16 rows — exactly 16 × 4096 × 9/16 — against the inline 40,960."""
    rng = np.random.default_rng(2)
    spec = random_spec("nvfp4", 64, 4096, rng)
    _, i_in, _ = pack_spec(spec, PackLayout(rows=16))
    _, i_bl, _ = pack_spec(spec, PackLayout(rows=16, scale_placement="block"))
    assert i_in.block_bytes == 40960 and i_bl.block_bytes == 36864 == 16 * 4096 * 9 // 16
    assert i_bl.scale_words == 1 and i_bl.scale_region_bytes == 4096
    _, i5, _ = pack_spec(random_spec("nvfp4", 64, 5120, rng), PackLayout(rows=16, scale_placement="block"))
    assert i5.scale_placement == "inline" and i5.block_bytes == 16 * 32 * 96           # 10 scale bytes per lane would span two words: inline stays
    _, i8, _ = pack_spec(random_spec("int8", 64, 4096, rng), PackLayout(rows=16, scale_placement="block"))
    assert i8.scale_placement == "block" and i8.block_bytes == 16 * 32 * 128 + 16 * 32 * 8   # 8 scale bytes per lane: one word
    with pytest.raises(ValueError):
        PackLayout(rows=16, scale_placement="leading") and pack_spec(spec, PackLayout(rows=16, scale_placement="leading"))
