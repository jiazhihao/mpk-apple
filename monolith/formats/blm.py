"""The block-lane-major (BLM) pack (design D8, §5.5): the layout engine shared by every format plugin.

A weight matrix ``[N, K]`` is cut into blocks of ``R`` rows. Inside a block the 32 lanes of a SIMD-group own 32
column stripes of ``K/32`` columns; the bytes a lane needs for one row of its stripe — the stripe's payload followed
by that stripe's block scales — form one **lane-row unit** of ``U`` bytes (padded to a multiple of 16). A block is
``R × 32`` units, ordered by the profile's lane order:

* ``contiguous``  — unit ``(lane, row)`` at ``(lane·R + row)·U``: a lane reads one contiguous run per block
  (the M3 Pro's order; ties the other on Apple9).
* ``interleaved16`` — the block is ``R·U/16`` steps of 32 lanes × 16 bytes: the ``j``-th 16-byte word of
  ``(row, lane)`` at ``((row·U/16 + j)·32 + lane)·16``, so one load instruction of the SIMD-group reads 512
  contiguous bytes (the order that streams at 95 % of nominal on Apple10; hardware report §3 N1).

Scale placement (``PackLayout.scale_placement``): ``inline`` keeps a lane-row's block scales inside its unit
(``[payload | scales | pad16]``: at K = 4096 an NVFP4 unit is 64 + 8 → 80 bytes, 11 % of padding on the bus);
``block`` (#101) keeps the unit to whole payload words and puts the block's scales in their own region after its
payload words — ``[row][lane][S]`` bytes, padded to 16 and by ``(scale_words − 1)·16`` more so the last lane's
whole-word loads stay inside the block — so the pack streams the weights' bytes: 36,864 per block of 16 rows at
K = 4096 (the inline 40,960). A lane's scales start ``(lane·S) % 16`` bytes into a word: the kernels load
``scale_words`` words and index the scales from that offset (``SCALE_SOFF``).

The matrix's per-tensor scale (NVFP4 ``weight_scale_2``, FP8 ``weight_scale``) is metadata (``PackInfo``), applied by
the kernel once per output. Everything here is numpy; the same index arithmetic is what the kernels use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .base import PackLayout

LANES = 32


@dataclass(frozen=True)
class PackInfo:
    format: str
    n: int                    # rows of the matrix
    k: int                    # columns
    rows: int                 # R, rows per block
    unit_bytes: int           # U, bytes per (lane, row) incl. scales and padding
    payload_bytes: int        # P, weight bytes per (lane, row)
    scale_bytes: int          # S, scale bytes per (lane, row)
    lane_order: str
    n_blocks: int
    tensor_scale: float = 1.0
    scale_group: int = 0      # weights per block scale (0 = no block scales)
    scale_placement: str = "inline"   # "inline": scales inside the unit; "block": the block's scale region after its payload words

    @property
    def scale_words(self) -> int:
        """16-byte words a lane loads for its row's scales: inline, the words past the payload's whole ones that the
        unit's scale bytes reach; block, the most any lane's run of S bytes spans from its start ``(lane·S) % 16``."""
        p, s = self.payload_bytes, self.scale_bytes
        if not s:
            return 0
        if self.scale_placement == "inline":
            return -(-(p + s) // 16) - p // 16
        return max(-(-((lane * s) % 16 + s) // 16) for lane in range(LANES))

    @property
    def scale_region_bytes(self) -> int:
        """The block's scale region (``block`` placement): ``[row][lane][S]`` padded to 16 plus the over-fetch margin."""
        if self.scale_placement == "inline" or not self.scale_bytes:
            return 0
        return _pad16(self.rows * LANES * self.scale_bytes) + (self.scale_words - 1) * 16

    @property
    def block_bytes(self) -> int:
        return self.rows * LANES * self.unit_bytes + self.scale_region_bytes

    @property
    def nbytes(self) -> int:
        return self.n_blocks * self.block_bytes

    @property
    def words_per_unit(self) -> int:
        return self.unit_bytes // 16

    def unit_offset(self, block: int, row: int, lane: int, word: int = 0) -> int:
        """Byte offset of the ``word``-th 16-byte word of lane-row ``(row, lane)`` in ``block`` — the kernel's index."""
        base = block * self.block_bytes
        if self.lane_order == "contiguous":
            return base + (lane * self.rows + row) * self.unit_bytes + word * 16
        return base + ((row * self.words_per_unit + word) * LANES + lane) * 16

    def scale_offset(self, block: int, row: int, lane: int) -> int:
        """Byte offset of lane-row ``(row, lane)``'s first scale byte (``block`` placement) — the kernels' ``SCALE_WORD``
        and ``SCALE_SOFF`` arithmetic."""
        if self.scale_placement != "block":
            raise ValueError("scale_offset: inline scales live inside the unit")
        return block * self.block_bytes + self.rows * LANES * self.unit_bytes + (row * LANES + lane) * self.scale_bytes


def _pad16(x: int) -> int:
    return (x + 15) // 16 * 16


def pack_blm(payload: np.ndarray, scales: Optional[np.ndarray], layout: PackLayout, *, format: str, k: int,
             tensor_scale: float = 1.0, scale_group: int = 0) -> Tuple[bytes, PackInfo]:
    """``payload``: uint8 ``[N, 32, P]`` (lane ℓ's stripe bytes per row); ``scales``: uint8 ``[N, 32, S]`` or None."""
    payload = np.ascontiguousarray(payload, dtype=np.uint8)
    if payload.ndim != 3 or payload.shape[1] != LANES:
        raise ValueError(f"pack_blm: payload must be [N, 32, P], got {payload.shape}")
    n, _, p_bytes = payload.shape
    s_bytes = 0 if scales is None else int(scales.shape[-1])
    if scales is not None and scales.shape[:2] != (n, LANES):
        raise ValueError(f"pack_blm: scales must be [N, 32, S], got {scales.shape}")
    if layout.scale_placement not in ("inline", "block"):
        raise ValueError(f"pack_blm: scale placement must be 'inline' or 'block', got {layout.scale_placement!r}")
    # the block placement only where it pays: an inline unit without padding (INT4 affine at K = 4096: 64 + 16) or a
    # ragged stripe whose tail half-word holds the scales for free stays inline, and so does a lane whose scales
    # span two words of the region (K = 5120 / 12288 for NVFP4: measured slower on the tile than the padded unit,
    # decode-kernels.md §9) — block placement means one 16-byte scale load per lane-row
    block_scales = (layout.scale_placement == "block" and s_bytes > 0 and _pad16(p_bytes) + s_bytes < _pad16(p_bytes + s_bytes)
                    and max(-(-((lane * s_bytes) % 16 + s_bytes) // 16) for lane in range(LANES)) == 1)
    unit = _pad16(p_bytes) if block_scales else _pad16(p_bytes + s_bytes)
    r = layout.rows
    n_blocks = -(-n // r)
    units = np.zeros((n_blocks * r, LANES, unit), dtype=np.uint8)
    units[:n, :, :p_bytes] = payload
    if scales is not None and not block_scales:
        units[:n, :, p_bytes:p_bytes + s_bytes] = np.ascontiguousarray(scales, dtype=np.uint8)
    blocks = units.reshape(n_blocks, r, LANES, unit)                       # [b, row, lane, U]
    if layout.lane_order == "contiguous":
        data = blocks.transpose(0, 2, 1, 3)                                 # [b, lane, row, U]
    else:
        w = unit // 16
        data = blocks.reshape(n_blocks, r, LANES, w, 16).transpose(0, 1, 3, 2, 4)   # [b, row, word, lane, 16]
    info = PackInfo(format, n, k, r, unit, p_bytes, s_bytes, layout.lane_order, n_blocks, float(tensor_scale),
                    scale_group, "block" if block_scales else "inline")
    payload_region = np.ascontiguousarray(data).reshape(n_blocks, r * LANES * unit)
    if not block_scales:
        return payload_region.tobytes(), info
    sc = np.zeros((n_blocks * r, LANES, s_bytes), dtype=np.uint8)
    sc[:n] = np.ascontiguousarray(scales, dtype=np.uint8)
    region = np.zeros((n_blocks, info.scale_region_bytes), dtype=np.uint8)
    region[:, : r * LANES * s_bytes] = sc.reshape(n_blocks, r * LANES * s_bytes)   # [b][row][lane][S]
    out = np.concatenate([payload_region, region], axis=1)                 # [b, block_bytes]
    assert out.shape[1] == info.block_bytes
    return np.ascontiguousarray(out).tobytes(), info


def unpack_blm(data: bytes, info: PackInfo) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inverse of :func:`pack_blm`: ``(payload [N, 32, P], scales [N, 32, S] or None)``."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size != info.nbytes:
        raise ValueError(f"unpack_blm: expected {info.nbytes} bytes, got {buf.size}")
    r, u = info.rows, info.unit_bytes
    per_block = buf.reshape(info.n_blocks, info.block_bytes)
    pbytes = per_block[:, : r * LANES * u]
    if info.lane_order == "contiguous":
        blocks = pbytes.reshape(info.n_blocks, LANES, r, u).transpose(0, 2, 1, 3)
    else:
        w = u // 16
        blocks = pbytes.reshape(info.n_blocks, r, w, LANES, 16).transpose(0, 1, 3, 2, 4).reshape(info.n_blocks, r, LANES, u)
    units = blocks.reshape(info.n_blocks * r, LANES, u)[: info.n]
    payload = np.ascontiguousarray(units[:, :, : info.payload_bytes])
    scales = None
    if info.scale_bytes and info.scale_placement == "block":
        s = info.scale_bytes
        region = per_block[:, r * LANES * u: r * LANES * u + r * LANES * s]
        scales = np.ascontiguousarray(region.reshape(info.n_blocks * r, LANES, s)[: info.n])
    elif info.scale_bytes:
        scales = np.ascontiguousarray(units[:, :, info.payload_bytes: info.payload_bytes + info.scale_bytes])
    return payload, scales


def split_lanes(row_bytes: np.ndarray, k: int, bytes_per_column_num: int, bytes_per_column_den: int) -> np.ndarray:
    """``[N, B]`` row bytes → ``[N, 32, B/32]`` lane stripes; ``bytes_per_column`` = num/den (½ for nibbles)."""
    n, b = row_bytes.shape
    if k % LANES:
        raise ValueError(f"K={k} must be a multiple of {LANES} lanes")
    per_lane_cols = k // LANES
    per_lane_bytes = per_lane_cols * bytes_per_column_num
    if per_lane_bytes % bytes_per_column_den or b != k * bytes_per_column_num // bytes_per_column_den:
        raise ValueError(f"row bytes {b} do not match K={k}")
    per_lane_bytes //= bytes_per_column_den
    return np.ascontiguousarray(row_bytes.reshape(n, LANES, per_lane_bytes))


def join_lanes(lanes: np.ndarray) -> np.ndarray:
    n, l, per = lanes.shape
    return np.ascontiguousarray(lanes.reshape(n, l * per))
