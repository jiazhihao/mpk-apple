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

    @property
    def block_bytes(self) -> int:
        return self.rows * LANES * self.unit_bytes

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
    if layout.scale_placement != "inline":
        raise NotImplementedError("pack_blm: only inline scale placement is implemented")
    unit = _pad16(p_bytes + s_bytes)
    r = layout.rows
    n_blocks = -(-n // r)
    units = np.zeros((n_blocks * r, LANES, unit), dtype=np.uint8)
    units[:n, :, :p_bytes] = payload
    if scales is not None:
        units[:n, :, p_bytes:p_bytes + s_bytes] = np.ascontiguousarray(scales, dtype=np.uint8)
    blocks = units.reshape(n_blocks, r, LANES, unit)                       # [b, row, lane, U]
    if layout.lane_order == "contiguous":
        data = blocks.transpose(0, 2, 1, 3)                                 # [b, lane, row, U]
    else:
        w = unit // 16
        data = blocks.reshape(n_blocks, r, LANES, w, 16).transpose(0, 1, 3, 2, 4)   # [b, row, word, lane, 16]
    info = PackInfo(format, n, k, r, unit, p_bytes, s_bytes, layout.lane_order, n_blocks, float(tensor_scale),
                    scale_group)
    return np.ascontiguousarray(data).tobytes(), info


def unpack_blm(data: bytes, info: PackInfo) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Inverse of :func:`pack_blm`: ``(payload [N, 32, P], scales [N, 32, S] or None)``."""
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size != info.nbytes:
        raise ValueError(f"unpack_blm: expected {info.nbytes} bytes, got {buf.size}")
    r, u = info.rows, info.unit_bytes
    if info.lane_order == "contiguous":
        blocks = buf.reshape(info.n_blocks, LANES, r, u).transpose(0, 2, 1, 3)
    else:
        w = u // 16
        blocks = buf.reshape(info.n_blocks, r, w, LANES, 16).transpose(0, 1, 3, 2, 4).reshape(info.n_blocks, r, LANES, u)
    units = blocks.reshape(info.n_blocks * r, LANES, u)[: info.n]
    payload = np.ascontiguousarray(units[:, :, : info.payload_bytes])
    scales = None
    if info.scale_bytes:
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
