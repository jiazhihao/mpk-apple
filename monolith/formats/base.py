"""The Format contract. Concrete formats (``nvfp4``, ``fp8_e4m3``, ``bf16``, ``int8``) arrive as separate packages."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class PackLayout:
    """How a weight matrix ``[N, K]`` is cut into blocks for the crew (design D8, §5.5).

    ``rows`` = R rows per block; ``lane_order`` = how the 32 lanes' words are ordered inside a block
    (``contiguous``: lane ℓ's stripe of all R rows is one run; ``interleaved16``: the lanes' k-th 16-byte words are
    adjacent); ``scale_placement`` = where per-block scales go (``inline`` after each lane-row, ``leading`` before
    the block).
    """

    rows: int = 16
    lane_order: str = "interleaved16"
    lanes: int = 32
    scale_placement: str = "inline"

    def __post_init__(self) -> None:
        if self.lane_order not in ("contiguous", "interleaved16"):
            raise ValueError(f"PackLayout.lane_order must be 'contiguous' or 'interleaved16', got {self.lane_order!r}")
        if self.rows <= 0 or self.lanes != 32:
            raise ValueError("PackLayout: rows must be positive and lanes must be 32")


@dataclass
class DequantSpec:
    """The exact recipe that turns checkpoint tensors into the float weight the oracle uses.

    ``tensors`` holds the raw checkpoint arrays by role (``weight``, ``weight_scale``, …); ``shape`` is the logical
    ``[N, K]``; ``params`` carries format constants (block size, scale dtype …). ``Format.dequantize(spec)`` is the
    oracle; ``Format.pack(spec, layout)`` is what the runtime streams.
    """

    format: str
    shape: Tuple[int, int]
    tensors: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)


class Format(ABC):
    """A storage format plugin."""

    name: str = ""
    msl_decode: str = ""      # MSL snippet used by the gemv template: decodes one 16-byte word into weights
    bytes_per_weight: float = 0.0

    @abstractmethod
    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        """Group the checkpoint tensors of one matrix into a :class:`DequantSpec`."""

    @abstractmethod
    def dequantize(self, spec: DequantSpec) -> Any:
        """Exact float32 ``[N, K]`` numpy array — the reference the numerics contract is defined against."""

    @abstractmethod
    def pack(self, spec: DequantSpec, layout: PackLayout) -> bytes:
        """The block-lane-major pack the kernels stream."""
