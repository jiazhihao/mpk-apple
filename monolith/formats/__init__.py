"""Quantization-format plugins: ``unpack(checkpoint tensors) → exact dequant recipe``, ``pack(...)`` → block-lane-major
pack in the profile's lane order, plus the MSL decode snippet the GEMV template uses (design §5.5)."""

from .base import DequantSpec, Format, PackLayout
from .registry import FORMATS, register_format

__all__ = ["DequantSpec", "Format", "PackLayout", "FORMATS", "register_format"]
