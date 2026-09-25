"""Speculative decoders. The ``Drafter`` contract is drafter-agnostic; ``spec/dspark/`` is the first plugin
(design D10, §5.8, §5.14)."""

from .drafter import DraftBlock, DraftContext, Drafter
from .registry import DRAFTERS, register_drafter
from . import dspark  # noqa: E402,F401  (registration)

__all__ = ["DraftBlock", "DraftContext", "Drafter", "DRAFTERS", "register_drafter"]
