"""The op registry: for each op kind, its class, block domain, cost model and the kernel binding per chip profile.
A missing binding for the target profile fails the build (the coverage guard). Importing this package registers the
decode op kinds the layer library lowers to (``embed``, ``rmsnorm_stat``, ``gemv``, ``lm_head``, ``gqa_decode``,
``gdn_mixer``, ``argmax``); kernels bind to them per profile as they land (plan M3)."""

from .registry import OPS, CostModel, KernelBinding, OpDef, register_op
from . import attention, draft, embed, gdn, gemv, norm, sample  # noqa: F401  (registration)

__all__ = ["OPS", "CostModel", "KernelBinding", "OpDef", "register_op"]
