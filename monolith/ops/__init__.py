"""The op registry: for each op kind, its class, block domain, cost model and the kernel binding per chip profile.
A missing binding for the target profile fails the build (the coverage guard)."""

from .registry import OPS, CostModel, KernelBinding, OpDef, register_op

__all__ = ["OPS", "CostModel", "KernelBinding", "OpDef", "register_op"]
