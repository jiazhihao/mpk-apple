"""Sampling ops (design D7: never leaves the GPU). ``argmax``: ``logits [T, V]`` FP32 → ``token [T]`` i32, a REDUCE
over vocabulary spans combined in block order (deterministic; ties → lowest index). Gumbel-max, top-k/top-p/min-p
arrive with plan M3 as further attrs of the same op family."""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

ARGMAX = register_op(OpDef("argmax", OpClass.REDUCE, "span").bind("*", KernelBinding("argmax")))   # argmax_partial + argmax_final
