"""``rmsnorm_stat``: per-token ``r = rsqrt(mean(h²) + eps)`` of ``h [T, H]`` → ``stat [T]`` FP32 (attr ``eps``).
The fuse pass hoists it into the producing op's epilogue where it can (design §5.1); the scaling itself
(``x = h · r · (1 + w)``) is always applied inside the consuming GEMV (attr ``norm`` on ``gemv``)."""

from ..core.ir import OpClass
from .registry import OpDef, register_op

RMSNORM_STAT = register_op(OpDef("rmsnorm_stat", OpClass.REDUCE, "span"))
