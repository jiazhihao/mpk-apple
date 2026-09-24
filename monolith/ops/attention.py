"""``gqa_decode``: the full-attention mixer for ``T`` new tokens as one op per (q-head, KV-chunk) block.

Inputs ``(qkvg [T, N1], k_cache, v_cache, cos, sin, q_norm_w, k_norm_w)``; output ``[T, H·D]`` BF16; ``updates``
the two caches at ``position .. position+T`` (read from StepState). Attrs: ``heads``, ``kv_heads``, ``head_dim``,
``rotary_dim``, ``eps``, ``scaling``, ``gate`` (multiply by ``σ(gate)``), ``segments`` (row ranges of q | gate | k |
v inside ``qkvg``), ``rope`` (``"permuted"``: q/k rows and the tables are in the load-time head-dim permutation).
The body is the online-softmax merge; q/k norm ``(1 + w)`` and RoPE are applied to the fresh q/k before the append.

Kernel: ``gqa_decode`` writes per-(kv head, chunk, row) partials ``(o, m, d)`` to a workspace sized by
``kernels.gqa_workspace`` and ``gqa_merge`` folds them (two dispatches with a barrier between); macros ``D``,
``CH`` (64), ``RBMAX`` (4) — measured defaults in docs/research/decode-kernels.md §1.
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

GQA_DECODE = register_op(OpDef("gqa_decode", OpClass.MAP, "heads").bind("*", KernelBinding("gqa_decode")))   # + gqa_merge
