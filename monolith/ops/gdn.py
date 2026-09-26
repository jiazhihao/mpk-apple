"""``gdn_mixer``: the Gated-DeltaNet mixer for ``T`` tokens as one op per value head.

Inputs ``(proj [T, N1], conv_state, rec_state, conv_w, neg_exp_a_log, dt_bias, norm_w)``; output ``[T, Hv·dv]``
BF16; ``updates`` both states (checkpoint slot chosen by the accept scan). Attrs: ``k_heads``, ``v_heads``,
``dk``, ``dv``, ``conv_width``, ``eps``, ``segments`` (row ranges of q | k | v | z | a | b inside ``proj``).
Body order (the reference's): conv + SiLU over ``q|k|v`` → L2-norm q, k → ``β = σ(b)``, ``g = −exp(A_log)·softplus(a
+ dt_bias)`` (FP32) → delta-rule update and read-out → gated RMSNorm ``norm(o)·w·silu(z)``.
"""

from ..core.ir import OpClass
from .registry import OpDef, register_op

GDN_MIXER = register_op(OpDef("gdn_mixer", OpClass.MAP, "heads"))
