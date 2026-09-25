"""``gdn_mixer``: the Gated-DeltaNet mixer for ``T`` tokens as one op per value head.

Inputs ``(proj [T, N1], conv_state, rec_state, conv_w, neg_exp_a_log, dt_bias, norm_w)``; output ``[T, Hv·dv]``
BF16; ``updates`` both states (checkpoint slot chosen by the accept scan). Attrs: ``k_heads``, ``v_heads``,
``dk``, ``dv``, ``conv_width``, ``eps``, ``segments`` (row ranges of q | k | v | z | a | b inside ``proj``).
Body order (the reference's): conv + SiLU over ``q|k|v`` → L2-norm q, k → ``β = σ(b)``, ``g = −exp(A_log)·softplus(a
+ dt_bias)`` (FP32) → delta-rule update and read-out → gated RMSNorm ``norm(o)·w·silu(z)``.

Kernel: ``gdn_mixer`` (one block per value head; the recurrence runs in column slices with the slice's state in
registers, ``TP`` tokens per pass); macros ``DK``, ``DV``, ``CW``, ``SL``, ``TP`` — measured defaults in
docs/research/decode-kernels.md §2. The a|b coefficients may come from a second projection value (mixed formats).
The states have two slots by step parity (the step's pass reads one and writes the other). ``gdn_commit`` — the same
inputs and attrs, emitted by the speculative program after the accept scan (attr ``commit_kind`` of the mixer op) —
recomputes the recurrence for the committed positions (``StepState.n_inject``) so the rejected ones never reach the
state (design §5.8).
"""

from ..core.ir import OpClass
from .registry import KernelBinding, OpDef, register_op

GDN_MIXER = register_op(OpDef("gdn_mixer", OpClass.MAP, "heads").bind("*", KernelBinding("gdn_mixer")))
GDN_COMMIT = register_op(OpDef("gdn_commit", OpClass.MAP, "heads").bind("*", KernelBinding("gdn_mixer", {"COMMIT": 1})))
