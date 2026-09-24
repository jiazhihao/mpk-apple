"""The DSpark round's op kinds (design §5.6, §5.8; issue #24). Registered without kernel bindings until the
kernels land, so a program with a drafter fails the coverage guard until then — by design.

* ``feature_proj``: ``fc`` over the concatenated tapped residual streams of the ``n_new`` verified positions
  → ``hidden_norm`` → the drafter's context features ``[n_new, H_draft]`` (a GEMV with a fused standard RMSNorm
  epilogue; ``n_new`` read from StepState.accepted + 1).
* ``draft_attn``: the drafter's attention for the block: keys = its context KV cache (positions < start) ∪ the
  ``n_new`` context positions (appended from the features' k/v projection) ∪ the block's own k/v; queries = the γ
  block positions; no mask (bidirectional inside the block, full context); q/k norm and RoPE as the target's.
* ``markov_bias``: ``logits_k += W₂ · W₁[prev_k]`` and the argmax → ``draft_k`` (γ serial steps: ``prev_{k+1} = draft_k``).
* ``confidence``: ``σ(w · [h_k ; W₁[prev_k]] + b)`` per block position.
* ``verify_select``: SERIAL — the verify length ``L`` from the confidences (the reference's threshold rule, or the
  profile's cost table); writes ``StepState.verify_len`` / ``t_this_step`` / ``pending_tokens``.
* ``accept_scan``: SERIAL — compares the target's sampled tokens with the drafts, commits the accepted prefix and
  the bonus token to the ring, advances position / kv_len / anchor, chooses the state checkpoint.
"""

from ..core.ir import OpClass
from .registry import OpDef, register_op

FEATURE_PROJ = register_op(OpDef("feature_proj", OpClass.MAP, "rows"))
DRAFT_ATTN = register_op(OpDef("draft_attn", OpClass.MAP, "heads"))
MARKOV_BIAS = register_op(OpDef("markov_bias", OpClass.SERIAL, "span"))
CONFIDENCE = register_op(OpDef("confidence", OpClass.SERIAL, "span"))
VERIFY_SELECT = register_op(OpDef("verify_select", OpClass.SERIAL, "span"))
ACCEPT_SCAN = register_op(OpDef("accept_scan", OpClass.SERIAL, "span"))
