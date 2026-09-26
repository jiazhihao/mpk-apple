"""Gated GQA attention with per-head q/k RMSNorm and partial RoPE — stages 1–3 of a full-attention layer: the
``q|gate|k|v`` GEMV (input norm fused; q/k rows in the load-time head-dim permutation so the kernel applies
full-width rotary pairs), the ``gqa_decode`` mixer (norms, RoPE, KV append, online-softmax attention, ``σ(gate)``),
and ``o_proj`` with the residual add.

Semantics follow transformers' ``modeling_qwen3_5.py`` attention (Apache-2.0): ``q_proj`` produces ``[heads, 2, D]``
rows (query | gate per head), q/k norms scale by ``(1 + w)`` over the head dim, RoPE covers the first
``rotary_dim`` dims with ``rotate_half`` pairing, ``scaling = D^-0.5``, output ``attn · σ(gate)``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from ..packs.transforms import rope_head_perm
from .linear import Linear, Part
from .module import LowerContext, Module, StateEntry, WeightSpec


class GQAAttention(Module):
    def __init__(self, hidden: int, heads: int, kv_heads: int, head_dim: int, rotary_dim: int, rope_theta: float,
                 eps: float, *, hf_prefix: str, prefix: str = "", max_context: int, gate: bool = True) -> None:
        super().__init__(prefix=prefix)
        if heads % kv_heads:
            raise ValueError("GQAAttention: heads must be a multiple of kv_heads")
        self.hidden, self.heads, self.kv_heads, self.head_dim = hidden, heads, kv_heads, head_dim
        self.rotary_dim, self.rope_theta, self.eps, self.gate = rotary_dim, rope_theta, eps, gate
        self.max_context = max_context
        self.hf_prefix = hf_prefix
        hd, kd = heads * head_dim, kv_heads * head_dim
        q_rows = heads * (2 if gate else 1) * head_dim
        self.qkv = Linear(hidden, [Part("q_proj", f"{hf_prefix}q_proj.weight", q_rows),
                                   Part("k_proj", f"{hf_prefix}k_proj.weight", kd),
                                   Part("v_proj", f"{hf_prefix}v_proj.weight", kd)],
                          prefix=f"{prefix}qkv.", row_perm=self._row_perm())
        self.o_proj = Linear(hd, [Part("o_proj", f"{hf_prefix}o_proj.weight", hidden)], prefix=f"{prefix}o_proj.",
                             epilogue="residual")

    # ---- layout ---------------------------------------------------------------------------------------------------
    def _row_perm(self) -> np.ndarray:
        """Stacked rows ``[q_proj | k_proj | v_proj]`` → ``[q (head-permuted) | gate | k (head-permuted) | v]``."""
        d, hp = self.head_dim, rope_head_perm(self.head_dim, self.rotary_dim)
        stride = 2 * d if self.gate else d
        q = np.concatenate([h * stride + hp for h in range(self.heads)])
        base = self.heads * stride
        k = base + np.concatenate([h * d + hp for h in range(self.kv_heads)])
        v = base + self.kv_heads * d + np.arange(self.kv_heads * d)
        if self.gate:
            gate = np.concatenate([h * stride + d + np.arange(d) for h in range(self.heads)])
            return np.concatenate([q, gate, k, v])
        return np.concatenate([q, k, v])

    def kernel_segments(self) -> List[tuple]:
        """Row ranges of the packed ``qkv`` slab: ``(name, offset, rows)`` for q, gate, k, v."""
        hd, kd = self.heads * self.head_dim, self.kv_heads * self.head_dim
        segs = [("q", 0, hd)]
        off = hd
        if self.gate:
            segs.append(("gate", off, hd))
            off += hd
        segs += [("k", off, kd), ("v", off + kd, kd)]
        return segs

    def weight_map(self) -> Dict[str, WeightSpec]:
        perm = rope_head_perm(self.head_dim, self.rotary_dim)
        return {"q_norm": WeightSpec(f"{self.hf_prefix}q_norm.weight", (self.head_dim,), "f32", transform="one_plus", aux=True, perm=perm),
                "k_norm": WeightSpec(f"{self.hf_prefix}k_norm.weight", (self.head_dim,), "f32", transform="one_plus", aux=True, perm=perm)}

    def state_entries(self, checkpoints: int = 1) -> List[StateEntry]:
        shape = (self.max_context, self.kv_heads, self.head_dim)
        return [StateEntry(f"{self.prefix}k_cache", shape, DType.BF16), StateEntry(f"{self.prefix}v_cache", shape, DType.BF16)]

    # ---- oracle -----------------------------------------------------------------------------------------------
    def forward(self, x: Any, residual: Any, state: Dict[str, Any], pos: int) -> Any:
        import torch

        from . import oracle

        t = x.shape[0]
        d, hd, kd = self.head_dim, self.heads * self.head_dim, self.kv_heads * self.head_dim
        proj = self.qkv.forward(x)
        q_rows = self.heads * (2 if self.gate else 1) * d
        if self.gate:
            qg = proj[:, :q_rows].reshape(t, self.heads, 2 * d)
            q, gate = qg[..., :d], qg[..., d:]
        else:
            q, gate = proj[:, :q_rows].reshape(t, self.heads, d), None
        k = proj[:, q_rows: q_rows + kd].reshape(t, self.kv_heads, d)
        v = proj[:, q_rows + kd:].reshape(t, self.kv_heads, d)
        q = oracle.rms_norm(q, self.param("q_norm"), self.eps)
        k = oracle.rms_norm(k, self.param("k_norm"), self.eps)
        cos, sin = oracle.rope_tables(self.rope_theta, self.rotary_dim, torch.arange(pos, pos + t))
        cos, sin = cos.to(x.dtype), sin.to(x.dtype)
        q = oracle.apply_partial_rope(q, cos, sin)
        k = oracle.apply_partial_rope(k, cos, sin)
        o = oracle.attention_step(q, k, v, state[f"{self.prefix}k_cache"], state[f"{self.prefix}v_cache"], pos, d ** -0.5)
        if gate is not None:
            o = o * torch.sigmoid(gate.to(torch.float32))
        return self.o_proj.forward(o.to(x.dtype).reshape(t, hd), residual)

    # ---- IR ---------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, h: Value, norm, ctx: LowerContext) -> Value:
        proj = self.qkv.lower(g, h, norm=norm)
        kc, vc = ctx.states[f"{self.prefix}k_cache"], ctx.states[f"{self.prefix}v_cache"]
        cos, sin = ctx.consts["rope_cos"], ctx.consts["rope_sin"]
        qn = self.const_value(g, f"{self.prefix}q_norm", (self.head_dim,), DType.F32)
        kn = self.const_value(g, f"{self.prefix}k_norm", (self.head_dim,), DType.F32)
        o = g.value(f"{self.prefix}attn", (h.shape[0], self.heads * self.head_dim), DType.BF16)
        g.op("gqa_decode", [proj.value, kc, vc, cos, sin, qn, kn], [o], domain=BlockDomain("heads", self.heads),
             klass=OpClass.MAP, updates=[kc.name, vc.name], heads=self.heads, kv_heads=self.kv_heads,
             head_dim=self.head_dim, rotary_dim=self.rotary_dim, eps=self.eps, scaling=self.head_dim ** -0.5,
             gate=self.gate, segments=self.kernel_segments(), rope="permuted")
        return self.o_proj.lower(g, o, residual=h, name=f"{self.prefix}h").value
