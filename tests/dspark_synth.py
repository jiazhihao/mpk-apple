"""A synthetic DSpark drafter + target head for the lowering and GPU tests: a two-layer drafter with head dim 64,
hidden 256 (= heads × head dim, so an identity output projection exposes the attention), two taps, a block of 3,
Markov rank 256 (the packed-embedding gather needs K % 256 == 0) and the confidence head. Every parameter is BF16 in
the synthetic checkpoint, as in the real drafters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import SafetensorsDir, write_safetensors
from monolith.nn import LMHead, Module
from monolith.nn.pack_plan import bind_formats
from monolith.spec.dspark import DSparkConfig, DSparkDrafter

CFG = {"architectures": ["X"], "hidden_size": 256, "intermediate_size": 256, "num_hidden_layers": 2, "num_attention_heads": 4,
       "num_key_value_heads": 2, "head_dim": 64, "rms_norm_eps": 1e-6, "vocab_size": 64, "rope_theta": 10000.0, "block_size": 3,
       "target_layer_ids": [0, 1], "mask_token_id": 63, "markov_rank": 256, "markov_head_type": "vanilla",
       "enable_confidence_head": True, "confidence_head_with_markov": True, "target_hidden_size": 256}
MAX_CONTEXT = 32


class HeadAndDrafter(Module):
    """The two module trees one pack holds: the target's head (the drafter's logits go through it) and the drafter."""

    def __init__(self, head: LMHead, drafter: DSparkDrafter) -> None:
        super().__init__()
        self.lm_head, self.drafter = head, drafter

    def tables(self) -> Dict[str, Tuple[str, Any]]:
        return self.drafter.tables()


def write_checkpoint(path: Path, seed: int = 11) -> Dict[str, np.ndarray]:
    """Writes ``model.safetensors`` + ``config.json``; returns the BF16-valued parameters as float32 arrays."""
    rng = np.random.default_rng(seed)
    c = CFG
    h, inter, d, heads, kv, v, r = c["hidden_size"], c["intermediate_size"], c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"], c["vocab_size"], c["markov_rank"]
    ht = c["target_hidden_size"]

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm(n):
        return (1.0 + rng.standard_normal(n) * 0.1).astype(np.float32)

    raw = {"embed_tokens.weight": w(v, h, scale=0.5), "fc.weight": w(h, len(c["target_layer_ids"]) * ht), "hidden_norm.weight": norm(h),
           "norm.weight": norm(h), "markov_head.markov_w1.weight": w(v, r, scale=0.5), "markov_head.markov_w2.weight": w(v, r, scale=0.2),
           "confidence_head.proj.weight": w(1, h + r, scale=0.1), "confidence_head.proj.bias": w(1, scale=0.5),
           "lm_head.weight": w(v, h, scale=0.3)}
    for i in range(c["num_hidden_layers"]):
        L = f"layers.{i}."
        raw.update({L + "input_layernorm.weight": norm(h), L + "post_attention_layernorm.weight": norm(h),
                    L + "self_attn.q_proj.weight": w(heads * d, h), L + "self_attn.k_proj.weight": w(kv * d, h),
                    L + "self_attn.v_proj.weight": w(kv * d, h), L + "self_attn.o_proj.weight": w(h, heads * d),
                    L + "self_attn.q_norm.weight": norm(d), L + "self_attn.k_norm.weight": norm(d),
                    L + "mlp.gate_proj.weight": w(inter, h), L + "mlp.up_proj.weight": w(inter, h), L + "mlp.down_proj.weight": w(h, inter)})
    write_safetensors(path / "model.safetensors", {k: ("BF16", f32_to_bf16(a)) for k, a in raw.items()}, {"format": "pt"})
    with open(path / "config.json", "w") as f:
        json.dump(CFG, f)
    return {k: bf16_to_f32(f32_to_bf16(a)) for k, a in raw.items()}


def build(path: Path, **drafter_options: Any) -> Tuple[DSparkDrafter, LMHead, DSparkConfig, HeadAndDrafter]:
    """The drafter and the target head with their storage formats bound from the synthetic checkpoint."""
    cfg = DSparkConfig.from_pretrained(str(path))
    head = LMHead(cfg.hidden_size, cfg.vocab_size, hf_name="lm_head.weight", prefix="lm_head.")
    drafter = DSparkDrafter(cfg, target_lm_head=head, max_context=MAX_CONTEXT, **drafter_options)
    pair = HeadAndDrafter(head, drafter)
    ckpt = SafetensorsDir(str(path))
    try:
        for m in (head, drafter):
            bind_formats(m, ckpt)
    finally:
        ckpt.close()
    return drafter, head, cfg, pair
