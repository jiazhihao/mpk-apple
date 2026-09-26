"""Checkpoint conventions of the HF ``qwen3_5`` checkpoints: text tensors live under ``model.language_model.``,
``lm_head.weight`` is present only when untied; ``model.visual.*`` (the vision tower) and ``mtp.*`` (the MTP head,
unused: speculation targets a DSpark drafter, design D10) are ignored. Everything else — stacking, permutations,
transforms — is declared by the layers' ``weight_map`` and handled by ``monolith.nn.pack_plan``.

adapted from lithos-ai/mirage python/mirage/mpk/models/qwen38/modeling.py ``load_weights`` @ 5beaed8 (Apache-2.0):
the key routing, without tensor parallelism.
"""

from __future__ import annotations

from typing import Any, Dict

from ...formats.safetensors_reader import SafetensorsDir
from ...nn.pack_plan import bind_formats, load_oracle_weights

TEXT_PREFIX = "model.language_model."
IGNORED_PREFIXES = ("model.visual.", "mtp.")


def is_text_tensor(name: str) -> bool:
    return name == "lm_head.weight" or (name.startswith(TEXT_PREFIX) and not name.startswith(IGNORED_PREFIXES))


def bind_checkpoint_formats(model, ckpt_dir: str) -> Dict[str, str]:
    ckpt = SafetensorsDir(ckpt_dir)
    try:
        return bind_formats(model, ckpt)
    finally:
        ckpt.close()


def load_oracle(model, ckpt_dir: str, *, device: Any = None) -> None:
    """Load the dequantized weights into the module tree for the torch oracle (BF16 parameters)."""
    load_oracle_weights(model, ckpt_dir, device=device)
