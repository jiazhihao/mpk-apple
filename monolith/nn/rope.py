"""RoPE tables for the pack (numpy, torch-free). Two layouts:

* ``hf``: ``[max_pos, rotary_dim]`` ``cat(freqs, freqs)`` — what the reference ``rotate_half`` consumes;
* ``permuted``: ``[max_pos, head_dim]`` in the load-time head-dim permutation (``packs.transforms.rope_head_perm``):
  the rotary frequencies sit at ``[0, rotary/2)`` and ``[D/2, D/2 + rotary/2)``, every other slot is the identity
  (cos = 1, sin = 0), so the kernel applies full-width ``(i, i + D/2)`` pairs and partial RoPE costs nothing.

adapted from lithos-ai/mirage python/mirage/mpk/models/qwen38/modeling.py ``build_rope_tables`` @ 5beaed8 (Apache-2.0).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def inv_freq(theta: float, rotary_dim: int) -> np.ndarray:
    return (1.0 / (theta ** (np.arange(0, rotary_dim, 2, dtype=np.float32) / rotary_dim))).astype(np.float32)


def rope_tables_hf(theta: float, rotary_dim: int, max_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.arange(max_pos, dtype=np.float32)
    freqs = np.outer(pos, inv_freq(theta, rotary_dim)).astype(np.float32)
    emb = np.concatenate([freqs, freqs], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rope_tables_permuted(theta: float, head_dim: int, rotary_dim: int, max_pos: int) -> Tuple[np.ndarray, np.ndarray]:
    half = rotary_dim // 2
    pos = np.arange(max_pos, dtype=np.float32)
    freqs = np.outer(pos, inv_freq(theta, rotary_dim)).astype(np.float32)      # [P, half]
    full = np.zeros((max_pos, head_dim), dtype=np.float32)
    full[:, :half] = freqs
    full[:, head_dim // 2: head_dim // 2 + half] = freqs
    return np.cos(full).astype(np.float32), np.sin(full).astype(np.float32)
