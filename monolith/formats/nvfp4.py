"""NVFP4 as ModelOpt stores it (``nvidia/Qwen3.8-27B-NVFP4``): E2M1 codes two per byte (low nibble first) in
``weight`` ``U8 [N, K/2]``, an E4M3 block scale per 16 columns in ``weight_scale`` ``F8_E4M3 [N, K/16]``, and one FP32
tensor scale ``weight_scale_2``. Dequantization: ``w = e2m1(code) · e4m3(block_scale) · weight_scale_2``
(the ModelOpt/vLLM convention; ``input_scale`` is an activation-quantization parameter and is ignored — design D11).

Lane-row unit (K = 5120): 80 bytes of nibbles (5 words) + 10 scale bytes, padded to 96; ``SCALE_GROUP = 16``.

Convention checked on the real checkpoint (layer 0 ``down_proj``, 2026-09-24): the stored block-scale codes are all
non-negative and top out at exactly 0x7E (448), i.e. ``weight_scale_2 = amax / (6 · 448)``; the dequantized matrix has
RMS 0.011 and absmax 0.98, while reading the tensor scale as a divisor or dropping it gives 7.9e4 or 29.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

import numpy as np

from .base import DequantSpec, Format, PackLayout
from .blm import PackInfo, join_lanes, pack_blm, split_lanes, unpack_blm
from .fp import (E2M1_MAX, E4M3_MAX, e2m1_to_f32, e4m3_to_f32, f32_to_e2m1, f32_to_e4m3, pack_nibbles,
                 unpack_nibbles)
from .registry import register_format

BLOCK = 16


@register_format("nvfp4")
class NVFP4(Format):
    bytes_per_weight = 0.5 + 1.0 / BLOCK
    weights_per_word = 32
    scale_group = BLOCK
    msl_decode = """
#define WEIGHTS_PER_WORD 32u
#define SCALE_GROUP 16u
static inline float fp4_e2m1(uint q) {
  uint e = (q >> 1) & 3u, m = q & 1u;
  float v = (e == 0u) ? float(m) * 0.5f : as_type<float>(((e + 126u) << 23) | (m << 22));
  return (q & 8u) ? -v : v;
}
static inline void decode_word(uint4 q, thread float* out) {
  uint w[4] = {q.x, q.y, q.z, q.w};
  for (uint e = 0; e < 32; e++) out[e] = fp4_e2m1((w[e >> 3] >> ((e & 7u) * 4u)) & 0xFu);
}
static inline float fp8_e4m3_scale(uint q) {
  uint e = (q >> 3) & 15u, m = q & 7u;
  float v = as_type<float>(((q & 0x7Fu) << 20) + (120u << 23));
  v = (e == 0u) ? float(m) * (1.0f / 512.0f) : v;
  return (q & 0x80u) ? -v : v;
}
// scale of group g of this lane-row: byte g of the unit's scale region (held in registers as uints)
static inline float decode_scale(thread const uint* sw, uint g) { return fp8_e4m3_scale((sw[g >> 2] >> ((g & 3u) * 8u)) & 0xFFu); }
"""

    def unpack(self, tensors: Mapping[str, Any], *, shape: Tuple[int, int]) -> DequantSpec:
        n, k = shape
        w = np.asarray(tensors["weight"], dtype=np.uint8)
        sc = np.asarray(tensors["weight_scale"], dtype=np.uint8)
        s2 = float(np.asarray(tensors["weight_scale_2"], dtype=np.float32).reshape(()))
        if w.shape != (n, k // 2) or sc.shape != (n, k // BLOCK) or k % BLOCK:
            raise ValueError(f"nvfp4: weight {w.shape}, scales {sc.shape} do not match shape {shape}")
        return DequantSpec("nvfp4", (n, k), {"weight": w, "weight_scale": sc}, {"weight_scale_2": s2, "block": BLOCK})

    def dequantize(self, spec: DequantSpec) -> np.ndarray:
        n, k = spec.shape
        codes = unpack_nibbles(spec.tensors["weight"])                                   # [N, K]
        vals = e2m1_to_f32(codes).reshape(n, k // BLOCK, BLOCK)
        scales = e4m3_to_f32(spec.tensors["weight_scale"]).reshape(n, k // BLOCK, 1)
        return (vals * scales * np.float32(spec.params["weight_scale_2"])).reshape(n, k).astype(np.float32)

    def quantize(self, w: np.ndarray) -> DequantSpec:
        w = np.asarray(w, dtype=np.float32)
        n, k = w.shape
        if k % BLOCK:
            raise ValueError("nvfp4: K must be a multiple of 16")
        amax = float(np.abs(w).max()) or 1.0
        s2 = np.float32(amax / (E2M1_MAX * E4M3_MAX))
        blocks = w.reshape(n, k // BLOCK, BLOCK)
        bmax = np.abs(blocks).max(axis=-1)                                                # [N, K/16]
        sc = f32_to_e4m3(np.where(bmax > 0, bmax / E2M1_MAX / s2, 0.0))
        scf = e4m3_to_f32(sc)
        denom = scf[..., None] * s2
        scaled = np.divide(blocks, denom, out=np.zeros_like(blocks), where=denom > 0)
        codes = f32_to_e2m1(scaled.reshape(n, k))
        return DequantSpec("nvfp4", (n, k), {"weight": pack_nibbles(codes), "weight_scale": sc},
                           {"weight_scale_2": float(s2), "block": BLOCK})

    def _lanes(self, spec: DequantSpec) -> Tuple[np.ndarray, np.ndarray]:
        n, k = spec.shape
        payload = split_lanes(spec.tensors["weight"], k, 1, 2)                          # [N, 32, K/64]
        scales = split_lanes(spec.tensors["weight_scale"], k, 1, BLOCK)                 # [N, 32, K/512]
        return payload, scales

    def pack(self, spec: DequantSpec, layout: PackLayout) -> Tuple[bytes, PackInfo]:
        n, k = spec.shape
        if (k // 32) % BLOCK:
            raise ValueError(f"nvfp4: a lane's stripe must hold whole scale groups (K={k})")
        payload, scales = self._lanes(spec)
        return pack_blm(payload, scales, layout, format="nvfp4", k=k, tensor_scale=spec.params["weight_scale_2"],
                        scale_group=BLOCK)

    def unpack_pack(self, data: bytes, info: PackInfo) -> DequantSpec:
        payload, scales = unpack_blm(data, info)
        return DequantSpec("nvfp4", (info.n, info.k), {"weight": join_lanes(payload), "weight_scale": join_lanes(scales)},
                           {"weight_scale_2": info.tensor_scale, "block": BLOCK})
