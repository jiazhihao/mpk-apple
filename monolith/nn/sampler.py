"""Sampling on the GPU (design D7). ``GreedySampler``: argmax of the BF16 logits widened to FP32, ties → lowest
index, one token per step position. Gumbel-max / top-k / top-p / min-p arrive with plan M3 as further samplers."""

from __future__ import annotations

from typing import Any

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .module import LowerContext, Module


class GreedySampler(Module):
    def forward(self, logits: Any) -> Any:
        import torch

        return torch.argmax(logits.to(torch.float32), dim=-1)

    def lower(self, g: Graph, logits: Value, ctx: LowerContext) -> Value:
        token = g.value("token", (logits.shape[0],), DType.I32)
        g.op("argmax", [logits], [token], domain=BlockDomain("span", logits.shape[1]), klass=OpClass.REDUCE)
        return token
