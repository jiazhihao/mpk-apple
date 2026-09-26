"""``Linear``: one or more checkpoint matrices row-stacked into the slabs one input feeds — ``q|gate|k|v``,
``in_proj_qkv|z|a|b``, ``gate|up`` — with the GEMV fusions of design §5.6 as epilogues.

A stacked projection is one ``gemv`` op per *format group*: consecutive parts that share a storage format form one
slab (the 27B's FP8 ``qkv|z`` and BF16 ``a|b`` become two ops that read the same normalized input, un-barriered
siblings). Row permutations (head-dim permutation for RoPE, ``gate/up`` chunk interleaving) apply to a slab's
stacked rows and therefore require a single format group.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.dtypes import DType
from ..core.ir import BlockDomain, Graph, OpClass, Value
from .module import Module, WeightSpec

EPILOGUES = (None, "residual", "silu_mul")


@dataclass(frozen=True)
class Part:
    """One checkpoint matrix ``[rows, K]`` inside a stacked projection."""

    local: str
    hf_name: str
    rows: int


@dataclass
class SlabGroup:
    """A slab the packer writes and a ``gemv`` streams: consecutive parts of one format, with the row permutation."""

    name: str
    format: str
    parts: List[Tuple[str, WeightSpec]]
    row_perm: Optional[np.ndarray]
    rows: int
    k: int

    @property
    def segments(self) -> List[Tuple[str, int]]:
        return [(local, spec.shape[0]) for local, spec in self.parts]


@dataclass
class Projection:
    """The output(s) of a stacked projection: one value per slab group and where each part's columns live."""

    values: List[Value]
    segments: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)   # local -> (value index, column offset, columns)

    @property
    def value(self) -> Value:
        if len(self.values) != 1:
            raise ValueError("projection has several outputs (mixed-format parts); use .values")
        return self.values[0]


class Linear(Module):
    def __init__(self, in_features: int, parts: Sequence[Part], *, prefix: str = "", row_perm: Optional[np.ndarray] = None,
                 epilogue: Optional[str] = None, chunk: Optional[int] = None) -> None:
        super().__init__(prefix=prefix)
        if epilogue not in EPILOGUES:
            raise ValueError(f"Linear: epilogue must be one of {EPILOGUES}, got {epilogue!r}")
        self.k = in_features
        self.parts = tuple(parts)
        self.n = sum(p.rows for p in self.parts)
        self.row_perm = None if row_perm is None else np.asarray(row_perm, dtype=np.int64)
        if self.row_perm is not None and sorted(self.row_perm.tolist()) != list(range(self.n)):
            raise ValueError(f"Linear {prefix}: row_perm is not a permutation of {self.n} rows")
        self.epilogue = epilogue
        self.chunk = chunk
        if epilogue == "silu_mul" and (len(self.parts) != 2 or self.parts[0].rows != self.parts[1].rows or not chunk):
            raise ValueError("Linear: silu_mul needs exactly two equal parts (gate, up) and a chunk size")

    # ---- structure ----------------------------------------------------------------------------------------------
    def slab_groups(self) -> List[SlabGroup]:
        groups: List[SlabGroup] = []
        for p in self.parts:
            fmt = self.format_of(p.local)
            spec = WeightSpec(p.hf_name, (p.rows, self.k), fmt)
            if groups and groups[-1].format == fmt:
                groups[-1].parts.append((p.local, spec))
                groups[-1].rows += p.rows
            else:
                groups.append(SlabGroup("", fmt, [(p.local, spec)], None, p.rows, self.k))
        for grp in groups:
            grp.name = self.prefix + "+".join(local for local, _ in grp.parts)
            grp.parts = [(local, WeightSpec(s.hf_name, s.shape, s.format, slab=grp.name)) for local, s in grp.parts]
        if len(groups) > 1 and self.row_perm is not None:
            raise ValueError(f"Linear {self.prefix}: a row permutation cannot span parts of different formats "
                             f"({[g.format for g in groups]})")
        if groups and self.row_perm is not None:
            groups[0].row_perm = self.row_perm
        return groups

    def weight_map(self) -> Dict[str, WeightSpec]:
        return {local: spec for grp in self.slab_groups() for local, spec in grp.parts}

    # ---- oracle -------------------------------------------------------------------------------------------------
    def forward(self, x: Any, residual: Any = None) -> Any:
        """``[T, K] → [T, N]`` (or ``[T, N/2]`` for ``silu_mul``) with FP32 accumulation and one rounding; the residual
        epilogue adds before that rounding."""
        import torch

        w = torch.cat([self.param(p.local) for p in self.parts], dim=0)
        acc = x.to(torch.float32) @ w.to(torch.float32).t()
        if self.epilogue == "residual":
            if residual is None:
                raise ValueError("Linear with a residual epilogue needs the residual")
            acc = acc + residual.to(torch.float32)
        elif self.epilogue == "silu_mul":
            half = self.n // 2
            acc = torch.nn.functional.silu(acc[:, :half]) * acc[:, half:]
        return acc.to(x.dtype)

    # ---- IR -----------------------------------------------------------------------------------------------------
    def lower(self, g: Graph, x: Value, *, norm=None, residual: Optional[Value] = None, name: Optional[str] = None) -> Projection:
        if (self.epilogue == "residual") != (residual is not None):
            raise ValueError(f"Linear {self.prefix}: residual epilogue and residual input must go together")
        t = x.shape[0]
        proj = Projection([])
        for i, grp in enumerate(self.slab_groups()):
            w = self.weight_value(g, grp.name, (grp.rows, grp.k), grp.format)
            ins = [x, w]
            if norm is not None:
                ins += [norm.stat, norm.weight]
            if residual is not None:
                ins.append(residual)
            n_out = grp.rows // 2 if self.epilogue == "silu_mul" else grp.rows
            out_name = name if (name and len(self.slab_groups()) == 1) else f"{grp.name}.y"
            y = g.value(out_name, (t, n_out), DType.BF16)
            attrs: Dict[str, Any] = dict(norm=norm is not None, epilogue=self.epilogue, segments=grp.segments, format=grp.format)
            if norm is not None:
                attrs["eps"] = norm.eps
            if self.epilogue == "silu_mul":
                attrs["chunk"] = self.chunk
            g.op("gemv", ins, [y], domain=BlockDomain("rows", grp.rows), klass=OpClass.MAP, **attrs)
            proj.values.append(y)
            col = 0
            for local, rows in grp.segments:
                proj.segments[local] = (i, col, rows)
                col += rows
        return proj
