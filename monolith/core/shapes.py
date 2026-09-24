"""Shapes symbolic in the per-step token count ``T`` and the context length ``ctx``.

The step program is compiled once for ``T_max`` and replayed with the actual ``T`` read from ``StepState``
(design §5.7), so a shape may carry a symbol; the memory planner binds it to its maximum, kernels bind it at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple, Union


@dataclass(frozen=True)
class Sym:
    """A named symbolic dimension."""

    name: str

    def __repr__(self) -> str:
        return self.name


Dim = Union[int, Sym]
Shape = Tuple[Dim, ...]

T = Sym("T")        # tokens in this step: 1 for plain decode, 1 + L when verifying L drafts, ≤ T_max
CTX = Sym("ctx")    # committed context length (KV length)


def is_static(shape: Iterable[Dim]) -> bool:
    return all(isinstance(d, int) for d in shape)


def bind(shape: Iterable[Dim], bindings: Mapping[Sym, int]) -> Tuple[int, ...]:
    """Replace every symbol by its binding; raises ``KeyError`` for an unbound symbol."""
    out = []
    for d in shape:
        if isinstance(d, Sym):
            if d not in bindings:
                raise KeyError(f"unbound symbolic dimension {d.name}")
            out.append(int(bindings[d]))
        else:
            out.append(int(d))
    return tuple(out)


def numel(shape: Iterable[Dim], bindings: Mapping[Sym, int] | None = None) -> int:
    n = 1
    for d in bind(shape, bindings or {}):
        n *= d
    return n
