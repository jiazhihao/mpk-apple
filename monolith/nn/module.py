"""The Module contract (design §5.14): ``forward()`` is the torch oracle, ``lower()`` emits IR, ``weight_map()``
names the checkpoint tensors a module consumes. Models add ``layers()``, ``state_spec()`` and ``feature_taps()``.

The contract is deliberately torch-free at import time: the runtime path (pack → program → replay) never needs torch;
oracles import it inside ``forward``.

adapted from lithos-ai/mirage python/mirage/mpk/layers_v2/_base.py @ 5beaed8 (Apache-2.0): the three-method contract
and the streaming ``load_weights`` routing (longest matching module prefix); the CUDA-specific ``compile`` /
``auto_grid_dim`` methods were not taken.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from ..core.dtypes import DType
from ..core.ir import Graph, Value


@dataclass(frozen=True)
class WeightSpec:
    """One checkpoint tensor a module consumes.

    ``hf_name`` is the full checkpoint key; ``format`` the storage-format plugin (``bf16``, ``fp8_e4m3``, ``nvfp4``,
    ``int8``); ``transform`` an optional named load-time transform (row-stacking, head permutation, ``1+w`` …)
    applied by the packer; ``slab`` the pack slab this tensor is placed in (defaults to the module prefix).
    """

    hf_name: str
    shape: Tuple[int, ...]
    format: str
    transform: Optional[str] = None
    slab: Optional[str] = None


@dataclass(frozen=True)
class StateEntry:
    """A persistent per-sequence state buffer the runtime allocates (KV cache, recurrent state, conv state)."""

    name: str
    shape: Tuple[Any, ...]
    dtype: DType
    checkpoints: int = 1        # slots for speculative rollback (γ + 1 for states a verify pass advances)


@dataclass(frozen=True)
class StateSpec:
    entries: Tuple[StateEntry, ...] = ()


class Module:
    """Base of every layer, model and drafter.

    Subclasses implement:
      * ``forward(*xs)`` — the eager torch reference, the semantics (imports torch lazily);
      * ``lower(g, *xs)`` — emit IR ops into ``g`` and return the output ``Value`` (or a tuple); never touches Metal;
      * ``weight_map()`` — ``{local name: WeightSpec}`` for the tensors this module itself owns (not its children's).

    Children are discovered from attributes that are ``Module``s or lists/dicts of them, so a composite module is
    written the natural way (``self.mlp = GatedMLP(...)``).
    """

    def __init__(self, *, prefix: str = "") -> None:
        self.prefix = prefix
        self._params: Dict[str, Any] = {}

    # ---- the contract ----------------------------------------------------------------------------------------
    def forward(self, *xs: Any) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.forward (the torch oracle) is not implemented")

    def lower(self, g: Graph, *xs: Value) -> Any:
        raise NotImplementedError(f"{type(self).__name__}.lower (IR emission) is not implemented")

    def weight_map(self) -> Dict[str, WeightSpec]:
        return {}

    # ---- structure ------------------------------------------------------------------------------------------
    def named_children(self) -> Iterator[Tuple[str, "Module"]]:
        for name, attr in vars(self).items():
            if name.startswith("_"):
                continue
            if isinstance(attr, Module):
                yield name, attr
            elif isinstance(attr, (list, tuple)):
                for i, m in enumerate(attr):
                    if isinstance(m, Module):
                        yield f"{name}.{i}", m
            elif isinstance(attr, dict):
                for k, m in attr.items():
                    if isinstance(m, Module):
                        yield f"{name}.{k}", m

    def named_modules(self, prefix: str = "") -> Iterator[Tuple[str, "Module"]]:
        yield prefix, self
        for name, child in self.named_children():
            yield from child.named_modules(f"{prefix}.{name}" if prefix else name)

    def full_weight_map(self) -> Dict[str, Tuple["Module", str, WeightSpec]]:
        """``{hf_name: (owning module, local name, spec)}`` over the whole subtree; duplicate keys are an error."""
        out: Dict[str, Tuple[Module, str, WeightSpec]] = {}
        for _, mod in self.named_modules():
            for local, spec in mod.weight_map().items():
                if spec.hf_name in out:
                    raise ValueError(f"{spec.hf_name!r} is claimed by two modules")
                out[spec.hf_name] = (mod, local, spec)
        return out

    # ---- weights --------------------------------------------------------------------------------------------
    def param(self, local: str) -> Any:
        try:
            return self._params[local]
        except KeyError:
            raise KeyError(f"{type(self).__name__}: weight {local!r} is not loaded") from None

    def set_param(self, local: str, tensor: Any) -> None:
        self._params[local] = tensor

    def load_weights(self, weights: Iterable[Tuple[str, Any]], *, strict: bool = True) -> Set[str]:
        """Streaming load: route each ``(hf_name, tensor)`` to the module whose ``weight_map`` claims it.

        Returns the set of consumed keys. Unknown keys raise (``strict``) or are ignored; a claimed key that never
        arrives raises after the stream ends. ``process_weights()`` then runs post-load transforms bottom-up.
        """
        table = self.full_weight_map()
        consumed: Set[str] = set()
        for name, tensor in weights:
            hit = table.get(name)
            if hit is None:
                if strict:
                    raise KeyError(f"{type(self).__name__}.load_weights: unexpected checkpoint key {name!r}")
                continue
            mod, local, _spec = hit
            mod.set_param(local, tensor)
            consumed.add(name)
        missing = sorted(set(table) - consumed)
        if missing:
            raise ValueError(f"{type(self).__name__}.load_weights: weights never loaded: {missing[:8]}"
                             + (" …" if len(missing) > 8 else ""))
        self.process_weights()
        return consumed

    def process_weights(self) -> None:
        """Post-load hook, bottom-up (children first). Override for transforms that need several tensors."""
        for _, child in self.named_children():
            child.process_weights()


class Model(Module):
    """A registered architecture: the module tree plus what the runtime must allocate around it."""

    config: Any = None

    def layers(self) -> Sequence[Module]:
        raise NotImplementedError

    def state_spec(self) -> StateSpec:
        raise NotImplementedError

    def feature_taps(self) -> List[int]:
        """Layers whose residual stream a drafter may read (empty if the model exposes none)."""
        return []
