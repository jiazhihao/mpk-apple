"""Model-agnostic glue between a module tree and the packer / the oracle loader.

* :func:`bind_formats` reads each slab tensor's storage format off the checkpoint (``formats.checkpoint``) and binds
  it on the owning module, so ``weight_map`` / ``slab_groups`` / ``lower`` all see the same format;
* :func:`pack_model` turns the tree into slab, aux and table requests and writes the pack;
* :func:`load_oracle_weights` streams dequantized tensors into the tree for the torch oracles (every floating
  parameter cast to BF16, as the HF reference holds them).

Nothing here knows a model: names, shapes, transforms and permutations all come from the modules' ``weight_map``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..formats import FORMATS, PackLayout
from ..formats.checkpoint import group_tensors, logical_shape
from ..formats.fp import bf16_to_f32
from ..formats.safetensors_reader import SafetensorsDir
from ..packs.packer import AuxRequest, Packer, Segment, SlabRequest, TableRequest
from .module import Model, Module


def _base(hf_name: str) -> str:
    if not hf_name.endswith(".weight"):
        raise ValueError(f"slab tensor {hf_name!r} is not a '<base>.weight' key")
    return hf_name[: -len(".weight")]


def checkpoint_groups(ckpt: SafetensorsDir):
    names = ckpt.names()
    dtypes = {n: ckpt.info(n).dtype for n in names}
    return group_tensors(names, dtypes), dtypes


def bind_formats(model: Module, ckpt: SafetensorsDir) -> Dict[str, str]:
    """Bind every slab tensor's storage format from the checkpoint; returns ``{hf_name: format}``."""
    groups, _ = checkpoint_groups(ckpt)
    bound: Dict[str, str] = {}
    for _, mod in model.named_modules():
        for local, spec in mod.weight_map().items():
            if spec.aux:
                continue
            g = groups.get(_base(spec.hf_name))
            if g is None:
                raise KeyError(f"checkpoint has no tensor group for {spec.hf_name!r}")
            fmt = g.format
            if fmt == "f32":
                fmt = "bf16"                     # a float32 matrix is packed as the BF16 the reference model holds
            if FORMATS.resolve(fmt) is None:
                raise ValueError(f"{spec.hf_name}: storage format {g.format!r} has no format plugin")
            mod.set_format(local, fmt)
            bound[spec.hf_name] = fmt
    return bound


def slab_requests(model: Module, layout: PackLayout) -> List[SlabRequest]:
    out: List[SlabRequest] = []
    seen = set()
    for _, mod in model.named_modules():
        groups = getattr(mod, "slab_groups", None)
        if groups is None:
            continue
        for grp in groups():
            if grp.name in seen:
                raise ValueError(f"slab {grp.name!r} is requested twice")
            seen.add(grp.name)
            segs = [Segment(_base(spec.hf_name)) for _, spec in grp.parts]
            out.append(SlabRequest(grp.name, grp.format, segs, layout, grp.row_perm))
    return out


def aux_requests(model: Module) -> List[AuxRequest]:
    out: List[AuxRequest] = []
    for _, mod in model.named_modules():
        for local, spec in mod.weight_map().items():
            if spec.aux:
                out.append(AuxRequest(f"{mod.prefix}{local}", spec.hf_name, spec.transform, spec.perm))
    return out


def table_requests(model: Model) -> List[TableRequest]:
    return [TableRequest(name, arr, dtype) for name, (dtype, arr) in model.tables().items()]


def pack_model(model: Model, ckpt_dir: str, out_dir: str, layout: PackLayout, *, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Write ``weights.pack`` + ``manifest.json`` for a bound module tree."""
    pk = Packer(ckpt_dir, out_dir)
    for req in slab_requests(model, layout):
        pk.add_slab(req)
    for req in aux_requests(model):
        pk.add_aux(req)
    for req in table_requests(model):
        pk.add_table(req)
    return pk.write(extra)


# ---- oracle loading ---------------------------------------------------------------------------------------------

def dequantized_tensors(model: Module, ckpt: SafetensorsDir) -> Iterator[Tuple[str, np.ndarray, str]]:
    """``(hf_name, float32 array, checkpoint format)`` for every tensor the tree claims, exactly as the pack decoder
    would produce it (quantized groups through their format plugin)."""
    groups, dtypes = checkpoint_groups(ckpt)
    shapes = {n: tuple(ckpt.info(n).shape) for n in ckpt.names()}
    for hf_name, (mod, local, spec) in model.full_weight_map().items():
        g = groups.get(hf_name[: -len(".weight")]) if hf_name.endswith(".weight") else None
        if g is not None and g.format not in ("bf16", "f32", ""):
            tensors = {"weight": ckpt.get(g.weight), **{side: ckpt.get(full) for side, full in g.sides.items()}}
            shape = logical_shape(g, shapes)
            fmt = FORMATS.get(g.format)
            yield hf_name, np.asarray(fmt.dequantize(fmt.unpack(tensors, shape=(int(shape[0]), int(shape[1])))), dtype=np.float32), g.format
            continue
        info = ckpt.info(hf_name)
        arr = ckpt.get(hf_name)
        if info.dtype == "BF16":
            yield hf_name, bf16_to_f32(arr), "bf16"
        elif info.dtype in ("F32", "F16"):
            yield hf_name, np.asarray(arr, dtype=np.float32), "f32"
        else:
            raise ValueError(f"{hf_name}: cannot load dtype {info.dtype} for the oracle")


def load_oracle_weights(model: Module, ckpt_dir: str, *, device: Any = None) -> None:
    """Load the tree's parameters as BF16 torch tensors (the reference model's ``dtype=bfloat16`` semantics)."""
    import torch

    ckpt = SafetensorsDir(ckpt_dir)

    def stream():
        for name, arr, _fmt in dequantized_tensors(model, ckpt):
            yield name, torch.from_numpy(np.array(arr, dtype=np.float32, copy=True)).to(torch.bfloat16).to(device)

    model.load_weights(stream(), strict=True)
    ckpt.close()
