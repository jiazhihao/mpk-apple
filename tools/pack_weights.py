#!/usr/bin/env python3
"""Pack a checkpoint from a plan file (until model packages generate plans from their weight maps, plan M4):

    python tools/pack_weights.py --model <ckpt dir> --out <pack dir> --plan plan.json [--lane-order interleaved16 --rows 16]

plan.json:
  {"slabs": [{"name": "l0.qkv", "format": "fp8_e4m3", "segments": ["…q_proj", "…k_proj", "…v_proj"],
              "row_perm": {"kind": "interleave_chunks", "n_a": 3584, "n_b": 3584, "chunk": 8}}, …],
   "aux":   [{"name": "l0.input_norm", "source": "…input_layernorm.weight", "transform": "one_plus"}, …]}
A segment is a source base name or {"source": …, "rows": [start, stop]}; row_perm kinds: interleave_chunks,
head_dim_perm (n_heads, head_dim, rotary_dim, stride, offset).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monolith.formats import PackLayout                                  # noqa: E402
from monolith.packs import AuxRequest, Packer, Segment, SlabRequest, head_dim_perm, interleave_chunks   # noqa: E402


def _perm(d):
    if d is None:
        return None
    kind = d["kind"]
    if kind == "interleave_chunks":
        return interleave_chunks(d["n_a"], d["n_b"], d["chunk"])
    if kind == "head_dim_perm":
        return head_dim_perm(d["n_heads"], d["head_dim"], d["rotary_dim"], stride=d.get("stride"), offset=d.get("offset", 0))
    raise ValueError(f"unknown row_perm kind {kind!r}")


def _segment(s):
    if isinstance(s, str):
        return Segment(s)
    rows = s.get("rows")
    return Segment(s["source"], None if rows is None else np.arange(rows[0], rows[1]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--lane-order", default="interleaved16", choices=["contiguous", "interleaved16"])
    ap.add_argument("--rows", type=int, default=16)
    a = ap.parse_args(argv)
    with open(a.plan) as f:
        plan = json.load(f)
    layout = PackLayout(rows=a.rows, lane_order=a.lane_order)
    pk = Packer(a.model, a.out)
    for s in plan.get("slabs", []):
        pk.add_slab(SlabRequest(s["name"], s["format"], [_segment(x) for x in s["segments"]], layout, _perm(s.get("row_perm"))))
    for x in plan.get("aux", []):
        pk.add_aux(AuxRequest(x["name"], x["source"], x.get("transform")))
    m = pk.write({"plan": str(a.plan), "layout": {"rows": a.rows, "lane_order": a.lane_order}})
    print(f"packed {len(m['slabs'])} slabs and {len(m['aux'])} aux tensors, {m['nbytes'] / 1e6:.1f} MB -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
