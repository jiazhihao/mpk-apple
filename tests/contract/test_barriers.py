"""The barrier pass (design §5.1, §5.12; #29) on a hand-built program: an op waits (its barrier) only where it reads
what the open group wrote or writes what it read / wrote, buffer granularity; the predicated per-T variants of one
GEMV join without a check; the first op always waits; ``all`` restores v0. And the sibling order on a ``bus_first``
profile."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.compiler import compile_program, place_barriers  # noqa: E402
from monolith.core import Profile  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime.program import KernelSpec, OpSpec, Program  # noqa: E402


def _op(name, bindings, writes=None, group=None):
    meta = {}
    if writes is not None:
        meta["writes"] = writes
    if group is not None:
        meta["variant_group"] = group
    return OpSpec("k", [(i, b, 0) for i, b in enumerate(bindings)], (1, 1, 1), (32, 1, 1), True, [], name, meta)


def test_pass_on_a_hand_built_program():
    ops = [
        _op("a", ["w", "x", "y1"], writes=[2]),                # writes y1
        _op("b", ["w2", "x", "y2"], writes=[2]),               # independent of a: no barrier after a
        _op("c", ["y1", "y2", "z"], writes=[2]),               # reads y1, y2: barrier after b
        _op("v1", ["w3", "z", "u"], writes=[2], group=7),      # variants of one GEMV: no barriers among them
        _op("v2", ["w3", "z", "u"], writes=[2], group=7),
        _op("v4", ["w3", "z", "u"], writes=[2], group=7),
        _op("d", ["u", "q"], writes=[1]),                      # reads u: barrier after the variants
        _op("e", ["q", "x"]),                                  # no writes record: treated as writing everything → barrier after d
        _op("f", ["r", "s"], writes=[1]),                      # independent of e's (x, q): but e wrote q and x … f reads r: no hazard
    ]
    prog = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=ops)
    n = place_barriers(prog)
    flags = [o.barrier_before for o in prog.ops]
    assert flags == [True, False, True, True, False, False, True, True, False] and n == 5
    # the one-op look-back: c conflicts with {a, b} through a's y1 but not with b — b waits for a instead and c runs
    # beside b (a gate GEMV encoded before its core)
    ops2 = [_op("a", ["w", "x", "y1"], writes=[2]), _op("b", ["w2", "x", "y2"], writes=[2]), _op("c", ["y1", "z"], writes=[1])]
    prog2 = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=ops2)
    place_barriers(prog2)
    assert [o.barrier_before for o in prog2.ops] == [True, True, False]
    assert place_barriers(prog, "all") == 9 and all(o.barrier_before for o in prog.ops)
    with pytest.raises(ValueError):
        place_barriers(prog, "some")
    # a write-after-read hazard: g reads x, h writes x
    prog2 = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=[_op("g", ["x", "o1"], writes=[1]), _op("h", ["x2", "x"], writes=[1])])
    place_barriers(prog2)
    assert prog2.ops[0].barrier_before and prog2.ops[1].barrier_before
    # states updated in place: the same buffer read and written by consecutive ops
    prog3 = Program(kernels={"k": KernelSpec("", "k")}, buffers={}, ops=[_op("g", ["s"], writes=[0]), _op("h", ["s", "o"], writes=[1])])
    place_barriers(prog3)
    assert prog3.ops[1].barrier_before


def test_sibling_order_follows_the_profile(tmp_path):
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    base = {"gpu_cores": 20, "nominal_gbps": 307.0}
    alu = Profile.from_dict("alu", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "sibling_order": "alu_first"}})
    bus = Profile.from_dict("bus", {**base, "engine": {"family": "Apple9", "lane_order": "interleaved16", "sibling_order": "bus_first"}})
    pa = compile_program(m, PackFile(tmp_path / "pack"), alu, t=1)
    pb = compile_program(m, PackFile(tmp_path / "pack"), bus, t=1)
    ka, kb = [o.name.split(":")[0] for o in pa.ops], [o.name.split(":")[0] for o in pb.ops]
    ia, ib = ka.index("gdn_mixer"), kb.index("gdn_mixer")
    assert ka[ia + 1] == "gemv" and pa.ops[ia + 1].meta["sibling"] and not pa.ops[ia + 1].barrier_before   # core first: the gate runs beside it
    assert kb[ib - 1] == "gemv" and pb.ops[ib - 1].meta["sibling"] and not pb.ops[ib].barrier_before        # gate first: the core runs beside it
    assert pb.ops[ib + 1].barrier_before and kb[ib + 1] == "gdn_norm" and pa.ops[ia + 2].barrier_before
    ja, jb = ka.index("gqa_decode"), kb.index("gqa_decode")
    assert ka[ja + 1] == "gemv" and kb[jb - 1] == "gemv" and kb[jb + 1] == "gqa_merge"
    assert sorted(ka) == sorted(kb)


def test_attention_kernel_follows_the_profile(tmp_path):
    """``engine.attention = "v2"`` (or the override) emits gqa_decode_v2 / gqa_merge_v2 with the v2 workspace
    (32-key chunks, the threadgroup count in ``n_sg``); v1 stays the default; the partial workspaces are shared."""
    from monolith.compiler.emit import _gqa_v2  # noqa: F401  (the switch exists)

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    pack_model(m, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16))
    base = {"gpu_cores": 20, "nominal_gbps": 307.0}
    p1 = Profile.from_dict("a", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    p2 = Profile.from_dict("b", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "attention": "v2"}})
    assert p1.attention == "v1" and p2.attention == "v2"
    with pytest.raises(ValueError):
        Profile.from_dict("c", {**base, "engine": {"family": "Apple10", "lane_order": "interleaved16", "attention": "v3"}})
    prog1 = compile_program(m, PackFile(tmp_path / "pack"), p1, t=2)
    prog2 = compile_program(m, PackFile(tmp_path / "pack"), p2, t=2)
    prog3 = compile_program(m, PackFile(tmp_path / "pack"), p1, t=2, attention="v2")
    for prog, fn in ((prog1, "gqa_decode"), (prog2, "gqa_decode_v2"), (prog3, "gqa_decode_v2")):
        core = [o for o in prog.ops if o.name == "gqa_decode"][0]
        assert prog.kernels[core.kernel].function == fn and core.meta["attention"] == ("v2" if fn.endswith("v2") else "v1")
        merge = [o for o in prog.ops if o.name == "gqa_merge"][0]
        assert prog.kernels[merge.kernel].function == fn.replace("decode", "merge")
    k2 = prog2.kernels[[o for o in prog2.ops if o.name == "gqa_decode"][0].kernel]
    assert k2.macros["RMAX"] == f"{4 * 2}u" and k2.macros["RG"] == "4u" and "CH" not in k2.macros
    import struct

    prm = prog2.buffers[[b for b in [o for o in prog2.ops if o.name == "gqa_decode"][0].bindings if b[0] == 9][0][1]].init
    heads, kv, t_active, position, n_sg = struct.unpack_from("<IIIII", prm)
    assert (heads, kv, t_active, n_sg) == (8, 2, 2, 20)                     # v2: n_sg carries the threadgroup count
    n_chunks_max = struct.unpack_from("<I", prm, 60)[0]
    assert n_chunks_max == 16 // 32 + 1 or n_chunks_max == 1
    # the argmax partials share one workspace across programs' ops (one name → one buffer)
    ws = [n for n in prog1.buffers if n.startswith("ws.") and n.endswith(".shared")]
    assert any("argmax.val" in n for n in ws)

