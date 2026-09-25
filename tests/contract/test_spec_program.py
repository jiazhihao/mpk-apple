"""The speculative round assembled into the target's dynamic-T step program (design §5.8, #38) without a GPU: the
synthetic hybrid target of ``test_nn_lowering`` + a synthetic DSpark drafter → one program with the verify pass, the
accept scan, the recurrent-state commit pass, the draft pass and the verify select; two packs mapped; the state
slots; the verify rule's two modes."""

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark_synth import build, write_checkpoint  # noqa: E402
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.compiler import compile_program, verify_costs  # noqa: E402
from monolith.compiler.emit import ACCEPT_LOG  # noqa: E402
from monolith.core import Profile, StepStateLayout  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime.program import Program  # noqa: E402

PROF = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
PROF_COST = Profile.from_dict("pc", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16",
                                                                                         "cost_T": {"bf16": {"1": 1.0, "2": 1.1, "4": 1.3, "8": 2.0}}}})


@pytest.fixture
def pair(tmp_path):
    tdir, ddir = tmp_path / "target", tmp_path / "drafter"
    tdir.mkdir(); ddir.mkdir()
    _checkpoint(tdir)
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=16)
    pack_model(model, str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=3)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    return model, drafter, PackFile(tdir / "pack"), PackFile(ddir / "pack")


def test_round_program(pair):
    model, drafter, tp, dp = pair
    prog = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="threshold", verify_threshold=0.5)
    names = [o.name.split(":")[0] for o in prog.ops]
    assert "advance" not in names and names[-1] == "verify_select"
    i_final, i_acc, i_commit, i_tap = names.index("argmax_final"), names.index("accept_scan"), names.index("gdn_commit"), names.index("tap_concat")
    assert i_final < i_acc < i_commit < i_tap and names.count("gdn_commit") == 1 and names.count("gdn_mixer") == 1 and names.count("gdn_norm") == 1
    assert names.count("draft_attn") == 2 and names.count("confidence") == 1 and names.count("accept_scan") == 1
    kern = {o.name: prog.kernels[o.kernel] for o in prog.ops}
    assert kern["gdn_mixer"].macros["SLOTS"] == "2u" and kern["gdn_mixer"].macros["STEP_STATE"] == "1" and "COMMIT" not in kern["gdn_mixer"].macros
    assert kern["gdn_commit"].macros["COMMIT"] == "1" and kern["gdn_commit"].macros["SLOTS"] == "2u"
    ops = {o.name: o for o in prog.ops}
    assert ops["gdn_commit"].bindings[:4] == ops["gdn_mixer"].bindings[:4] and any(b[0] == 15 for b in ops["gdn_commit"].bindings)
    # two slots for the recurrent states, one for the KV caches
    c = model.config
    kd, vd = c.linear_num_key_heads * c.linear_key_head_dim, c.linear_num_value_heads * c.linear_value_head_dim
    assert prog.buffers["layers.0.linear_attn.conv_state"].nbytes == 2 * (2 * kd + vd) * (c.linear_conv_kernel_dim - 1) * 2
    assert prog.buffers["layers.0.linear_attn.rec_state"].nbytes == 2 * c.linear_num_value_heads * 32 * 64 * 4
    assert prog.buffers["layers.1.self_attn.k_cache"].nbytes == 16 * 2 * 32 * 2
    # the accept scan writes the ring and the log; the taps are the embedding and layer 1's residual stream
    acc = ops["accept_scan"]
    assert [b for b in acc.bindings if b[0] == 4] == [(4, ACCEPT_LOG, 0)] and prog.buffers[ACCEPT_LOG].nbytes == 65536 * 4
    assert acc.bindings[0] == (0, "token", 0) and acc.bindings[2][1] == "ring"
    tap = ops["tap_concat"]
    assert tap.bindings[0] == (0, "embed_tokens.h", 0) and tap.bindings[1] == (1, "layers.1.mlp.h", 0)
    # the verify rule: threshold mode with 0.5
    vs = ops["verify_select"]
    prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    gamma, thr, t_max, mode = struct.unpack_from("<IfII", prm)
    assert (gamma, thr, t_max, mode) == (3, 0.5, 8, 0)
    # two mapped packs, distinct entries
    weights = [n for n, b in prog.buffers.items() if b.role == "weights"]
    assert weights == ["pack.0", "pack.1"] and prog.buffers["pack.1"].file.endswith("drafter/pack/weights.pack")
    again = Program.from_json(prog.to_json())
    assert [o.bindings for o in again.ops] == [o.bindings for o in prog.ops]
    # the target's GEMVs come as predicated per-T variants (1, 2, 4, 8 at t_max 8), the drafter's block GEMVs as one
    gate_up = [o for o in prog.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]
    assert [o.meta["t_variant"] for o in gate_up] == [1, 2, 4, 8] and [o.meta["t_range"] for o in gate_up] == [[0, 1], [1, 2], [2, 4], [4, 8]]
    assert [prog.kernels[o.kernel].macros.get("T_HI") for o in gate_up] == ["1", "2", "4", "8"] and prog.kernels[gate_up[0].kernel].macros["T_LO"] == "0"
    assert all(prog.kernels[o.kernel].macros["T"] == str(o.meta["t_variant"]) for o in gate_up)
    fc = [o for o in prog.ops if o.name == "gemv:draft.fc.fc"]
    assert [o.meta["t_variant"] for o in fc] == [1, 2, 4, 8] and prog.kernels[fc[0].kernel].macros["T_SRC"] == "1"
    blk = [o for o in prog.ops if o.name == "gemv:draft.layers.0.self_attn.qkv.q_proj+k_proj+v_proj"]
    assert len(blk) == 1 and "T_HI" not in prog.kernels[blk[0].kernel].macros and blk[0].meta["t_range"] is None
    plain = compile_program(model, tp, PROF, dynamic_t=True)
    assert len([o for o in plain.ops if o.name == "gemv:layers.1.mlp.gate_up.gate_proj+up_proj"]) == 1


def test_verify_costs_and_cost_mode(pair):
    model, drafter, tp, dp = pair
    assert verify_costs(PROF, tp, 3, 8) is None                                  # no bf16 table on this profile
    costs = verify_costs(PROF_COST, tp, 3, 8)
    assert costs == pytest.approx([1.0, 1.1, 1.2, 1.3])                          # T = 1..4, interpolated at 3
    assert verify_costs(PROF_COST, tp, 7, 4) == pytest.approx([1.0, 1.1, 1.2, 1.3])   # clamped to t_max − 1
    prog = compile_program(model, tp, PROF_COST, dynamic_t=True, drafter=drafter, drafter_pack=dp)
    vs = [o for o in prog.ops if o.name == "verify_select"][0]
    prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    gamma, thr, t_max, mode = struct.unpack_from("<IfII", prm)
    assert mode == 1 and struct.unpack_from("<4f", prm, 16) == pytest.approx((1.0, 1.1, 1.2, 1.3))
    # the cost rule on a profile without the table falls back to the threshold rule at 0.5
    prog2 = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp)
    vs2 = [o for o in prog2.ops if o.name == "verify_select"][0]
    assert struct.unpack_from("<IfII", prog2.buffers[[b for b in vs2.bindings if b[0] == 3][0][1]].init)[1:] == (0.5, 8, 0)


def test_fixed_length_sts_and_logs(pair):
    model, drafter, tp, dp = pair
    prog = compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed", verify_length=2)
    vs = [o for o in prog.ops if o.name == "verify_select"][0]
    prm = prog.buffers[[b for b in vs.bindings if b[0] == 3][0][1]].init
    assert struct.unpack_from("<IfII", prm)[1:] == (2.0, 8, 2) and struct.unpack_from("<I", prm, 16 + 64)[0] == 65536
    assert any(b[0] == 4 and b[1] == "conf_log" for b in vs.bindings) and prog.buffers["conf_log"].nbytes == 65536 * 16 * 4
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed")
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, verify="fixed", verify_length=9)
    # STS temperatures reach the confidence kernel's params
    from dspark_synth import build as _build

    d2, _, _, _ = _build(pair_dir(tp), target_lm_head=model.lm_head, sts=[0.5, 1.0, 2.0])
    prog2 = compile_program(model, tp, PROF, dynamic_t=True, drafter=d2, drafter_pack=dp)
    cf = [o for o in prog2.ops if o.name == "confidence"][0]
    assert struct.unpack_from("<3f", prog2.buffers[[b for b in cf.bindings if b[0] == 5][0][1]].init, 16) == (0.5, 1.0, 2.0)
    with pytest.raises(ValueError):
        _build(pair_dir(tp), target_lm_head=model.lm_head, sts=[1.0])


def pair_dir(tp):
    return tp.dir.parent.parent / "drafter"


def test_round_needs_the_dynamic_program_and_a_fitting_layout(pair):
    model, drafter, tp, dp = pair
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, t=1, drafter=drafter, drafter_pack=dp)
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter, drafter_pack=dp, layout=StepStateLayout(t_max=3, gamma_max=2))
    with pytest.raises(ValueError):
        compile_program(model, tp, PROF, dynamic_t=True, drafter=drafter)
    # a plain program still ends with the advance and keeps two state slots
    plain = compile_program(model, tp, PROF, dynamic_t=True)
    names = [o.name.split(":")[0] for o in plain.ops]
    assert names[-1] == "advance" and "accept_scan" not in names and "gdn_commit" not in names
    assert plain.buffers["layers.0.linear_attn.rec_state"].nbytes == 2 * model.config.linear_num_value_heads * 32 * 64 * 4
    static = compile_program(model, tp, PROF, t=1)
    gdn = [o for o in static.ops if o.name == "gdn_mixer"][0]
    assert static.kernels[gdn.kernel].macros["SLOTS"] == "2u" and any(b[0] == 15 for b in gdn.bindings)
