"""The DSpark draft pass as an emitted dynamic-T program on the GPU vs the drafter's torch oracle (#24): a synthetic
drafter + target head packed from a synthetic checkpoint, two steps (an injection into an empty context, then a
second injection with a new anchor), comparing the context features, the block hidden, the base and corrected
logits, the drafts, the confidences, the context caches and the StepState bookkeeping of verify_select."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dspark_synth import build, write_checkpoint  # noqa: E402

from monolith.compiler import emit_program  # noqa: E402
from monolith.compiler.passes import DEFAULT_PASSES  # noqa: E402
from monolith.core import DType, Graph, T  # noqa: E402
from monolith.formats import PackLayout  # noqa: E402
from monolith.formats.fp import bf16_to_f32, f32_to_bf16  # noqa: E402
from monolith.nn.pack_plan import load_oracle_weights, pack_model  # noqa: E402
from monolith.packs import PackFile  # noqa: E402
from monolith.runtime import Engine  # noqa: E402
from monolith.runtime import _native as nt  # noqa: E402
from monolith.spec import DraftContext  # noqa: E402
from monolith.spec.dspark.weights import load_oracle  # noqa: E402


def _bars(got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    return float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref) + 1e-30)), float(np.abs(got - ref).max()), float(np.abs(ref).max())


def _bf16(buf, shape):
    return bf16_to_f32(np.frombuffer(buf, dtype=np.uint16)).reshape(shape)


@pytest.mark.parametrize("threshold", [0.0, 0.5])
def test_draft_program_matches_oracle(tmp_path, threshold):
    torch = pytest.importorskip("torch")
    from monolith.bench import profile_for_device

    dev = nt.Device()
    info = dev.info()
    prof = profile_for_device(info.gpu_cores, info.apple_family)
    if prof is None:
        pytest.skip("no chip profile for this device")
    write_checkpoint(tmp_path)
    drafter, head, cfg, pair = build(tmp_path, confidence_threshold=threshold)
    pack_model(pair, str(tmp_path), str(tmp_path / "pack"), PackLayout(rows=16, lane_order=prof.lane_order))
    pf = PackFile(tmp_path / "pack")
    load_oracle(drafter, str(tmp_path))
    load_oracle_weights(head, str(tmp_path))
    gamma, ht, hid = cfg.block_size, cfg.target_hidden, cfg.hidden_size
    g = Graph("draft")
    taps = [g.input(f"tap.{i}", (T, ht), DType.BF16) for i in range(cfg.n_taps)]
    anchor = g.input("anchor", (1,), DType.I32)
    block = drafter.lower_draft(g, DraftContext(taps, anchor))
    drafter.lower_select(g, block, prof)
    g.check()
    for p in DEFAULT_PASSES:
        p(g)
    prog = emit_program(g, pack=pf, profile=prof, dynamic_t=True, tail=None)
    eng = Engine(prog, dev)
    lay = prog.layout
    st = eng.buffers[prog.step_state]
    rng = np.random.default_rng(5)
    torch.manual_seed(5)
    state = {e.name: torch.zeros(e.shape, dtype=torch.bfloat16) for e in drafter.state_entries()}
    ctx_len = 0
    for step, (n_new, a0) in enumerate(((3, 17), (2, 42))):
        tap_np = [f32_to_bf16((rng.standard_normal((lay.t_max, ht)) * 2).astype(np.float32)) for _ in range(cfg.n_taps)]
        for i, arr in enumerate(tap_np):
            eng.buffers[f"tap.{i}"].write(arr.tobytes(), 0)
        s = lay.unpack(st.read(0, lay.size))
        s.update(t_this_step=1, anchor=a0, n_inject=n_new, prefill_left=0)
        st.write(lay.pack(s), 0)
        r = eng.run(1, steps_per_cb=1, in_flight=1)
        assert r.steps == 1 and not r.done
        # the oracle on the same rows
        taps_t = torch.cat([torch.from_numpy(bf16_to_f32(arr[:n_new])).to(torch.bfloat16) for arr in tap_np], dim=1)
        with torch.no_grad():
            feats = drafter.project_features(taps_t)
            hidden = drafter.draft_block(a0, feats, state, ctx_len=ctx_len)
            toks, corrected = drafter.draft_tokens(hidden, a0)
            conf = drafter.confidences(hidden, [a0] + toks[:-1])
            base = head.forward(hidden)
        got_feats = _bf16(eng.read("draft.feats"), (lay.t_max, hid))[:n_new]
        cos, max_abs, scale = _bars(got_feats, feats.float().numpy())
        assert cos > 0.9999 and max_abs <= 1e-2 * scale, ("features", step, cos, max_abs, scale)
        got_hidden = _bf16(eng.read("draft.hidden"), (gamma, hid))
        cos, max_abs, scale = _bars(got_hidden, hidden.float().numpy())
        assert cos > 0.999 and max_abs <= 3e-2 * scale, ("block hidden", step, cos, max_abs, scale)
        got_base = _bf16(eng.read("draft.base_logits"), (gamma, cfg.vocab_size))
        cos, max_abs, scale = _bars(got_base, base.float().numpy())
        assert cos > 0.999 and max_abs <= 3e-2 * scale, ("base logits", step, cos, max_abs, scale)
        s = lay.unpack(st.read(0, lay.size))
        got_toks = s["draft_tokens"][:gamma]
        for k in range(gamma):
            lg = _bf16(eng.read(f"draft.markov.{k}.logits"), (cfg.vocab_size,))
            ref = corrected[k].float().numpy()
            cos, max_abs, scale = _bars(lg, ref)
            assert cos > 0.999, ("corrected logits", step, k, cos)
            top2 = np.sort(ref)[-2:]
            assert got_toks[k] == toks[k] or top2[1] - top2[0] <= 0.05, ("draft", step, k, got_toks, toks)   # a near-tie may flip
            if got_toks[k] != toks[k]:
                break                                                       # the chain diverges after a flip
        else:
            assert np.abs(np.array(s["confidence"][:gamma]) - conf.numpy()).max() <= 2e-2, ("confidence", step)
        # verify_select's bookkeeping: the context grew by the injected rows, the next step's tokens are set
        assert s["drafter_ctx_len"] == ctx_len + n_new and s["n_inject"] == 0 and s["gamma"] == gamma
        L = s["verify_len"]
        exp_L = gamma if threshold <= 0 else drafter.confident_prefix(torch.tensor(s["confidence"][:gamma]), threshold, gamma)
        assert L == exp_L and s["t_this_step"] == 1 + L and s["pending_tokens"][: 1 + L] == [a0] + got_toks[:L]
        # the context caches of layer 0 hold the injected positions
        kc = _bf16(eng.read("draft.layers.0.self_attn.k_ctx"), (drafter.max_context, cfg.num_key_value_heads, cfg.head_dim))
        cos, max_abs, scale = _bars(kc[: ctx_len + n_new], state["draft.layers.0.self_attn.k_ctx"][: ctx_len + n_new].float().numpy())
        assert cos > 0.999, ("context keys", step, cos)
        ctx_len += n_new
    print(f"\ndraft program: {len(prog.ops)} dispatches, drafts {got_toks} (oracle {toks}), L = {L}")
