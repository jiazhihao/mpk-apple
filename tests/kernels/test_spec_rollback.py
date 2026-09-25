"""Speculative greedy decode equals plain greedy decode (design §5.8, #38/#39): the synthetic hybrid target of the
contract tier (a GDN layer whose recurrent state the commit pass must roll back on every rejected draft, and an
attention layer) with a random synthetic DSpark drafter (almost every draft is rejected) — the whole round replayed
from one encode, tokens drained from the ring — against the plain step program on the same weights. No torch."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
from dspark_synth import build, write_checkpoint  # noqa: E402
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.formats import PackLayout  # noqa: E402
from monolith.generate import Session  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    root = tmp_path_factory.mktemp("spec")
    tdir, ddir = root / "target", root / "drafter"
    tdir.mkdir(); ddir.mkdir()
    _checkpoint(tdir)
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=64)
    pack_model(model, str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, seed=5, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=3)
    return tdir, ddir


def _model(tdir):
    return Qwen3_5Model.from_checkpoint(str(tdir), max_context=64)


@pytest.mark.parametrize("verify", ["threshold", "cost"])
def test_speculative_equals_plain_greedy(packs, verify):
    tdir, ddir = packs
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False)
    model = _model(tdir)
    drafter, _, cfg, _ = build(ddir, target_lm_head=model.lm_head)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=drafter, drafter_pack=str(ddir / "pack"), verify=verify)
    rng = np.random.default_rng(3)
    for n_prompt, n_new in ((5, 24), (11, 20), (1, 12)):
        ids = [int(x) for x in rng.integers(0, 50, n_prompt)]
        ref = plain.generate(ids, n_new)
        got = spec.generate(ids, n_new)
        assert got.tokens == ref.tokens, (n_prompt, got.tokens, ref.tokens)
        assert len(got.tokens) == n_new and got.decode_tokens == n_new - 1
        assert got.accepted is not None and len(got.accepted) == got.steps and sum(got.committed) >= got.decode_tokens
        assert all(1 <= c <= 4 for c in got.committed) and all(a == c - 1 for a, c in zip(got.accepted, got.committed))
        assert all(a <= cfg.block_size for a in got.accepted)
        print(f"\nprompt {n_prompt}: {got.steps} steps for {got.decode_tokens} tokens, accepted {got.accepted}")
    # the drafter's context grew with every committed token and the ring was drained in order
    st = spec.engine(0).state()
    assert st["drafter_ctx_len"] == st["position"] == 1 + 12 - 1 and st["ring_head"] == st["ring_tail"] == 12
    assert st["error"] == 0
