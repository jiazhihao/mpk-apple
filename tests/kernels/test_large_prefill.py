"""Separate large-prefill/small-decode programs preserve sequence state across chunk boundaries."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'contract'))
from test_nn_lowering import _checkpoint
from dspark_synth import write_checkpoint as write_dspark, build
from lm_synth import write_checkpoint as write_lm
from monolith.generate import Session
from monolith.formats import PackLayout
from monolith.models.qwen3_5 import Qwen3_5Model
from monolith.nn.pack_plan import pack_model
from monolith.spec.lm import LMDrafter


@pytest.mark.parametrize('draft_kind', [None, 'lm', 'dspark'])
@pytest.mark.parametrize('accelerator', ['off', 'on'])
def test_large_prefill_shares_state_with_small_decode(tmp_path, draft_kind, accelerator):
    target, draft = tmp_path / 'target', tmp_path / 'draft'
    target.mkdir(); draft.mkdir()
    _checkpoint(target)
    def model():
        return Qwen3_5Model.from_checkpoint(str(target), max_context=512)
    pack_model(model(), str(target), str(target / 'pack'), PackLayout(rows=16))
    reference = Session(model(), str(target / 'pack'), prefill_chunk_size=8, eos=-1, autotune=False,
                        accelerator='off', attention='v1')
    m = model()
    options = {}
    if draft_kind == 'lm':
        write_lm(draft, vocab_size=50)
        d = LMDrafter.from_checkpoint(str(draft), target_lm_head=m.lm_head, max_context=512, gamma=3)
    elif draft_kind == 'dspark':
        write_dspark(draft, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1])
        d, _, _, _ = build(draft, target_lm_head=m.lm_head, max_context=512)
    if draft_kind:
        pack_model(d, str(draft), str(draft / 'pack'), PackLayout(rows=16))
        options = dict(drafter=d, drafter_pack=str(draft / 'pack'), verify='fixed', verify_length=3)
    session = Session(m, str(target / 'pack'), eos=-1, autotune=False, attention='v1', accelerator=accelerator, **options)
    # Small-to-large allocation growth, exact boundary, partial chunk, several chunks, then reuse the small graph.
    for count in (5, 128, 129, 259, 5):
        ids = np.random.default_rng(count).integers(0, 50, count).tolist()
        expected = reference.generate(ids, 12).tokens
        result = session.generate(ids, 12)
        assert result.tokens == expected
        pre = session.prefill_engine(count)
        dec = session.engine(0 if draft_kind else 1)
        assert pre is not dec
        assert pre.buffers['step_state'] is dec.buffers['step_state']
        for name, spec in dec.program.buffers.items():
            if spec.role in ('state', 'weights', 'ring'):
                assert pre.buffers[name] is dec.buffers[name]
        assert dec.state()['position'] == count + 11
        assert dec.state()['error'] == 0
        vocab = m.config.vocab_size
        assert dec.program.buffers['logits'].nbytes == (8 if draft_kind else 1) * vocab * 2
        assert session.decode_t_max == 8
        assert session.prefill_chunk_size == 128
