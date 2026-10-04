"""Paired recipes own their barrier epochs even when model buffers are shared."""
from types import SimpleNamespace

from tools.bench.dspark_round_latency import comparison_buffers


def test_pairing_preserves_model_state_but_isolates_worker_counters():
    # A 240-worker candidate's counter 160 is the 160-worker control's release
    # word. They must never alias in a paired run with different crew sizes.
    model={name:object() for name in ('weights','step_state','ring','layer.rec_state','layer.k_cache','draft.hidden')}
    synchronization={name:object() for name in (
        'mega.flags','mega.tasks','draft.mixer.0.mega.flags','draft.mixer.0.mega.tasks',
        'target.gdn.0.mega.flags','draft.markov.chain.mega.flags')}
    original=dict(model,**synchronization)
    shared=comparison_buffers(SimpleNamespace(buffers=original))
    assert shared==model
    assert all(shared[name] is value for name,value in model.items())
    assert all(name in original for name in synchronization)
