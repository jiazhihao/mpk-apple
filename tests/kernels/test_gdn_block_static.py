"""Full projection-inclusive static block: compare with its split control."""
import numpy as np
import pytest

from monolith.formats.fp import f32_to_bf16
from monolith.runtime import Engine, _native as nt
from tools.bench.gdn_block_bench import fixture, program, initialize, checked_run, snapshot
from tools.bench.gdn_block_static import normalize, merge


@pytest.fixture(scope='module')
def block(tmp_path_factory):
    pytest.importorskip('torch')
    pytest.importorskip('safetensors')
    dev = nt.Device()
    root = tmp_path_factory.mktemp('gdn-block')
    fixture(root, hidden=1024, hk=8, hv=16)
    # Reopening an existing fixture must restore the mixed FP8/BF16 slabs.
    module, pack = fixture(root, hidden=1024, hk=8, hv=16)
    return dev, module, program(dev, module, pack)


@pytest.mark.parametrize('sgs,workers', [(4,4), (16,8)])
def test_full_block_fusion_preserves_control_and_state(block, sgs, workers):
    dev, module, p = block
    control = normalize(p, sgs, mode='coop', groups=workers)
    fused = merge(control, workers, sgs)
    assert len(fused.ops) == 1
    engines = [Engine(pr, dev) for pr in (control, fused)]
    for e in engines:
        initialize(e, module)
    rng = np.random.default_rng(100)
    for step in range(4):
        x = f32_to_bf16(rng.normal(0,.1,(8,1024)).astype(np.float32)).tobytes()
        for e in engines:
            e.buffers['hidden'].write(x,0)
            e.buffers[e.program.step_state].write(e.program.layout.pack({'step':step,'t_this_step':8}),0)
            checked_run(e,1)
        assert snapshot(engines[0],1024) == snapshot(engines[1],1024)
    for e in engines:
        checked_run(e,64)
    assert snapshot(engines[0],1024) == snapshot(engines[1],1024)
