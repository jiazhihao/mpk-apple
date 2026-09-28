"""Disabling per-head norms must work through every compiler attention route."""
import pytest
from monolith.compiler import compile_program
from monolith.core import Profile
from monolith.formats import PackLayout
from monolith.models.qwen3 import Qwen3Model
from monolith.nn.pack_plan import pack_model
from monolith.packs import PackFile
from tests.contract.test_qwen3_package import _checkpoint


@pytest.mark.parametrize('attention', ['v1', 'v2', 'v3', 'mma', 'auto'])
def test_no_query_key_norm_weights_or_loads(tmp_path, attention):
    _checkpoint(tmp_path)
    model = Qwen3Model.from_checkpoint(str(tmp_path), max_context=16)
    for layer in model.layers():
        layer.mixer.qk_norm = False
    assert not any('.q_norm.' in name or '.k_norm.' in name for name in model.full_weight_map())
    pack_model(model, str(tmp_path), str(tmp_path / 'pack'), PackLayout())
    pack = PackFile(tmp_path / 'pack')
    profile = Profile.from_dict('test', {'gpu_cores': 20, 'nominal_gbps': 307,
        'engine': {'family': 'Apple10', 'lane_order': 'interleaved16'}})
    program = compile_program(model, pack, profile, t=4, attention=attention)
    attention_kernels = [k for k in program.kernels.values() if k.function.startswith('gqa_decode')]
    assert attention_kernels and all(k.macros['QK_NORM'] == '0' for k in attention_kernels)
