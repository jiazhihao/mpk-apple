"""The layer library and the first model package without torch or a real checkpoint: a synthetic BF16 checkpoint
of a two-layer hybrid (one GDN layer with v_heads = 2·k_heads, one gated attention layer, tied lm_head) is built,
packed from the module tree and read back; the lowering is checked against the design's stage count."""

import json
from collections import Counter

import numpy as np
import pytest

from monolith.core import Graph
from monolith.formats import PackLayout
from monolith.formats.fp import bf16_to_f32, f32_to_bf16
from monolith.formats.safetensors_reader import write_safetensors
from monolith.models import resolve_model
from monolith.models.qwen3_5 import Qwen3_5Config, Qwen3_5Model
from monolith.nn.pack_plan import aux_requests, pack_model, slab_requests
from monolith.packs import PackFile, interleave_chunks, rope_head_perm

CFG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "hidden_size": 64, "intermediate_size": 96, "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 32, "layer_types": ["linear_attention", "full_attention"],
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4, "rms_norm_eps": 1e-6, "vocab_size": 50, "max_position_embeddings": 1024,
        "attn_output_gate": True, "tie_word_embeddings": True, "hidden_act": "silu",
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    },
}
P = "model.language_model."


def _checkpoint(tmp_path):
    rng = np.random.default_rng(3)
    c = CFG["text_config"]
    h, inter, d = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    kd, vd = c["linear_num_key_heads"] * c["linear_key_head_dim"], c["linear_num_value_heads"] * c["linear_value_head_dim"]
    conv_dim = 2 * kd + vd

    def w(*shape):
        return (rng.standard_normal(shape) * 0.05).astype(np.float32)

    raw = {
        f"{P}embed_tokens.weight": w(c["vocab_size"], h),
        f"{P}norm.weight": w(h),
        f"{P}layers.0.input_layernorm.weight": w(h), f"{P}layers.0.post_attention_layernorm.weight": w(h),
        f"{P}layers.0.linear_attn.in_proj_qkv.weight": w(conv_dim, h), f"{P}layers.0.linear_attn.in_proj_z.weight": w(vd, h),
        f"{P}layers.0.linear_attn.in_proj_a.weight": w(4, h), f"{P}layers.0.linear_attn.in_proj_b.weight": w(4, h),
        f"{P}layers.0.linear_attn.out_proj.weight": w(h, vd), f"{P}layers.0.linear_attn.conv1d.weight": w(conv_dim, 1, 4),
        f"{P}layers.0.linear_attn.dt_bias": w(4), f"{P}layers.0.linear_attn.A_log": w(4), f"{P}layers.0.linear_attn.norm.weight": w(16),
        f"{P}layers.0.mlp.gate_proj.weight": w(inter, h), f"{P}layers.0.mlp.up_proj.weight": w(inter, h), f"{P}layers.0.mlp.down_proj.weight": w(h, inter),
        f"{P}layers.1.input_layernorm.weight": w(h), f"{P}layers.1.post_attention_layernorm.weight": w(h),
        f"{P}layers.1.self_attn.q_proj.weight": w(2 * 2 * d, h), f"{P}layers.1.self_attn.k_proj.weight": w(d, h),
        f"{P}layers.1.self_attn.v_proj.weight": w(d, h), f"{P}layers.1.self_attn.o_proj.weight": w(h, 2 * d),
        f"{P}layers.1.self_attn.q_norm.weight": w(d), f"{P}layers.1.self_attn.k_norm.weight": w(d),
        f"{P}layers.1.mlp.gate_proj.weight": w(inter, h), f"{P}layers.1.mlp.up_proj.weight": w(inter, h), f"{P}layers.1.mlp.down_proj.weight": w(h, inter),
        "model.visual.patch_embed.proj.weight": w(8, 8), "mtp.fc.weight": w(8, 8),
    }
    f32_keys = {f"{P}layers.0.linear_attn.A_log", f"{P}layers.0.linear_attn.norm.weight"}     # stored F32 in the real checkpoint
    tensors = {k: (("F32", v) if k in f32_keys else ("BF16", f32_to_bf16(v))) for k, v in raw.items()}
    write_safetensors(tmp_path / "model.safetensors", tensors, {"format": "pt"})
    with open(tmp_path / "config.json", "w") as f:
        json.dump(CFG, f)
    # what the reference model holds: every floating parameter as BF16
    held = {k: bf16_to_f32(f32_to_bf16(v)) for k, v in raw.items()}
    return held


def test_registry_and_weight_map(tmp_path):
    held = _checkpoint(tmp_path)
    assert resolve_model("Qwen3_5ForConditionalGeneration") is Qwen3_5Model
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    claimed = set(m.full_weight_map())
    text = {k for k in held if k.startswith(P)}
    assert claimed == text                        # every text tensor, nothing from the vision tower or the MTP head
    assert m.lm_head.tied is m.embed_tokens and m.lm_head.weight_map() == {}
    assert [e.name for e in m.state_spec().entries] == ["layers.0.linear_attn.conv_state", "layers.0.linear_attn.rec_state",
                                                        "layers.1.self_attn.k_cache", "layers.1.self_attn.v_cache"]
    assert m.state_spec().entries[1].shape == (4, 16, 16) and m.state_spec().entries[2].shape == (16, 1, 32)


def test_lowering_stage_count(tmp_path):
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    g = Graph("step")
    tok = m.lower(g)
    g.check()
    kinds = Counter(op.kind for op in g.ops)
    # 5 all-to-all stages per layer (design §5.1) + embed + lm_head + argmax; the norm statistics are separate ops
    # until the fuse pass hoists them (2 per layer + the final norm)
    assert kinds == {"gemv": 8, "gdn_mixer": 1, "gqa_decode": 1, "rmsnorm_stat": 5, "embed": 1, "lm_head": 1, "argmax": 1}
    assert tok.name == "token" and tuple(tok.shape)[0].name == "T"
    gdn = next(op for op in g.ops if op.kind == "gdn_mixer")
    assert gdn.attrs["updates"] == ["layers.0.linear_attn.conv_state", "layers.0.linear_attn.rec_state"]
    assert [s[0] for s in gdn.attrs["segments"]] == ["q", "k", "v", "z", "a", "b"]
    attn = next(op for op in g.ops if op.kind == "gqa_decode")
    assert attn.attrs["rope"] == "permuted" and attn.attrs["segments"][1][0] == "gate"
    assert sorted(m.tap_values) == [0, 1]
    # the tied lm_head reads the embedding slab
    embed_w = next(op for op in g.ops if op.kind == "embed").inputs[1]
    assert next(op for op in g.ops if op.kind == "lm_head").inputs[1] is embed_w
    with pytest.raises(ValueError):
        g.op("gdn_mixer", [g.values["layers.0.mlp.h"]], [g.value("bad", (1,), tok.dtype)], domain=gdn.domain,
             klass=gdn.klass, updates=["layers.1.self_attn.k_cache"])      # updating a state it does not read


def test_pack_from_model_tree(tmp_path):
    held = _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    layout = PackLayout(rows=16, lane_order="interleaved16")
    names = [r.name for r in slab_requests(m, layout)]
    assert names[:2] == ["embed_tokens.weight", "layers.0.linear_attn.in_proj.in_proj_qkv+in_proj_z+in_proj_a+in_proj_b"]
    assert "lm_head.weight" not in names
    manifest = pack_model(m, str(tmp_path), str(tmp_path / "pack"), layout)
    pf = PackFile(tmp_path / "pack")
    assert len(manifest["slabs"]) == 9 and {a["name"] for a in manifest["aux"]} >= {"rope_cos", "rope_sin", "layers.0.linear_attn.a_log"}
    L0, L1 = f"{P}layers.0.", f"{P}layers.1."
    # stacked GDN input projection: rows in checkpoint order
    got = pf.dequantize_slab("layers.0.linear_attn.in_proj.in_proj_qkv+in_proj_z+in_proj_a+in_proj_b")
    exp = np.concatenate([held[L0 + f"linear_attn.{n}.weight"] for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")])
    assert np.array_equal(got, exp)
    # attention: [q (head-permuted) | gate | k (head-permuted) | v]
    d, rot = 32, 8
    hp = rope_head_perm(d, rot)
    q_proj, k_proj, v_proj = held[L1 + "self_attn.q_proj.weight"], held[L1 + "self_attn.k_proj.weight"], held[L1 + "self_attn.v_proj.weight"]
    q = np.concatenate([q_proj[h * 2 * d: h * 2 * d + d][hp] for h in range(2)])
    gate = np.concatenate([q_proj[h * 2 * d + d: (h + 1) * 2 * d] for h in range(2)])
    exp = np.concatenate([q, gate, k_proj[hp], v_proj])
    assert np.array_equal(pf.dequantize_slab("layers.1.self_attn.qkv.q_proj+k_proj+v_proj"), exp)
    # gate/up chunk interleave (8 = pack rows / 2)
    gu = np.concatenate([held[L1 + "mlp.gate_proj.weight"], held[L1 + "mlp.up_proj.weight"]])[interleave_chunks(96, 96, 8)]
    assert np.array_equal(pf.dequantize_slab("layers.1.mlp.gate_up.gate_proj+up_proj"), gu)
    # aux transforms: (1 + w) norms, permuted per-head norms, -exp(A_log) from the BF16-valued parameter, tables
    assert np.array_equal(pf.aux_array("layers.0.input_norm.weight"), (1 + held[L0 + "input_layernorm.weight"]).astype(np.float32))
    assert np.array_equal(pf.aux_array("layers.1.self_attn.q_norm"), (1 + held[L1 + "self_attn.q_norm.weight"])[hp].astype(np.float32))
    assert np.array_equal(pf.aux_array("layers.0.linear_attn.a_log"), (-np.exp(held[L0 + "linear_attn.A_log"])).astype(np.float32))
    assert np.array_equal(pf.aux_array("layers.0.linear_attn.norm_w"), held[L0 + "linear_attn.norm.weight"].astype(np.float32))
    assert pf.aux_array("layers.0.linear_attn.conv_w").shape == (2 * 32 + 64, 1, 4)
    cos = pf.aux_array("rope_cos")
    assert cos.shape == (16, 32) and cos.dtype == np.uint16
    cosf = bf16_to_f32(cos)
    assert np.allclose(cosf[:, 4:16], 1.0) and np.allclose(cosf[:, 20:], 1.0) and abs(cosf[1, 0] - np.cos(1.0)) < 1e-2
    assert len(aux_requests(m)) == 2 * 2 + 1 + 4 + 2       # norms (2/layer + final), GDN aux, q/k norms


def test_mixed_format_parts_split_into_slabs(tmp_path):
    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    lin = m.blocks[0].mixer.in_proj
    lin.set_format("in_proj_a", "fp8_e4m3")
    lin.set_format("in_proj_b", "fp8_e4m3")
    groups = lin.slab_groups()
    assert [g.name.split(".")[-1] for g in groups] == ["in_proj_qkv+in_proj_z", "in_proj_a+in_proj_b"]
    g = Graph("mixed")
    proj = lin.lower(g, g.input("x", (1, 64), m.embed_tokens.weight_map()["weight"] and __import__("monolith.core", fromlist=["DType"]).DType.BF16))
    assert len(proj.values) == 2 and proj.segments["in_proj_a"] == (1, 0, 4) and proj.segments["in_proj_z"] == (0, 2 * 32 + 64, 64)
    m.blocks[1].mixer.qkv.set_format("v_proj", "nvfp4")
    with pytest.raises(ValueError):
        m.blocks[1].mixer.qkv.slab_groups()                   # a row permutation cannot span formats


def test_coverage_of_the_lowered_model(tmp_path):
    """Which op kinds still lack a kernel: after #19/#20 (embed, rmsnorm_stat, gemv, lm_head, argmax are bound on
    every profile) only the two mixers remain (#21 gqa_decode, #22 gdn_mixer)."""
    from monolith.compiler import CoverageError, check_coverage
    from monolith.core.profile import Profile

    _checkpoint(tmp_path)
    m = Qwen3_5Model.from_checkpoint(str(tmp_path), max_context=16)
    g = Graph("step")
    m.lower(g)
    prof = Profile.from_dict("p", {"gpu_cores": 20, "nominal_gbps": 307.0, "engine": {"family": "Apple10", "lane_order": "interleaved16"}})
    with pytest.raises(CoverageError) as ei:
        check_coverage(g, prof)
    assert sorted({op.kind for op, _ in ei.value.missing}) == ["gdn_mixer", "gqa_decode"]
