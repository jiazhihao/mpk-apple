"""Audit a fixed-input layer against the independent CPU contract oracle.

Timing is deliberately excluded. Both engines receive the benchmark's same
BF16 input and nonzero KV prefix (or zero recurrent state). The CPU oracle is
run with BF16 dequantized weights, as in the model goldens, and with FP32
dequantized weights to distinguish materialization rounding from accumulation.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def compare(got, ref):
    got, ref = np.asarray(got, np.float64).ravel(), np.asarray(ref, np.float64).ravel()
    cosine = float(got @ ref / (np.linalg.norm(got) * np.linalg.norm(ref)))
    return dict(cosine=cosine, max_abs=float(np.abs(got - ref).max()),
                passes_cosine=bool(np.isfinite(cosine) and cosine >= .999))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True)
    ap.add_argument('--pack', required=True)
    ap.add_argument('--layer', required=True, type=int)
    ap.add_argument('--tokens', type=int, default=1)
    ap.add_argument('--context', type=int, default=1024)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.tokens < 1 or a.context < 0:
        ap.error('tokens must be positive and context nonnegative')

    import torch
    import mlx.core as mx
    from mlx_lm import load
    from monolith.core.dtypes import DType
    from monolith.formats.fp import bf16_to_f32, f32_to_bf16
    from monolith.formats.safetensors_reader import SafetensorsDir
    from monolith.generate import Session
    from monolith.nn.pack_plan import dequantized_tensors
    from tools.bench.layer_fixed_vs_mlx import kv_prefix, mlx_stack, our_stack
    from tools.bench.layer_vs_mlx import our_model

    torch.set_num_threads(4)
    sess = Session(our_model(a.model, None, a.context + a.tokens + 256), a.pack, eos=-1)
    if not 0 <= a.layer < len(sess.model.layers()):
        ap.error('layer is outside the checkpoint')
    shape = (a.tokens, sess.model.config.hidden_size)
    x = bf16_to_f32(f32_to_bf16(np.random.default_rng(17).normal(0, .1, shape).astype(np.float32)))
    eng, outputs = our_stack(sess, [a.layer], a.tokens, a.context, x, True)
    eng.run(4, steps_per_cb=1, in_flight=2)
    got = bf16_to_f32(np.frombuffer(eng.read(outputs[-1], x.size * 2), np.uint16)).reshape(shape)
    model, _ = load(a.model)
    step, _ = mlx_stack(model, [a.layer], a.tokens, a.context, x, True)
    mlx = np.asarray(step().astype(mx.float32)).reshape(shape)

    layer = sess.model.layers()[a.layer]
    ckpt = SafetensorsDir(a.model, rename=sess.model.checkpoint_rename, adapt=sess.model.checkpoint_adapt)
    try:
        weights = [(name, arr.copy()) for name, arr, _ in dequantized_tensors(layer, ckpt)]
    finally:
        ckpt.close()
    result = dict(checkpoint=str(Path(a.model).resolve()), pack=str(Path(a.pack).resolve()),
                  layer=a.layer, T=a.tokens, ctx=a.context, threshold=.999,
                  chip=sess.dev.info().name, mlx_version=importlib.metadata.version('mlx'), torch_version=torch.__version__,
                  reference='Module.forward CPU oracle; FP32 accumulation and BF16 op boundaries',
                  prefix='random KV; zero recurrent input slot', mpk_vs_mlx=compare(got, mlx), oracles={})
    for dtype, label in ((torch.bfloat16, 'bf16_dequantized_weights'), (torch.float32, 'fp32_dequantized_weights')):
        layer.load_weights((name, torch.from_numpy(arr).to(dtype)) for name, arr in weights)
        state = {e.name: torch.zeros(tuple(e.shape), dtype={DType.BF16: torch.bfloat16, DType.F32: torch.float32}[e.dtype])
                 for e in layer.mixer.state_entries()}
        if hasattr(layer.mixer, 'kv_heads'):
            keys, vals = kv_prefix(a.layer, a.context, layer.mixer.kv_heads, layer.mixer.head_dim)
            for suffix, value in (('k_cache', keys), ('v_cache', vals)):
                state[layer.mixer.prefix + suffix][:a.context] = torch.from_numpy(value).to(torch.bfloat16)
        with torch.no_grad():
            ref = layer.forward(torch.from_numpy(x).to(torch.bfloat16), state, a.context).float().numpy()
        result['oracles'][label] = dict(mpk=compare(got, ref), mlx=compare(mlx, ref))
    text = json.dumps(result)
    print(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text + '\n')
    return int(not all(r['mpk']['passes_cosine'] for r in result['oracles'].values()))


if __name__ == '__main__':
    raise SystemExit(main())
