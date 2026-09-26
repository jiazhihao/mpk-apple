"""The layer-level comparison against MLX (#113): the cost of one decoder layer of a decode step on both engines,
measured the same way — the slope of the step's time over the number of layers the model is truncated to — at T = 1
and at a verify pass's T, over a given context; per sub-op, ours from the program's per-op profile and MLX's from
microbenchmarks of its ops at the layer's shapes.

    python tools/bench/layer_vs_mlx.py --model ~/models/mlx-community-Qwen3-0.6B-4bit --pack <pack> \\
        [--layers 28,14] [--ts 1,4,8] [--ctx 128,1024] [--steps 48] [--reps 3] [--out <jsonl>]

Ours: the model package built with ``num_layers_override = k`` on the full pack, a plain session; T = 1 through
``Session.generate`` (GPU and wall ms per step), T > 1 through the static-T program run as prefill-like steps
(``prefill_left`` held above 0: every step processes T tokens, the position advances by T). MLX: mlx-lm's model with
``model.layers[:k]``, T = 1 through ``stream_generate`` (its ``generation_tps``, the gate's own number), T > 1 through
the pipelined ``async_eval`` loop of its generate step. Both engines see the same prompt (``ctx`` random tokens),
context and steps; every number is the best of ``reps`` paired alternations.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---- ours -------------------------------------------------------------------------------------------------------------

def our_model(model_dir: str, k: Optional[int], max_context: int):
    from monolith.models import resolve_model

    with open(Path(model_dir) / "config.json") as f:
        arch = json.load(f)["architectures"][0]
    cls = resolve_model(arch)
    if cls is None:
        raise RuntimeError(f"no model package registered for {arch!r}")
    opts: Dict[str, Any] = {"max_context": max_context}
    if k is not None:
        opts["num_layers_override"] = k
    return cls.from_checkpoint(model_dir, **opts)


def _armed(sess, ids: Sequence[int], t: int):
    """The engine for ``t`` tokens per step with the prompt in the caches and the step state set to run T-token steps
    that only advance the position (``prefill_left`` held at 1): the verify pass's cost without a drafter."""
    sess.generate(list(ids), 2)                                        # the prompt in the caches, the states real
    eng = sess.engine(t)
    layout = eng.program.layout
    stb = eng.buffers[eng.program.step_state]
    state = layout.unpack(stb.read(0, layout.size))
    state.update(done=0, stop_at=0, t_this_step=t, prefill_left=1, pending_tokens=[int(x) for x in ids[:t]] + [0] * (layout.field("pending_tokens").count - t))
    stb.write(layout.pack(state), 0)
    return eng


def our_step_ms(sess, ids: Sequence[int], t: int, steps: int) -> Tuple[float, float]:
    """(GPU ms, wall ms) per step of ``steps`` decode steps at ``t`` tokens per step after the prompt's prefill."""
    if t == 1:
        g = sess.generate(list(ids), steps + 1)
        return g.decode_ms / max(1, g.steps), g.decode_wall_ms / max(1, g.steps)
    eng = _armed(sess, ids, t)
    r = eng.run(steps, steps_per_cb=1, in_flight=2)
    return r.gpu_ms / max(1, r.steps), r.wall_ms / max(1, r.steps)


def our_op_profile(sess, ids: Sequence[int], t: int, steps: int = 6) -> Dict[str, Dict[str, float]]:
    """Per-op min ms of the step program at ``t``, folded per layer: ``{op kind: mean over layers of the per-layer
    sum}``, the non-layer ops under their own names; a layer is the ops between one layer's first GEMV and the next."""
    from monolith.trace import op_timings

    if t == 1:
        sess.generate(list(ids), 2)                                    # the state of a decode step (a T > 1 run leaves T in it)
        eng = sess.engine(1)
        layout = eng.program.layout
        stb = eng.buffers[eng.program.step_state]
        state = layout.unpack(stb.read(0, layout.size))
        state.update(done=0, stop_at=0)
        stb.write(layout.pack(state), 0)
    else:
        eng = _armed(sess, ids, t)
    runs = eng.profile(steps)
    per_layer: Dict[int, Dict[str, float]] = {}
    other: Dict[str, float] = {}
    layer = None
    for tm in op_timings(eng.program, runs):
        name = tm.name
        if name.startswith("gemv:layers.") or name.startswith("gemm_tile:layers."):
            layer = int(name.split("layers.")[1].split(".")[0])
        elif name.startswith(("lm_head", "embed", "argmax", "advance", "sample")) or (layer is None):
            other[tm.kind if tm.kind != "gemv" else name.split(":")[0]] = other.get(tm.kind, 0.0) + tm.ms_min
            continue
        key = tm.kind
        if tm.kind in ("gemv", "gemm_tile"):
            part = name.split(".")[-1].split("+")[0]                       # q_proj / o_proj / gate_proj / down_proj / in_proj_qkv …
            key = f"{tm.kind}:{part}"
        d = per_layer.setdefault(layer, {})
        d[key] = d.get(key, 0.0) + tm.ms_min
    kinds = sorted({k for d in per_layer.values() for k in d})
    n = max(1, len(per_layer))
    folded = {k: sum(d.get(k, 0.0) for d in per_layer.values()) / n for k in kinds}
    folded["_layer_total"] = sum(folded.values())
    folded["_dispatches_per_layer"] = sum(1 for tm in op_timings(eng.program, runs) if tm.name.startswith(("gemv:layers.", "gemm_tile:layers.")) or tm.kind in ("gqa_decode", "gqa_merge", "rmsnorm_stat", "norm_apply", "x_permute", "gdn_mixer", "gdn_norm")) / n
    return {"layer": folded, "other": other}


# ---- MLX ----------------------------------------------------------------------------------------------------------------

def mlx_step_ms(model, tokenizer, ids: Sequence[int], t: int, steps: int) -> float:
    """Wall ms per decode step at ``t`` tokens per step after the prompt; T = 1 is mlx-lm's own ``generation_tps``."""
    import mlx.core as mx
    from mlx_lm import stream_generate
    from mlx_lm.models.cache import make_prompt_cache

    if t == 1:
        tps = 0.0
        for r in stream_generate(model, tokenizer, prompt=list(ids), max_tokens=steps + 1):
            tps = r.generation_tps
        return 1e3 / tps
    cache = make_prompt_cache(model)
    model(mx.array(list(ids))[None], cache=cache)
    mx.eval([c.state for c in cache])
    y = mx.array(list(ids[:t]))[None]
    pending: List[Any] = []
    for _ in range(4):                                                  # warm: the graph, the cache's growth
        out = model(y, cache=cache); mx.async_eval(out); pending.append(out)
        if len(pending) > 1:
            mx.eval(pending.pop(0))
    mx.eval(*pending); pending.clear()
    t0 = time.perf_counter()
    for _ in range(steps):
        out = model(y, cache=cache)
        mx.async_eval(out)
        pending.append(out)
        if len(pending) > 1:
            mx.eval(pending.pop(0))
    mx.eval(*pending)
    return 1e3 * (time.perf_counter() - t0) / steps


def mlx_op_profile(model, t: int, ctx: int, reps: int = 64) -> Dict[str, float]:
    """MLX's ops at the first layer's shapes: ``reps`` independent calls evaluated together, ms per call."""
    import mlx.core as mx

    lyr = model.model.layers[0]
    attn, mlp = lyr.self_attn, lyr.mlp
    dtype = mx.bfloat16
    x = mx.random.normal((1, t, model.args.hidden_size)).astype(dtype)
    heads, kv, d = attn.n_heads, attn.n_kv_heads, model.args.head_dim
    q = mx.random.normal((1, heads, t, d)).astype(dtype)
    k = mx.random.normal((1, kv, ctx + t, d)).astype(dtype)
    v = mx.random.normal((1, kv, ctx + t, d)).astype(dtype)
    hq = mx.random.normal((1, t, heads, d)).astype(dtype)
    big = mx.random.normal((1, t, model.args.intermediate_size)).astype(dtype)

    def qmm(lin, inp):
        mode = getattr(lin, "mode", "affine")                          # mlx's affine (scales + biases) or nvfp4 / mxfp4 (scales alone)
        biases = getattr(lin, "biases", None)
        if biases is None or mode != "affine":
            return lambda: mx.quantized_matmul(inp, lin.weight, lin.scales, transpose=True, group_size=lin.group_size, bits=lin.bits, mode=mode)
        return lambda: mx.quantized_matmul(inp, lin.weight, lin.scales, biases, transpose=True, group_size=lin.group_size, bits=lin.bits)

    ops = {
        "q_proj": qmm(attn.q_proj, x), "k_proj": qmm(attn.k_proj, x), "v_proj": qmm(attn.v_proj, x),
        "o_proj": qmm(attn.o_proj, mx.random.normal((1, t, heads * d)).astype(dtype)),
        "gate_proj": qmm(mlp.gate_proj, x), "up_proj": qmm(mlp.up_proj, x), "down_proj": qmm(mlp.down_proj, big),
        "input_norm": lambda: mx.fast.rms_norm(x, lyr.input_layernorm.weight, lyr.input_layernorm.eps),
        "q_norm": lambda: mx.fast.rms_norm(hq, attn.q_norm.weight, attn.q_norm.eps),
        "rope_q": lambda: mx.fast.rope(q, d, traditional=False, base=model.args.rope_theta, scale=1.0, offset=ctx),
        "sdpa": lambda: mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale, mask=None if t == 1 else "causal"),
        "swiglu": lambda: mx.multiply(mx.sigmoid(big) * big, big),
        "residual": lambda: mx.add(x, x),
    }
    out: Dict[str, float] = {}
    for name, fn in ops.items():
        mx.eval(fn())
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            mx.eval(*[fn() for _ in range(reps)])
            best = min(best, (time.perf_counter() - t0) * 1e3 / reps)
        out[name] = best
    per_layer = (out["q_proj"] + out["k_proj"] + out["v_proj"] + out["o_proj"] + out["gate_proj"] + out["up_proj"] + out["down_proj"]
                 + 2 * out["input_norm"] + 2 * out["q_norm"] + 2 * out["rope_q"] + out["sdpa"] + out["swiglu"] + 2 * out["residual"])
    out["_layer_sum"] = per_layer
    return out


# ---- main -----------------------------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pack", required=True)
    ap.add_argument("--layers", default=None, help="comma list of layer counts to truncate to (default: all and half)")
    ap.add_argument("--ts", default="1,4,8")
    ap.add_argument("--ctx", default="128,1024")
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-mlx", action="store_true")
    ap.add_argument("--no-ops", action="store_true", help="skip the per-op profiles")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from monolith.generate import Session

    with open(Path(a.model) / "config.json") as f:
        cfg = json.load(f)
    n_layers = int(cfg.get("num_hidden_layers") or cfg["text_config"]["num_hidden_layers"])
    layers = [int(x) for x in a.layers.split(",")] if a.layers else [n_layers, n_layers // 2]
    ts = [int(x) for x in a.ts.split(",")]
    ctxs = [int(x) for x in a.ctx.split(",")]
    rng = np.random.default_rng(0)
    vocab = int(cfg.get("vocab_size") or cfg["text_config"]["vocab_size"])
    prompts = {c: [int(x) for x in rng.integers(1000, min(vocab, 100000), c)] for c in ctxs}
    max_context = max(ctxs) + a.steps * max(ts) + 64
    mlx_model = mlx_tok = None
    if not a.no_mlx:
        from mlx_lm import load as mlx_load

        mlx_model, mlx_tok = mlx_load(a.model)
        full_layers = list(mlx_model.model.layers)
    rows: List[Dict[str, Any]] = []
    name = Path(a.model).name
    results: Dict[Tuple[str, int, int, int], Dict[str, float]] = {}
    for k in layers:
        sess = Session(our_model(a.model, k, max_context), a.pack, eos=-1)
        info = sess.dev.info()
        if mlx_model is not None:
            mlx_model.model.layers = full_layers[:k]
        for t in ts:
            for c in ctxs:
                ours_gpu, ours_wall, mlx_wall = float("inf"), float("inf"), float("inf")
                for rep in range(a.reps):
                    order = ("ours", "mlx") if rep % 2 == 0 else ("mlx", "ours")
                    for eng in order:
                        if eng == "ours":
                            g, w = our_step_ms(sess, prompts[c], t, a.steps)
                            ours_gpu, ours_wall = min(ours_gpu, g), min(ours_wall, w)
                        elif mlx_model is not None:
                            mlx_wall = min(mlx_wall, mlx_step_ms(mlx_model, mlx_tok, prompts[c], t, a.steps))
                results[("ours", k, t, c)] = {"gpu": ours_gpu, "wall": ours_wall}
                results[("mlx", k, t, c)] = {"wall": mlx_wall}
                print(f"  layers {k:3d} T {t} ctx {c:5d}: ours {ours_gpu:7.3f} ms GPU / {ours_wall:7.3f} wall per step; mlx-lm {mlx_wall:7.3f} wall", flush=True)
                rows.append({"chip": info.name, "date": time.strftime("%Y-%m-%d %H:%M"), "model": name, "layers": k, "T": t, "ctx": c, "steps": a.steps,
                             "ours_gpu_ms": round(ours_gpu, 4), "ours_wall_ms": round(ours_wall, 4), "mlx_wall_ms": round(mlx_wall, 4)})
        if not a.no_ops and k == layers[0]:
            for t in ts:
                prof = our_op_profile(sess, prompts[ctxs[0]], t)
                print(f"  ours per-layer ops at T {t} (min ms, mean over layers): " + ", ".join(f"{kk} {v:.4f}" for kk, v in sorted(prof["layer"].items())), flush=True)
                rows.append({"chip": info.name, "model": name, "T": t, "engine": "ours", "per_layer_ops": {kk: round(v, 5) for kk, v in prof["layer"].items()},
                             "other_ops": {kk: round(v, 5) for kk, v in prof["other"].items()}})
                if mlx_model is not None:
                    for c in ctxs:
                        mp = mlx_op_profile(mlx_model, t, c)
                        print(f"  mlx per-layer ops at T {t} ctx {c} (ms per call): " + ", ".join(f"{kk} {v:.4f}" for kk, v in sorted(mp.items())), flush=True)
                        rows.append({"chip": info.name, "model": name, "T": t, "ctx": c, "engine": "mlx", "per_layer_ops": {kk: round(v, 5) for kk, v in mp.items()}})
        sess.engines.clear()
        del sess
    # the per-layer slopes
    print("\n| model | T | ctx | ours ms / layer (GPU) | ours (wall) | mlx-lm ms / layer (wall) | ours / mlx | ours step (all layers, wall) | mlx step |")
    print("|---|---|---|---|---|---|---|---|---|")
    if len(layers) >= 2:
        k1, k2 = layers[0], layers[1]
        for t in ts:
            for c in ctxs:
                o1, o2 = results[("ours", k1, t, c)], results[("ours", k2, t, c)]
                m1, m2 = results[("mlx", k1, t, c)], results[("mlx", k2, t, c)]
                so_g = (o1["gpu"] - o2["gpu"]) / (k1 - k2)
                so_w = (o1["wall"] - o2["wall"]) / (k1 - k2)
                sm = (m1["wall"] - m2["wall"]) / (k1 - k2) if mlx_model is not None else float("nan")
                print(f"| {name} | {t} | {c} | {so_g * 1e3:.1f} µs | {so_w * 1e3:.1f} µs | {sm * 1e3:.1f} µs | {so_w / sm if sm else float('nan'):.2f} | {o1['wall']:.3f} ms | {m1['wall']:.3f} ms |")
                rows.append({"chip": info.name, "date": time.strftime("%Y-%m-%d %H:%M"), "model": name, "T": t, "ctx": c, "slope": True,
                             "ours_layer_gpu_ms": round(so_g, 5), "ours_layer_wall_ms": round(so_w, 5), "mlx_layer_wall_ms": round(sm, 5),
                             "ours_intercept_ms": round(o1["wall"] - so_w * k1, 4), "mlx_intercept_ms": round(m1["wall"] - sm * k1, 4) if mlx_model is not None else None})
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"rows appended to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
