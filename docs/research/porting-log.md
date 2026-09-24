# Porting log — what adding a model, a format or a drafter took (plan M8, feeds the porting guide #48)

Recorded as it happens, with the time and the files touched, so the porting guide is derived from evidence.

## Model 2: Qwen3-8B (`nvidia/Qwen3-8B-NVFP4`) — 2026-09-24

* **Library generalization (separate PR, `monolith/nn/attention.py`):** one flag, `GQAAttention(norm_one_plus=False)`,
  so the per-head q/k RMSNorm scales by `w` (the standard RMSNorm) instead of `1 + w`. With `gate=False` and
  `rotary_dim = head_dim` (an identity head-dim permutation) the hybrid's attention module *is* the dense Qwen3
  attention. ~15 minutes including its test.
* **The package (`monolith/models/qwen3/`, 3 files, ~170 lines):** `config.py` (the fields the tree needs, RoPE-type
  and sliding-window guards), `model.py` (the tree from library modules: `Embedding`, `DecoderLayer(RMSNorm,
  GQAAttention, RMSNorm, GatedMLP)`, `LMHead` with its own BF16 weight, `GreedySampler`; full-RoPE tables), `weights.py`
  (the `model.` prefix; NVIDIA's `input_scale` / `k_scale` / `v_scale` side tensors are ignored by the weight-only path).
  Written against transformers' `modeling_qwen3.py`. ~25 minutes.
* **No kernel, runtime, compiler or format change.** The checkpoint packs from the tree in 4 s (144 NVFP4 slabs +
  the BF16 embedding and lm_head, 6.3 GiB); `python -m monolith.generate` decodes it from the first run.
* **Goldens:** `monolith.formats.dequant` → a 15 GB BF16 checkpoint, `tools/goldens/hf_golden.py` on the CPU
  (transformers' `Qwen3ForCausalLM`), checked in under `tests/models/qwen3/goldens/`.
* **Tests:** a torch-free contract test on a synthetic checkpoint (weight map, lowering, coverage, pack round-trip)
  and the GPU golden test (greedy tokens and every layer's prefill residual stream read from the program's buffers).
* Total: about an hour from the first line to the running model, most of it waiting for downloads and the CPU golden.
