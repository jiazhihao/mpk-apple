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

## Format 2: affine INT4 groups (`formats/int4_affine`, MLX / AWQ / GPTQ) — 2026-09-24

* **The plugin (`monolith/formats/int4_affine.py`, ~150 lines):** `unpack` (U32 nibbles, F16/BF16 scales and
  biases → FP32 pairs), `dequantize`, `quantize` (mlx 0.32's `affine_quantize` rule reproduced bit-exactly in FP32 —
  the edge of larger magnitude snapped to an integer code, so it is *not* a fixed point under re-quantization; the
  contract test bounds the drift instead of asking for identity), `pack` / `unpack_pack` (FP32 (scale, bias) pairs per
  group per lane), the MSL snippet (`decode_word`, `decode_scale`, `decode_bias`). ~1 hour with its tests, once the
  MLX rule was read from its kernel source (`w_max` starts at 0, halves away from zero).
* **What the plugin path did not cover — four engine-side extensions, each small but each a real edit outside
  `formats/`:** (1) the decode contract had no bias: `SCALE_BIAS` in `gemv_T.metal` (`bias · Σx` per scale group,
  both activation paths); (2) mlx_lm quantizes `embed_tokens` and ties the head to it: the `EMBED_DEQUANT` gather;
  (3) the 0.8B's `down_proj` has K = 3584 — a lane stripe of 112 columns is 3.5 words and starts mid-group: the
  *ragged stripe* (partial tail word masked, scale bytes after the raw payload, `LANE_OFF` / `GROUP_SEG`, per-lane group
  lists in the pack), which the first formats never met because their shapes were multiples of 1024; (4) the
  conversion's tensor names (`language_model.model.*`) — a package name map through the reader. ~2.5 hours.
* **The value convention (the expensive part, ~2 hours):** the MLX 0.8B packed, every slab equal to the checkpoint's
  dequantization, every kernel exact on the real slabs — and the engine still disagreed with its own oracle from the
  first token. Bisection: a BF16 pack of the dequantized conversion disagreed too (so not the INT4 path); the HF
  checkpoint rewritten with MLX names / dtypes / conv layout agreed (so not the ingestion); hybrids swapping one tensor
  category from the conversion into the HF checkpoint pointed at the small tensors; a direct comparison showed mlx_lm
  stores the zero-centered RMSNorm weights as `1 + w` (its RMSNorm multiplies as stored) — both our paths added 1 again
  and ran a broken model whose flat logits made them disagree. Fix: a package-declared **value adapter** beside the
  name map (`checkpoint_adapt`, applied by the reader for the packer and the oracle alike): the tensors the package
  declares `one_plus` are read back as `bf16(1 + w) − 1`. After it: 32 greedy tokens equal, first token " Paris".
* **Lessons for the guide:** a new format usually arrives with a new *converter*, and the converter's conventions
  (names, folded constants, layouts) are the port's real risk — compare every small tensor against the HF checkpoint
  before running anything; a checkpoint whose engine and oracle disagree from token 0 on a prompt that decodes to
  noise is a broken model, not a broken kernel — check the prompt's ids against the model's own tokenizer, and the
  model's own answer to a real prompt, first. Total: about six hours; the plugin itself was one of them.

## Drafter 1: `Dogacel/Qwen3-8B-DSpark` as `spec/dspark/` — 2026-09-24

* The `Drafter` contract's module (config, tree, weight map, oracle) from the library: `Embedding`, `Linear`,
  `RMSNorm(one_plus=False)`, `DecoderLayer`, `GatedMLP`, plus one spec-local module, `DraftAttention` (three key
  sources, no mask) — ~250 lines. The k/v projections are claimed twice (block and context), which needed a loader
  that routes a tensor to every claimant and a `dequantized_tensors` fix to look tensors up by checkpoint name.
* Reference: DeepSpec's `Qwen3DSparkModel` loaded by direct construction (its `from_pretrained` re-serializes the
  config and drops the DSpark fields; TorchSpec's checkpoint omits `block_size`). ~1.5 hours including reading the
  reference.
