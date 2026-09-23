# DSpark — the speculative-decoding target: what it is, what exists for our model, what it costs here

Status: web research 2026-09-23 (nothing measured yet). Why it is in the design: [design](../design/design.md) D10 and
§5.8; plan M0 (drafters, baselines) and M6.

## 1. The method

DSpark (DeepSeek, "Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation",
[arXiv 2607.05147](https://arxiv.org/abs/2607.05147), July 2026; code and drafters MIT) is three pieces on top of a
DFlash-style block drafter ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036), MIT):

1. **Parallel backbone (DFlash).** A small transformer (5 layers for Qwen3-class targets) that proposes a whole block of
   γ tokens in one forward pass. Input = the anchor token (the last committed token) + γ−1 mask embeddings; attention is
   bidirectional inside the block. It reads the target's context through **KV injection**: hidden states from five
   target layers (uniformly spaced) are concatenated and projected, `Wc·[H^(l1);…;H^(l5)]`, and the result is fed
   through every draft layer's K/V projections as extra key/value entries — bypassing the draft layers' Q, output
   projection and FFN — and cached per context position across iterations. The drafter shares and freezes the target's
   `embed_tokens` and `lm_head`; the base logits of block position k are `Uₖ = lm_head(hₖ)`.
2. **Serial Markov head.** A rank-r bias that restores intra-block dependency:
   `B(xₖ₋₁, ·) = W₁[xₖ₋₁] W₂ ∈ ℝᵛ`, `W₁ ∈ ℝ^(V×r)`, `W₂ ∈ ℝ^(r×V)`, r = 256; applied left to right,
   `pₖ(v) ∝ exp(Uₖ(v) + Bₖ(v))`, sampling or argmax per position. Reported overhead 0.2–1.3 % of latency on GPUs.
3. **Confidence head.** `cₖ = σ(wᵀ[hₖ; W₁[xₖ₋₁]])`, trained toward the analytical acceptance rate
   `cₖ* = 1 − ½‖pₖᵈ − pₖᵗ‖₁`. Survival of a prefix of length l is `∏ᵢ≤ₗ cᵢ`, calibrated per position by Sequential
   Temperature Scaling (STS). The **scheduler** verifies only the prefix whose survival justifies its cost — in the
   paper a load-aware rule over the serving system's throughput curve; at batch 1 it reduces to maximizing expected
   accepted tokens per unit of verify cost, which is what our `verify_select` does with the chip profile's `cost(T)`
   table.

Verification is chain-based (not tree): standard rejection sampling with `min(1, pₜ(xₖ)/p_d(xₖ))`, or greedy match;
on the first rejection the rest of the block is discarded and the target's own token is taken. Training: frozen
target, losses `0.1·CE + 0.9·TV + 1.0·confidence-BCE` with position weights `wₖ = exp(−(k−1)/γ)`, ~1.3 M chat/math/code
samples. Reported: accepted length +26–31 % over EAGLE-3 and +16–18 % over DFlash on Qwen3-4/8/14B; DeepSeek-V4 at
matched throughput 57–85 % faster per user than MTP-1.

Ecosystem (all July–September 2026): vLLM, SGLang (`--speculative-algorithm DSPARK`;
[write-up](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/)), transformers, llama.cpp
([PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173): `--spec-type draft-dspark`, GGUF drafters with
`markov_w1/w2`, `conf_proj`, `dflash.block_size`; `--spec-draft-n-max`, `--spec-draft-p-min`), DFlash's own MLX
backend for Apple silicon (`dflash generate mlx --draft … --draft-bits 4 --block-size 5`), and training recipes in
[DeepSpec](https://github.com/deepseek-ai/DeepSpec) (MIT), SpecForge and
[NVIDIA NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding)
(targets incl. Qwen3 dense/MoE, Gemma4, DeepSeek V4, GLM-5.2, Kimi K3).

## 2. Drafters that exist for our targets

| Drafter | Target it was trained against | Architecture | Size | License | Notes |
|---|---|---|---|---|---|
| `DimInfer/Qwen3.8-27B-Dspark-v1` | Qwen3.8-27B **Q4_K_M GGUF** (hidden states captured from the quantized target) | 5 full-attention layers, hidden 5120, 32 q / 8 kv heads, head 128; taps target layers [1, 16, 31, 46, 61]; Markov rank 256; block 15 at training | safetensors 3.7 GB; GGUF Q8_0 2.0 GB, BF16 3.7 GB; 1.86 B params excl. shared embed/lm_head | Apache-2.0 | llama.cpp: `-md …-Q8_0.gguf --spec-type draft-dspark --spec-draft-n-max 4 -ngl 99 -ngld 99`. RTX 4090D, batch 1, 256 tokens: Math500 2.51× (accepted 4.06), GSM8K 2.49× (4.09), HumanEval 2.12× (3.46), LiveCodeBench 1.69× (2.73); acceptance 43–77 % |
| `gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4` | Qwen3.8-27B **NVFP4 (W4A4)**, retrained on-policy | `Qwen3DSparkModel`, 5 layers, hidden 5120, 40 heads / 8 kv, intermediate 10240; block 7 | 1.30 GB; MLP and `o_proj` in NVFP4 W4 group-16, q/k/v in BF16; 0.9 B params | Apache-2.0 | SGLang `--speculative-algorithm DSPARK --speculative-dspark-block-size 7 --speculative-draft-model-quantization modelopt_fp4`; the closest to our weight formats |
| `RadixArk/Qwen3.8-27B-DSpark` | Qwen3.8-27B NVFP4 / FP8 | 5 layers, hidden 5120, 32 q / 8 kv; taps [5, 19, 33, 47, 61]; Markov "VanillaMarkov" rank 256; block 7 (16 at training) | BF16 3.7 GB; 1.86 B params | **"other"** | SGLang; accepted length +26 % over 64 k prompts; 1.33–3.16× by workload and concurrency. Not used: license |
| `Dogacel/Qwen3-8B-DSpark` | Qwen3-8B | as DeepSpec's `dspark_qwen3_8b` config | — | (check) | model 2 in plan M8 |
| `deepseek-ai/DeepSeek-V4-{Flash,Pro}-DSpark` | DeepSeek-V4 | 3 MoE layers, block 5, greedy draft sampling | — | MIT | too large for any Mac we have; the reference ecosystem |

Also relevant: `z-lab/Qwen3.8-27B-DFlash2` (DFlash 2, no Markov/confidence heads; MLX backend) as a fallback drafter
and a second data point for acceptance on Apple hardware.

## 3. What a round costs on our hardware (estimates, to be measured in M6)

Per speculative round with γ = 7, relative to a T = 1 target pass (17.6 GB):

| Piece | Bytes | Share | Note |
|---|---|---|---|
| Drafter weights, once (T = γ in one pass) | 1.3 (NVFP4) / 2.0 (INT8) / 3.7 (BF16) GB | 7 / 11 / 21 % | the target's layer bodies at T = 7 with the drafter's weights |
| `lm_head` at T = γ | 0.72 GB (NVFP4) | 4 % | shared with the target |
| Markov bias, γ × W₂ | 7 × 127 MB (BF16) = 0.9 GB; 0.45 GB at INT8 | 5 / 2.5 % | sequential; ~0.45 ms per position on an M5 Pro; top-M pruning is an exactness question |
| Feature projection for the accepted positions | `Wc` 25,600 × 5,120 (262 MB BF16) + 5 layers' k/v projections | 1.5 % | once per round |
| Verify pass at T = 1 + L | 17.6 GB × `cost(1 + L)` | M5 Pro shader ALUs: FP8 ×1.08 / ×1.11 at T = 2 / 4, NVFP4 ×1.28 / ×1.79; accelerator ×1.5 at T = 8 **[M]** | the quantity `verify_select` optimizes |

Memory: target 21 GB + drafter 1.3–3.7 GB + injected-context KV (≈ 20 KB per committed token: 5 layers × 8 KV heads ×
128 × K and V in BF16) + state. On the 36 GB M3 Pro the NVFP4 or INT8 drafter fits comfortably; the BF16 one is tight.

Tokens per second ≈ `(1 + E[accepted]) / (t_draft + t_verify(1 + L))`. With the llama.cpp accepted lengths above
(2.7–4.1 at n-max 4) and the M5 Pro cost table, the break-even is comfortable on FP8 layers and marginal for the NVFP4
MLPs on the shader path — the reason the verify-length rule, the INT8/NVFP4 drafter and the Apple10 accelerator verify
path are all in the plan.

## 4. Open questions for M6

* Acceptance of drafters trained against Q4_K_M / NVFP4-W4A4 targets when the verifier is our W4A16 engine (same
  weights, different activation semantics): measure with llama.cpp `draft-dspark` on the M3 Pro first, then in our
  engine.
* The best `L` per chip: `verify_select` vs fixed L = 2 … 7, greedy and sampled.
* Whether `W₂` survives INT8 re-quantization without moving acceptance, and whether top-M bias pruning can be made exact.
* Whether the STS temperatures shipped with (or fitted for) a drafter transfer to Apple-sized contexts.

Sources: [DSpark paper](https://arxiv.org/abs/2607.05147) · [DFlash paper](https://arxiv.org/abs/2602.06036) ·
[SGLang integration](https://www.lmsys.org/blog/2026-07-06-dspark-sglang/) ·
[llama.cpp PR #25173](https://github.com/ggml-org/llama.cpp/pull/25173) ·
[DeepSpec](https://github.com/deepseek-ai/DeepSpec) · [z-lab/dflash](https://github.com/z-lab/dflash) ·
[NeMo AutoModel recipe](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/dspark-speculative-decoding) ·
[DimInfer drafter](https://huggingface.co/DimInfer/Qwen3.8-27B-Dspark-v1) ·
[gittensor NVFP4 drafter](https://huggingface.co/gittensor-model-hub/Qwen3.8-27B-DSpark-NVFP4) ·
[RadixArk drafter](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) ·
[DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) ·
[the DSpark batch item](https://www.deeplearning.ai/the-batch/deepseeks-dspark-gains-velocity)
