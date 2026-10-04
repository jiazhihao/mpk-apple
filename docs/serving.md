# lithos-metal serving and coding agents

Install lithos-metal with `pip install '.[serve]'` from a checkout; this builds the Metal runtime
and includes the kernel sources. The [installation guide](installation.md) covers the
precompiled Homebrew release bundle.

```bash
lithos-metal serve --model nvidia/Qwen3.8-27B-NVFP4
# Or:
lithos-metal serve --model nvidia/Qwen3.6-35B-A3B-NVFP4
```

The matching published LithosAI NVFP4 DSpark head is selected automatically. An explicit
`--draft PATH_OR_HUB_ID` overrides it; `--no-draft` disables speculation. Automatic pairing
uses the known target ID, a Hub snapshot path, checkpoint provenance, or a recognizable
local directory name plus architecture/dimensions. Unrecognized or renamed local copies
need an explicit `--draft`; arbitrary fine-tunes are not matched by tensor dimensions alone.

The default served model ID is the full Hugging Face ID, or the local directory name.
`--served-model-name NAME` overrides it. The launchers discover it through `/v1/models`:

```bash
lithos-metal opencode
lithos-metal claude
lithos-metal codex
lithos-metal hermes
lithos-metal codex --url http://127.0.0.1:8001 -- exec 'Explain this repository'
lithos-metal env                      # Shell exports for another OpenAI-compatible client
lithos-metal run -- your-client       # Run with OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL
```

Install the agent itself separately. lithos-metal uses process-local configuration and forwards
arguments after `--`; it does not change client approval/sandbox settings or global config
files. `--print-config` prints a redacted launch plan. Set `LITHOS_METAL_URL` for another endpoint and
`LITHOS_METAL_API_KEY` for an authenticated server. `lithos-metal env` intentionally prints shell-ready exports
including the configured key; avoid putting its output in public logs.

| Client | Connection |
|---|---|
| OpenCode | Inline `@ai-sdk/openai-compatible` provider, Chat Completions |
| Claude Code | Anthropic base URL and model environment, Messages and token counting |
| Codex | Custom `lithos-metal` provider, stateless Responses over SSE |
| Hermes | Custom provider and OpenAI endpoint environment |
| Other clients | `/v1/chat/completions`, `/v1/responses`, or `/v1/messages` |

Configuration references: [OpenCode](https://opencode.ai/docs/config/),
[Claude Code](https://code.claude.com/docs/en/llm-gateway),
[Codex](https://developers.openai.com/codex/config-advanced/),
[Hermes](https://hermes-agent.nousresearch.com/docs/user-guide/configuration/).

DSpark is currently the only serving drafter (`--draft-kind dspark`). `--drafter` is an alias
for `--draft`; use `--no-draft` for target-only generation. Local paths can name a checkpoint directory,
its `config.json`, or one of its safetensors files, with the configuration, all weight shards,
and tokenizer files alongside it. Hugging Face IDs use `snapshot_download`; `--revision` and
`--draft-revision` pin revisions, `--download-dir` changes the Hub cache, and
`--local-files-only` requires already downloaded snapshots. Remote Python code is not loaded.

`--pack` is optional. By default, target and draft packs are built once under
`${XDG_CACHE_HOME:-~/.cache}/lithos-metal/packs`; an existing `lmk/packs` or
`monolith/packs` cache is reused. `--pack /path/to/cache` selects another root.
`--draft-pack` optionally selects a separate draft cache. Entries are keyed by checkpoint path,
configuration/index hashes, shard sizes/mtimes, chip backend, layout, context capacity and
quantization. Concurrent startups share a file lock; a pack becomes visible only after
successful validation and an atomic rename. A truncated or incompatible existing entry fails
with an error instead of being reused. Delete that specific entry to rebuild it.

Existing manually prepared packs containing `manifest.json` can still be passed directly to
`--pack` and `--draft-pack`. They must match tensor shapes, formats and capacity. Legacy packs
without source identity metadata must have been prepared from the selected checkpoint.
Use a cache root to manage checkpoint identity automatically.

On the matching 40-core M5 Max, the 27B target plus seven-token DSpark drafter selects the
previously measured target/draft recipes automatically. The validated 35B-A3B hybrid
MoE and its six-layer DSpark drafter also select NVFP4 automatically; see the
[conversion and acceptance checks](qwen-hybrid-moe.md#nvfp4-draft-conversion).
`--draft-quantization auto` uses NVFP4 for these combinations and retains the Markov
embedding in BF16. An explicit existing
draft pack determines its precision. Other combinations retain source precision unless
`--draft-quantization nvfp4` is requested. Use `--draft-quantization none` for source-precision
draft weights; draft quantization can affect proposal acceptance. The 32-core M5 Max has an
independent backend and does not inherit the 40-core serving recipes.

Preconverted NVFP4 heads are published as
[`LithosAI/Qwen3.8-27B-DSpark-NVFP4`](https://huggingface.co/LithosAI/Qwen3.8-27B-DSpark-NVFP4)
and [`LithosAI/Qwen3.6-35B-A3B-DSpark-NVFP4`](https://huggingface.co/LithosAI/Qwen3.6-35B-A3B-DSpark-NVFP4).
They contain NVFP4 codes, block scales and tensor scales in standard safetensors.
The initial local cache build only arranges these bytes into the chip's layout;
it does not dequantize or requantize NVFP4 weights. Both exports reproduce the
previously tested packs byte for byte, even with CPU NVFP4 quantization and
dequantization disabled. The Markov embedding and auxiliary parameters retain
their selected source precision. To use BF16 draft matrices, choose the original
upstream checkpoint with `--draft-quantization none`; that option does not restore
BF16 weights from an NVFP4 checkpoint. The release cards include pinned provenance,
validation, and a compatibility patch for the recorded engine base revision.

The default is seven proposals plus one anchor (eight verification rows), or the
checkpoint's smaller supported block. `--draft-block-size` explicitly selects a
different supported proposal count. `load_session` and the generation CLI use
the same DSpark default, with fixed verification unless another rule is requested.
For the 27B model, both NVFP4 and source-precision recipes select the largest
128/4K/8K/16K/32K key not exceeding the prompt
length (128 is also used for shorter prompts). NVFP4 uses its long-context draft recipe
from 4K onward, combined with each context's target recipe from the existing files.
The 35B-A3B uses one configuration across contexts. Selection happens once per request.
`--kernel-config` can override the recipe JSON and
`--kernel-config-key` can pin an entry. Recipes live in the selected chip backend.

The checkpoint must include its tokenizer and chat template.
Experimental support for NVIDIA's Qwen3.6-35B-A3B hybrid MoE, including its
Koopah DSpark pairing and outstanding numerical qualification, is described in
[the model support and validation notes](qwen-hybrid-moe.md).
The server applies that template with an assistant generation prompt and thinking disabled where supported.
Checkpoint resolution and packing happen during startup. GPU loading and Metal compilation happen
on the first chat request. Subsequent requests reuse the session, but GPU weights/engines can
be rebuilt when switching between prefill and decode to fit the device's memory.
`GET /health` reports HTTP process liveness, not GPU readiness. `GET /v1/models` lists the served alias.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"What is the capital of France?"}],"max_completion_tokens":64}'
```

Use the OpenAI Python SDK with a local base URL (install `openai` separately):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="nvidia/Qwen3.8-27B-NVFP4",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_completion_tokens=64,
)
print(response.choices[0].message.content)
```

The APIs support text messages, developer/system instructions, function tools, tool-result
history, greedy or stochastic sampling, output limits and stop strings. Responses also adapts
client-executed custom tools (such as patch input); grammar constraints are not enforced.
Streaming text is emitted as bounded decode batches finish; complete tool calls are validated
and emitted at the end of generation. SSE keep-alives cover compilation and prefill. Client
disconnection stops further work at a prefill-chunk/decode-batch boundary; an active Metal
dispatch finishes first. Non-streaming requests retain the original native pump path.

Defaults are 256 output tokens, greedy decoding, `top_p=1`, seed 0, and thinking disabled.
Sampling changes rebuild the session while preserving the selected draft and kernel recipes.
Usage counts all generated tokens, including EOS and text truncated by stop strings.

Each request supplies the full conversation. Responses is stateless: `previous_response_id`
and background jobs are unsupported. One generation runs at a time; overlapping requests
receive HTTP 429. Images/audio/documents, server-hosted tools, strict JSON-schema decoding,
extended thinking, and grammar-constrained custom tools are not implemented. Invalid tools
and incomplete model tool calls produce errors instead of executable calls. Agents still
apply their own tool permissions. These are protocol adapters, not full cloud API replicas.
The default context capacity is 32768; use `--max-context` to fit the model and device memory.
DSpark reserves another `block_size - 1` positions internally for its attention block (six for
this drafter). Thus `--max-context 33018` uses the same 33024-position capacity as the earlier
27B benchmarks. Changing capacity creates another cache entry rather than modifying an old pack.
Prompt processing uses 128-token chunks by default (`--prefill-chunk-size`), with a separate compiled graph
from single-token decode. Short prompts use smaller prefill buckets. Increasing the chunk size increases
temporary GPU memory; it does not change the context capacity or the decode graph's token bound.

By default the server listens only on localhost. For access through an SSH tunnel, forward port 8000.
Set `LITHOS_METAL_API_KEY` before launch (`LMK_API_KEY` and `MONOLITH_API_KEY` remain aliases) to require `Authorization: Bearer <key>` on `/v1/*`;
use that key in the client. Messages also accepts `x-api-key`. For a shared deployment, terminate TLS at a reverse proxy and keep the
Mac endpoint private. No OpenAI account or OpenAI API key is required to run this local model.

Non-streaming responses expose `X-Lithos-Metal-Decode-Steps`, `X-Lithos-Metal-Decode-GPU-Ms`,
`X-Lithos-Metal-Decode-Step-Ms` and `X-Lithos-Metal-Verify-Tokens`. Step time is the total GPU decode
time divided by the number of rounds, including target verification, acceptance/commit and
draft generation. It excludes packing, compilation and prefill. The verification header
reports the configured width (legacy `X-Monolith-*` names are also returned); the terminal round can be shorter. HTTP wall time includes
all request work and must be reported separately.

Save an ordinary Chat Completions request as `request.json`, then measure both timings
(requires `httpx`; uses `MONOLITH_API_KEY` if set):

```bash
.venv/bin/python tools/bench/serve_latency.py \
  --request request.json --warmup 1 --reps 5 --out /tmp/serve-latency.json
```

Server/cache contract tests (requires `httpx`):
`.venv/bin/python -m pytest tests/contract/test_serve.py tests/contract/test_serving_cache.py`.

## CLI and protocol validation

[M] October 4, 2026, M5 Max 40-core: both published NVFP4 target/draft pairs passed
real HTTP checks for non-streaming and streaming text, XML tool-call conversion, a
Responses tool-result continuation, and streaming Anthropic tool use. Both selected
DSpark automatically and reported eight verification rows. Streamed and non-streamed
greedy text matched for each model. The 27B short text request measured 42.77 ms per
GPU decode round without streaming and 42.83 ms with streaming; these are smoke-test
observations, not a new benchmark suite. Cold compilation and prefill are excluded.

The 35B checks also passed from a fresh offline wheel installation outside the checkout,
with no PyTorch dependency. The official OpenAI and Anthropic Python SDKs reconstructed
streamed text and tool calls against a deterministic HTTP backend. Agent launch plans,
argument forwarding, credential redaction and preservation of OpenCode permissions are
contract-tested; the four interactive agent applications have not all been exercised
end to end. The Homebrew formula passed Ruby syntax checking; `brew install` awaits
publication of the tap and release.

## Measured serving configuration

[M] October 3, 2026: 40-core M5 Max, 48 GB, the checkpoint revisions recorded in
[the DSpark integration report](research/m5max-27b-dspark.md#checkpoints-and-integration),
new automatically created target/NVFP4 draft packs, 33024 internal context positions.
The tuned GDN and full-attention mixers and two-kernel target MLP are selected by the
backend. The draft retains its BF16 Markov embedding.

Complete nonterminal rounds, each verifying eight rows, accepting all seven proposals on
the fixed fixture, committing state and generating the next draft:

| Context | Complete GPU round, median | Minimum–maximum |
|---|---:|---:|
| 128 | 43.21 ms | 42.89–43.45 ms |
| 4K | 44.89 ms | 44.60–45.25 ms |
| 8K | 47.06 ms | 46.92–47.39 ms |
| 16K | 51.37 ms | 51.13–51.52 ms |
| 32K | 57.43 ms | 57.30–57.78 ms |

Each row uses nine measured samples after five warmups, real prefill and restored
StepState/GDN state for each replay, through `tools/bench/dspark_round_latency.py`.
Compilation, prefill and host setup are excluded. The fixture is the integration report's
archived `inputs.json` (SHA-256
`c913ab9975beb8c592556d4e9a63305edfdbb1b89ead4fc24126ad492a753628`).
All replays and split/full execution agree on committed tokens and next drafts. At 128,
new automatic packs also reproduce the old manually prepared packs' tokens/proposals.

For an actual HTTP request with a templated 128-token prompt and 128 output tokens, one
warmup followed by five requests measured **43.04 ms median mean GPU step time**, range
**42.75–43.49 ms**, with 43 decode rounds per request. This meets the requested **<45 ms
at 128 tokens**. The prompt was `Write a numbered list of 20 interesting facts about the
ocean. Additional context:` followed by ` ocean` repeated 99 times. Each request used
greedy decoding and no system message. This acceptance differs from the replay fixture;
the terminal round can be shorter than eight verification rows.

HTTP wall time for those requests was **10.91 seconds median** (10.82–11.00 seconds),
including prefill and the allocation/layout transition between prefill and decode; the
first request took 41.50 seconds including cold setup. Thus the sub-45-ms result describes
GPU rounds, not complete HTTP requests or per-output-token latency. End-to-end serving
still has substantial setup/prefill overhead on this 48-GB device.

The context policy was checked against keeping the 128-token recipe at all intermediate
contexts: those separate runs measured 49.41/54.36/65.45 ms at 4K/8K/16K, versus
44.89/47.06/51.37 ms with context-specific selection. Committed tokens and next drafts
matched exactly on this fixture; these were sequential checks, not paired kernel-fusion
comparisons. No kernel implementation changed for these serving defaults.
