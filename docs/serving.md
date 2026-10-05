# lithos-metal serving and coding agents

Install lithos-metal with `brew install lithos-ai/tap/lithos-metal`, or use
`pip install '.[serve]'` from a checkout to build the Metal runtime
and bundle the kernel sources. The [installation guide](installation.md) covers the
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

For Claude Code, install the client with `brew install --cask claude-code`, then run
`lithos-metal claude`. The launcher selects the local target model for all model roles,
declares the server's context limit, disables unsupported experimental request fields
and thinking, and omits the attribution prefix so separate sessions can reuse the prompt
cache. The Messages adapter rejects unsupported effort with a capability-specific error;
Claude Code retries without effort. Tool permissions retain the client's settings.
The local Qwen model generates the answers, including when Claude Code's interface or
system prompt refers to Claude.

Claude Code 2.1.285 was tested against the local Qwen3.8-27B NVFP4 + DSpark endpoint:
streamed text and a `Read` tool/result continuation both completed successfully. Its
standard prompt carried 22,348 input tokens on this checkout. First text took 110 s
uncached and 53 s on a second launch (14,993 cached tokens); partial prefix reuse still
leaves substantial prefill work. These client integration checks do not establish the
sub-second first-token target for long prompts.

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
Checkpoint resolution, packing, and generation warmup happen before the server accepts
requests. Warmup compiles the default sampling configuration's prefill/decode programs for
each selected context recipe, then exercises allocation and reuse with short generations.
Only one session's GPU weight buffers stay resident. `--no-warmup` defers this cost to
requests. A different sampling configuration can still need additional compilation.
Compiled pipelines and a bounded set of CPU programs survive engine replacement.
`GET /health` reports process liveness after startup;
`GET /v1/models` lists the served alias.

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
Streaming text is emitted after each speculative round, including requests that advertise
tools. Only verified target tokens are published. Incomplete UTF-8, possible stop-string
suffixes, and tool markers are held until they can be decoded safely. Complete tool calls
are validated and emitted as structured events at the end of generation. SSE keep-alives
cover compilation and prefill; empty events are not evidence of a first output token. Client
disconnection stops further work at a prefill-chunk/decode-round boundary; an active Metal
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
Large prompts use 128-token prefill chunks by default (`--prefill-chunk-size`). With an explicit
fixed-verification decoder recipe, short prompts and cached tails of up to that length reuse
the resident eight-row graph. This avoids remapping the prefill and decode weight layouts.
Large prompts switch to the decoder for their final input rows before publishing any text,
so that layout transition does not interrupt the output stream.
The server keeps at most two exact-token CPU checkpoints within a budget of one eighth of
Metal's recommended working set, capped at 4 GiB. This permits a complete 22K-token
Qwen/DSpark checkpoint (about 2.09 GiB), including both GDN recurrent slots and the occupied
target/draft KV rows. Earlier text blocks of the last message are eligible for caching;
Claude Code's project context is therefore retained while its final question is replayed.
The adapter preserves text-block boundaries without changing the rendered prompt. A cache
hit still requires an exact token-prefix match, including tool definitions and project
context. Plain string messages retain the message-boundary fallback. Shared system/tool
prefixes are retained when refreshing a conversation checkpoint. Short prompts are replayed
instead of copied into the cache. Cache contents are local to the process and are lost on
restart. Cache misses, long uncached prompts, and recipe changes can still take more than a
second; streaming alone cannot remove prefill time. Increasing the prefill chunk size increases
temporary GPU memory without changing context capacity or verification width.

Immutable weight files use shared read-only mappings, and the runtime declares weights and
parameters as read-only Metal resources. This reduces first-use residency overhead when
prefill and decode require different pack layouts. A sufficiently short cached suffix runs
entirely on the resident decoder and avoids both layout transitions. Per-chunk diagnostics
separate GPU time from host encoding, command-buffer commit and completion wait; wait time
overlaps GPU execution and must not be added to GPU time. `tools/bench/warm_start.py` runs
a captured Messages or Chat request with these diagnostics. Its
`--message-boundary-control` option reproduces the previous cache boundary with identical
prompt tokens. Startup compilation is outside its request timings.

The warm-start fix is validated by exact-prefix contract tests and GPU continuation
checks for the deep project checkpoint and the earlier system/tool fallback. The runtime
also checks shared immutable weight mappings across engine recreation. Development
measurements with a 22572-token Claude Code prompt reused 22550 tokens, leaving 22
for prefill. That development checkout also contained separate prefill kernel tuning;
its 232–550 ms backend first-text measurements are preliminary, exclude HTTP/client
overhead, and were collected while another GPU task was active. They are not a latency
guarantee or an isolated benchmark of this PR. Kernel tuning and raw result files are
not included here; an isolated endpoint benchmark remains outstanding.

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

Measure first nonempty streamed text and output cadence separately:

```bash
.venv/bin/python tools/bench/serve_stream_latency.py \
  --request request.json --reps 3 --tokenizer /path/to/local/checkpoint \
  --out /tmp/stream-latency.json
```

The first request is retained in the results, rather than discarded as warmup. With
`--tokenizer`, visible throughput excludes EOS and the first text event. The separate
usage-based estimate can include EOS or hidden tool tokens and must not be reported as
visible text throughput. Actual throughput depends on DSpark acceptance; a fast GPU round
does not guarantee 100 output tokens/second on every prompt.

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

[M] October 4, 2026 streaming update, 40-core M5 Max, 48 GB, Qwen3.8-27B-NVFP4
with its published NVFP4 DSpark head and seven proposals plus anchor. After default
startup warmup, sequential greedy HTTP requests measured:

| Request | Input tokens | First nonempty text | Visible output tokens/s |
|---|---:|---:|---:|
| Short identity question, no cached prompt | 21 | 135–162 ms | 107–108 |
| Ocean facts, no cached prompt | 128 | 695–706 ms | 66–67 |
| OpenCode identity question, shared prefix cached | 7,460 | 237 ms | 70 |
| OpenCode merge-sort answer, shared prefix cached | 7,468 | 219–362 ms | 127 |
| First OpenCode identity question, uncached | 7,460 | 31,899 ms | 68 |

The short and 128-token cases have three samples each; the coding case has two;
the cold/cached OpenCode identity cases have one each. The OpenCode requests advertise
ten tools and share a 7,436-token system/tool prefix. The coding answer is capped at
256 output tokens. Throughput retokenizes visible text with the target tokenizer,
excludes EOS and the first event, and measures from the first through last text event.
Maximum gaps between text events were under 49 ms, including the uncached request.
Cold compilation is now paid before readiness and the prefill/decode layout transition
happens before first text. Long uncached prefill remains slow: these results do **not**
establish sub-second first text or 100 tokens/s for every request. Acceptance-dependent
throughput and cold prompt latency must be reported separately.

An end-to-end OpenCode 1.18.34 check subsequently emitted a larger 14,265-token
request. Replaying that captured request after the cache-budget fix measured
59.88 seconds to first text uncached and 400 ms with the shared prefix cached.
Changing the user question to the merge-sort task reused 14,241 prefix tokens;
the resulting 14,273-token requests measured 256–264 ms to first text and
115 visible output tokens/s (two 256-output-token samples). Text-event gaps were
under 61 ms. These HTTP timings exclude OpenCode process startup. The actual
OpenCode command also completed successfully with the local provider.

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
