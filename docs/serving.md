# Local Chat Completions endpoint

Install the optional server dependencies in the existing environment (the Metal runtime must already be built):

```bash
.venv/bin/python -m pip install -e '.[serve]'
```

Pack the checkpoint once if needed, then start the server:

```bash
.venv/bin/python tools/pack_weights.py --model ~/models/Qwen3.5-0.8B --out /tmp/qwen35-pack
.venv/bin/python -m monolith.serve \
  --model ~/models/Qwen3.5-0.8B --pack /tmp/qwen35-pack \
  --served-model-name qwen3.5 --host 127.0.0.1 --port 8000
```

An existing compatible pack can be passed directly. The checkpoint must include its tokenizer and chat template.
The server applies that template with an assistant generation prompt and thinking disabled where supported.
Model loading and Metal compilation happen on the first chat request; subsequent requests reuse the session.
`GET /health` reports HTTP process liveness, not GPU readiness. `GET /v1/models` lists the served alias.

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.5","messages":[{"role":"user","content":"What is the capital of France?"}],"max_completion_tokens":64}'
```

Use the OpenAI Python SDK with a local base URL (install `openai` separately):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
response = client.chat.completions.create(
    model="qwen3.5",
    messages=[{"role": "user", "content": "What is the capital of France?"}],
    max_completion_tokens=64,
)
print(response.choices[0].message.content)
```

This implements the text-only, non-streaming subset of the
[Chat Completions API](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create).
Supported fields: `model`, `messages` (system/user/assistant, strings or text-part lists),
`max_completion_tokens` or `max_tokens`, `temperature`, `top_p`, `seed`, `stop`, `n=1`, `stream=false`.
Defaults are 256 output tokens, greedy decoding (`temperature=0`), `top_p=1`, and seed 0.
Sampling settings are compiled into the engine; changing them rebuilds the cached session.
The response includes an assistant message, finish reason, and tokenizer-based usage counts.
Stop strings truncate the final text after generation; usage counts all generated tokens, including EOS
and any text discarded by a stop string.

Each request supplies the full conversation; GPU sequence state resets between requests. One generation runs
at a time, with overlapping requests receiving HTTP 429. Run one server worker per GPU.
Invalid or unsupported fields return an OpenAI-shaped HTTP 400 error instead of being silently ignored.
Streaming, tool calls, images, JSON-schema output and the Responses API are not implemented.
The default context capacity is 4096; adjust with `--max-context` to fit the checkpoint and available memory.

By default the server listens only on localhost. For access through an SSH tunnel, forward port 8000.
Set `MONOLITH_API_KEY` before launch to require `Authorization: Bearer <key>` on `/v1/*`;
use that key in the client. For a shared deployment, terminate TLS at a reverse proxy and keep the
Mac endpoint private. No OpenAI account or OpenAI API key is required to run this local model.

Server contract tests: `.venv/bin/python -m pytest tests/contract/test_serve.py` (also requires `httpx`).
