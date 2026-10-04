"""Single-model, non-streaming Chat Completions server. Run with ``python -m monolith.serve``."""

from __future__ import annotations

import argparse
import logging
import os
import secrets
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextPart(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["text"]
    text: str


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    role: Literal["system", "user", "assistant"]
    content: str | list[TextPart]


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    model: str
    messages: list[Message] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    seed: int = Field(default=0, ge=0, le=2**64 - 1)
    stop: str | list[str] | None = None
    stream: Literal[False] = False
    n: Literal[1] = 1

    @model_validator(mode="after")
    def validate_options(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            raise ValueError("Pass only one of max_tokens and max_completion_tokens")
        stops = [self.stop] if isinstance(self.stop, str) else self.stop or []
        if len(stops) > 4 or any(not s for s in stops):
            raise ValueError("stop must contain at most four nonempty strings")
        return self

    @property
    def token_limit(self):
        return self.max_completion_tokens or self.max_tokens or 256


class APIError(Exception):
    def __init__(self, message, status=400, code="invalid_request_error", param=None):
        self.message, self.status, self.code, self.param = message, status, code, param


class Backend:
    """One cached session; sampling changes rebuild its compiled programs, never duplicate model residency."""

    def __init__(self, model_dir, pack_dir, max_context=4096, prefill_chunk_size=128, *, assets=None):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("The checkpoint must provide a chat template")
        self.model_dir, self.pack_dir, self.max_context = model_dir, pack_dir, max_context
        self.prefill_chunk_size = prefill_chunk_size
        self.session, self.sampling = None, None
        self.assets = assets
        self.last_metrics = {}

    def complete(self, request):
        from jinja2 import TemplateError
        from .generate import load_session

        messages = [{"role": m.role, "content": m.content if isinstance(m.content, str)
                     else "".join(p.text for p in m.content)} for m in request.messages]
        try:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                                     enable_thinking=False)
        except (ValueError, TemplateError) as exc:
            raise APIError(str(exc), param="messages") from exc
        limit = request.token_limit
        if not ids or len(ids) + limit - 1 > self.max_context:
            raise APIError(f"Prompt ({len(ids)} tokens) plus output budget ({limit}) exceeds context capacity "
                           f"({self.max_context}); reduce messages or max_completion_tokens.",
                           code="context_length_exceeded")
        assets = getattr(self, 'assets', None)
        recipe_key, options = assets.options(len(ids)) if assets else (None, {'max_context': self.max_context})
        sampling = (request.temperature, request.top_p, request.seed, recipe_key)
        if self.session is None or sampling != self.sampling:
            self.session = None
            self.session = load_session(self.model_dir, self.pack_dir, **options,
                                       temperature=request.temperature, top_p=request.top_p, seed=request.seed,
                                       autotune=False, prefill_chunk_size=self.prefill_chunk_size)
            self.sampling = sampling
        try:
            started = time.perf_counter()
            generation = self.session.generate(ids, limit)
            tokens = generation.tokens
            self.last_metrics = dict(steps=getattr(generation, 'steps', 0),
                decode_gpu_ms=getattr(generation, 'decode_ms', 0.0), wall_ms=(time.perf_counter()-started)*1000,
                verify_tokens=assets.gamma+1 if assets and assets.gamma else 1)
            self.last_metrics['decode_step_ms'] = self.last_metrics['decode_gpu_ms'] / max(1, self.last_metrics['steps'])
            logging.getLogger(__name__).info('Generation: %s', self.last_metrics)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        eos = self.session.eos
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        finish = "stop" if end is not None else "length"
        visible = tokens[:end] if end is not None else tokens
        content = self.tokenizer.decode(visible, skip_special_tokens=True)
        stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
        positions = [content.index(s) for s in stops if s in content]
        if positions:
            content, finish = content[:min(positions)], "stop"
        return content, finish, len(ids), len(tokens)


def create_app(backend, model_name, api_key=None):
    app = FastAPI(title="Monolith Chat", version="0.1")
    lock = threading.Lock()
    created = int(time.time())

    @app.exception_handler(APIError)
    async def api_error(_request, exc):
        kind = {401: "authentication_error", 429: "rate_limit_error", 500: "server_error"}.get(
            exc.status, "invalid_request_error")
        return JSONResponse(status_code=exc.status, content={"error": {
            "message": exc.message, "type": kind, "param": exc.param, "code": exc.code}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        param = ".".join(str(p) for p in error["loc"] if p != "body")
        return await api_error(request, APIError(f"{param}: {error['msg']}", param=param))

    async def authorize(request: Request):
        if api_key and not secrets.compare_digest(request.headers.get("authorization", "").encode(),
                                                 f"Bearer {api_key}".encode()):
            raise APIError("Invalid API key", 401, "invalid_api_key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": created,
                                             "owned_by": "monolith"}]}

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    def chat(request: ChatRequest):
        if request.model != model_name:
            raise APIError(f"Unknown model; use {model_name!r}", 404, "model_not_found", "model")
        if not lock.acquire(blocking=False):
            raise APIError("The model is busy; retry after the current request finishes", 429, "model_busy")
        try:
            content, finish, prompt_tokens, completion_tokens = backend.complete(request)
            metrics = dict(getattr(backend, 'last_metrics', {}))
        except APIError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Generation failed")
            raise APIError("Generation failed; see server logs", 500, "generation_failed") from exc
        finally:
            lock.release()
        headers = {f'X-Monolith-{name}': str(metrics[key]) for name, key in (
            ('Decode-Steps', 'steps'), ('Decode-GPU-Ms', 'decode_gpu_ms'),
            ('Decode-Step-Ms', 'decode_step_ms'), ('Verify-Tokens', 'verify_tokens')) if key in metrics}
        return JSONResponse(headers=headers, content={"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                                  "finish_reason": finish, "logprobs": None}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}})

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local checkpoint path")
    parser.add_argument("--draft", "--drafter", dest="draft", help="DSpark Hugging Face repo ID or local checkpoint path")
    parser.add_argument("--draft-kind", choices=['dspark'], default='dspark')
    parser.add_argument("--draft-block-size", type=int, help="Draft proposals per round (default: up to seven, plus one target anchor)")
    parser.add_argument("--pack", help="Local pack-cache directory (default: $XDG_CACHE_HOME/monolith/packs); existing packs also accepted")
    parser.add_argument("--draft-pack", help="Optional separate draft cache or existing draft pack")
    parser.add_argument("--draft-quantization", choices=['auto', 'none', 'nvfp4'], default='auto',
                        help="auto uses validated chip-specific NVFP4 draft recipes when available; otherwise source precision")
    parser.add_argument("--revision", help="Target Hugging Face revision")
    parser.add_argument("--draft-revision", help="Draft Hugging Face revision")
    parser.add_argument("--download-dir", help="Hugging Face download cache directory")
    parser.add_argument("--local-files-only", action='store_true', help="Resolve Hub IDs from the local Hub cache only")
    parser.add_argument("--kernel-config", help="Optional explicit target/draft recipe JSON; defaults to matching chip recipes")
    parser.add_argument("--kernel-config-key", help="Pin a context key in the selected recipe map")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=128, help="Prompt tokens per prefill pass (default: 128)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be positive")
    if args.draft_block_size is not None and args.draft_block_size < 1:
        parser.error('--draft-block-size must be positive')
    if not args.draft and (args.draft_pack or args.draft_revision or args.draft_block_size is not None or args.kernel_config or args.kernel_config_key
                          or args.draft_quantization != 'auto'):
        parser.error('draft options require --draft')
    return args


def main(argv=None):
    import uvicorn
    from .serving.setup import prepare

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    assets = prepare(args)
    api_key = os.environ.get("MONOLITH_API_KEY")
    backend = Backend(str(assets.model_dir), str(assets.pack_dir), args.max_context, args.prefill_chunk_size, assets=assets)
    app = create_app(backend, args.served_model_name or Path(args.model).name, api_key)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
