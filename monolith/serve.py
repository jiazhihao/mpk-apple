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

    def __init__(self, model_dir, pack_dir, max_context=4096, prefill_chunk_size=128):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        if not self.tokenizer.chat_template:
            raise ValueError("The checkpoint must provide a chat template")
        self.model_dir, self.pack_dir, self.max_context = model_dir, pack_dir, max_context
        self.prefill_chunk_size = prefill_chunk_size
        self.session, self.sampling = None, None

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
        sampling = (request.temperature, request.top_p, request.seed)
        if self.session is None or sampling != self.sampling:
            self.session = None
            self.session = load_session(self.model_dir, self.pack_dir, max_context=self.max_context,
                                       temperature=request.temperature, top_p=request.top_p, seed=request.seed,
                                       autotune=False, prefill_chunk_size=self.prefill_chunk_size)
            self.sampling = sampling
        try:
            tokens = self.session.generate(ids, limit).tokens
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        eos = self.session.eos
        finish = "stop" if eos in tokens else "length"
        visible = tokens[:tokens.index(eos)] if eos in tokens else tokens
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
        except APIError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Generation failed")
            raise APIError("Generation failed; see server logs", 500, "generation_failed") from exc
        finally:
            lock.release()
        return {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion", "created": int(time.time()),
                "model": model_name, "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                                  "finish_reason": finish, "logprobs": None}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens}}

    return app


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local checkpoint directory, including its chat template")
    parser.add_argument("--pack", required=True, help="Packed weights directory")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--prefill-chunk-size", type=int, default=128, help="Prompt tokens per prefill pass (default: 128)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be positive")
    api_key = os.environ.get("MONOLITH_API_KEY")
    backend = Backend(args.model, args.pack, args.max_context, args.prefill_chunk_size)
    app = create_app(backend, args.served_model_name or Path(args.model).name, api_key)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
