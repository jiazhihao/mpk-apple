"""lithos-metal text/tool server: Chat Completions, Responses and Anthropic Messages."""

from __future__ import annotations

import argparse
import asyncio
import queue
import logging
import os
import secrets
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError


from .serving.protocol import (APIError, ChatRequest, Message, TextPart, anthropic_request,
                               responses_request, parse_completion)
from .serving.events import WireResponse


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

    def complete(self, request, *, on_text=None, on_start=None, cancelled=None):
        from jinja2 import TemplateError
        from .generate import load_session

        messages, tools = request.template_inputs()
        try:
            ids = self.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, add_generation_prompt=True,
                                                     enable_thinking=False, **({'tools': tools} if tools else {}))
        except (ValueError, TemplateError) as exc:
            raise APIError(str(exc), param="messages") from exc
        limit = request.token_limit
        if not ids or len(ids) + limit - 1 > self.max_context:
            raise APIError(f"Prompt ({len(ids)} tokens) plus output budget ({limit}) exceeds context capacity "
                           f"({self.max_context}); reduce messages or max_completion_tokens.",
                           code="context_length_exceeded")
        if on_start:
            on_start(len(ids))
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
            def publish(tokens):
                if on_text and not tools:
                    text, _ = self.visible_text(tokens, request)
                    # Delay the unfinished word and stop-string prefix. This also
                    # avoids emitting a replacement character for partial UTF-8.
                    stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
                    hold = max([len(s) for s in stops] + [1])
                    safe = text[:-hold]
                    boundary = max(safe.rfind(' '), safe.rfind('\n')) + 1
                    on_text(safe[:boundary])
            options = dict(on_tokens=publish, cancelled=cancelled) if on_text is not None else {}
            generation = self.session.generate(ids, limit, **options)
            tokens = generation.tokens
            self.last_metrics = dict(steps=getattr(generation, 'steps', 0),
                decode_gpu_ms=getattr(generation, 'decode_ms', 0.0), wall_ms=(time.perf_counter()-started)*1000,
                verify_tokens=assets.gamma+1 if assets and assets.gamma else 1)
            self.last_metrics['decode_step_ms'] = self.last_metrics['decode_gpu_ms'] / max(1, self.last_metrics['steps'])
            logging.getLogger(__name__).info('Generation: %s', self.last_metrics)
        except ValueError as exc:
            raise APIError(str(exc)) from exc
        content, finish = self.visible_text(tokens, request)
        return content, finish, len(ids), len(tokens)

    def visible_text(self, tokens, request):
        eos = self.session.eos
        eos_ids = {eos} if isinstance(eos, int) else set(eos)
        end = next((i for i, token in enumerate(tokens) if token in eos_ids), None)
        finish = "stop" if end is not None else "length"
        visible = tokens[:end] if end is not None else tokens
        content = self.tokenizer.decode(visible, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        stops = [request.stop] if isinstance(request.stop, str) else request.stop or []
        positions = [content.index(s) for s in stops if s in content]
        if positions:
            content, finish = content[:min(positions)], "stop"
        return content, finish


def create_app(backend, model_name, api_key=None):
    app = FastAPI(title="lithos-metal", version="0.1.0")
    lock = threading.Lock()
    created = int(time.time())

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        kind = {401: "authentication_error", 429: "rate_limit_error", 500: "server_error"}.get(
            exc.status, "invalid_request_error")
        if request.url.path.startswith('/v1/messages'):
            return JSONResponse(status_code=exc.status, content={'type': 'error', 'error': {'type': kind, 'message': exc.message}})
        return JSONResponse(status_code=exc.status, content={"error": {
            "message": exc.message, "type": kind, "param": exc.param, "code": exc.code}})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        error = exc.errors()[0]
        param = ".".join(str(p) for p in error["loc"] if p != "body")
        return await api_error(request, APIError(f"{param}: {error['msg']}", param=param))

    async def authorize(request: Request):
        bearer = request.headers.get('authorization', '').removeprefix('Bearer ')
        key = request.headers.get('x-api-key', '') if request.url.path.startswith('/v1/messages') else ''
        if api_key and not any(secrets.compare_digest(value.encode(), api_key.encode()) for value in (bearer, key)):
            raise APIError("Invalid API key", 401, "invalid_api_key")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models", dependencies=[Depends(authorize)])
    async def models():
        return {"object": "list", "data": [{"id": model_name, "object": "model", "created": created,
            "owned_by": "lithos-metal", "context_window": getattr(backend, 'max_context', 32768)}]}

    def validate_model(request):
        if request.model != model_name:
            raise APIError(f"Unknown model; use {model_name!r}", 404, "model_not_found", "model")

    def execute(request, wire, **options):
        content, finish, prompt_tokens, completion_tokens = backend.complete(request, **options)
        message, finish = parse_completion(content, request, finish)
        return wire.body(message, finish, prompt_tokens, completion_tokens)

    def dispatch(request, protocol, custom=()):
        validate_model(request)
        if not lock.acquire(blocking=False):
            raise APIError("The model is busy; retry after the current request finishes", 429, "model_busy")
        wire = WireResponse(protocol, model_name, custom)
        if request.stream:
            return stream(request, wire)
        try:
            body = execute(request, wire)
            metrics = dict(getattr(backend, 'last_metrics', {}))
        except APIError:
            raise
        except Exception as exc:
            logging.getLogger(__name__).exception("Generation failed")
            raise APIError("Generation failed; see server logs", 500, "generation_failed") from exc
        finally:
            lock.release()
        headers = {f'X-{brand}-{name}': str(metrics[key]) for brand in ('Lithos-Metal', 'LMK', 'Monolith') for name, key in (
            ('Decode-Steps', 'steps'), ('Decode-GPU-Ms', 'decode_gpu_ms'),
            ('Decode-Step-Ms', 'decode_step_ms'), ('Verify-Tokens', 'verify_tokens')) if key in metrics}
        return JSONResponse(headers=headers, content=body)

    def stream(request, wire):
        events = queue.Queue()
        cancelled = threading.Event()
        def worker():
            try:
                options = dict(on_text=lambda text: events.put(('text', text)),
                               on_start=lambda count: events.put(('start', count)),
                               cancelled=cancelled.is_set) if isinstance(backend, Backend) else {}
                result = execute(request, wire, **options)
                if not options:
                    usage = result['usage']
                    events.put(('start', usage.get('input_tokens', usage.get('prompt_tokens', 0))))
                events.put(('result', result))
            except APIError as exc:
                events.put(('error', (exc.message, exc.code)))
            except Exception:
                logging.getLogger(__name__).exception('Streaming generation failed')
                events.put(('error', ('Generation failed; see server logs', 'generation_failed')))
            finally:
                lock.release()
        # A worker owns the lock from here, even if the response is never consumed.
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        async def generate():
            try:
                last_ping = time.monotonic()
                while True:
                    if events.empty():
                        if not thread.is_alive() and events.empty():
                            yield wire.error('Generation worker terminated', 'generation_failed')
                            break
                        await asyncio.sleep(.01)
                        if time.monotonic() - last_ping > 5:
                            yield ': keep-alive\n\n'
                            last_ping = time.monotonic()
                        continue
                    kind, value = events.get_nowait()
                    if kind == 'start':
                        wire.input_tokens = value
                        for event in wire.start():
                            yield event
                    elif kind == 'text':
                        if not value.startswith(wire.text):
                            raise RuntimeError('Decoded text changed after streaming')
                        for event in wire.delta(value[len(wire.text):]):
                            yield event
                    elif kind == 'error':
                        yield wire.error(*value)
                        break
                    else:
                        if wire.protocol == 'chat':
                            text = value['choices'][0]['message'].get('content') or ''
                        elif wire.protocol == 'messages':
                            text = ''.join(b['text'] for b in value['content'] if b['type'] == 'text')
                        else:
                            text = ''.join(p['text'] for b in value['output'] if b['type'] == 'message' for p in b['content'])
                        for event in wire.delta(text[len(wire.text):]):
                            yield event
                        for event in wire.finish(value, bool((request.stream_options or {}).get('include_usage'))):
                            yield event
                        break
            finally:
                cancelled.set()
        return StreamingResponse(generate(), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

    @app.post("/v1/chat/completions", dependencies=[Depends(authorize)])
    def chat(request: ChatRequest):
        return dispatch(request, 'chat')

    def convert(body, converter):
        try:
            return converter(body)
        except (ValidationError, KeyError, TypeError, AttributeError) as exc:
            raise APIError(f'Invalid request: {exc}') from exc

    @app.post('/v1/messages', dependencies=[Depends(authorize)])
    def messages(body: dict):
        return dispatch(convert(body, anthropic_request), 'messages')

    @app.post('/v1/messages/count_tokens', dependencies=[Depends(authorize)])
    def count_tokens(body: dict):
        request = convert({**body, 'max_tokens': 1}, anthropic_request)
        validate_model(request)
        messages, tools = request.template_inputs()
        ids = backend.tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                add_generation_prompt=True, enable_thinking=False, **({'tools': tools} if tools else {}))
        return {'input_tokens': len(ids)}

    @app.post('/v1/responses', dependencies=[Depends(authorize)])
    def responses(body: dict):
        request, custom = convert(body, responses_request)
        return dispatch(request, 'responses', custom)

    return app


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="lithos-metal serve", description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local checkpoint path")
    draft = parser.add_mutually_exclusive_group()
    draft.add_argument("--draft", "--drafter", dest="draft", help="Override the automatically selected DSpark head (Hub ID or local path)")
    draft.add_argument("--no-draft", action='store_true', help="Disable automatic DSpark speculative decoding")
    parser.add_argument("--draft-kind", choices=['dspark'], default='dspark')
    parser.add_argument("--draft-block-size", type=int, help="Draft proposals per round (default: up to seven, plus one target anchor)")
    parser.add_argument("--pack", help="Local pack-cache directory (default: $XDG_CACHE_HOME/lithos-metal/packs; reuses legacy cache); existing packs also accepted")
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
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--prefill-chunk-size", type=int, default=128, help="Prompt tokens per prefill pass (default: 128)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    from .models.catalog import default_draft
    if not args.draft and not args.no_draft:
        args.draft = default_draft(args.model)
    if args.max_context < 1:
        parser.error("--max-context must be positive")
    if args.prefill_chunk_size < 1:
        parser.error("--prefill-chunk-size must be positive")
    if args.draft_block_size is not None and args.draft_block_size < 1:
        parser.error('--draft-block-size must be positive')
    if not args.draft and (args.draft_pack or args.draft_revision or args.draft_block_size is not None or args.kernel_config or args.kernel_config_key
                          or args.draft_quantization != 'auto'):
        parser.error('draft options require --draft or a target with an automatic DSpark head')
    return args


def main(argv=None):
    import uvicorn
    from .serving.setup import prepare

    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    assets = prepare(args)
    api_key = os.environ.get("LITHOS_METAL_API_KEY") or os.environ.get("LMK_API_KEY") or os.environ.get("MONOLITH_API_KEY")
    backend = Backend(str(assets.model_dir), str(assets.pack_dir), args.max_context, args.prefill_chunk_size, assets=assets)
    model_name = args.served_model_name or (assets.model_dir.name if Path(args.model).expanduser().exists() else args.model)
    app = create_app(backend, model_name, api_key)
    logging.getLogger(__name__).info('lithos-metal ready: http://%s:%s — model=%s; DSpark=%s; verification rows=%s',
                                   args.host, args.port, model_name, args.draft or 'disabled', assets.gamma + 1)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
