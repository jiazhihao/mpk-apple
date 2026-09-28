"""HTTP contract and state isolation for the optional serving extra; no GPU needed."""

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from monolith.serve import Backend, create_app


def payload(**options):
    return {"model": "test-model", "messages": [{"role": "user", "content": "Hello"}], **options}


def test_response_and_auth():
    backend = SimpleNamespace(complete=lambda request: ("Hello!", "stop", 5, 2))
    client = TestClient(create_app(backend, "test-model", "secret"))
    assert client.get("/v1/models").status_code == 401
    assert client.post("/v1/chat/completions", json=payload()).status_code == 401
    client.headers["Authorization"] = "Bearer secret"
    assert client.get("/v1/models").json()["data"][0]["id"] == "test-model"
    result = client.post("/v1/chat/completions", json=payload()).json()
    assert result["object"] == "chat.completion"
    assert result["choices"][0]["message"] == {"role": "assistant", "content": "Hello!"}
    assert result["usage"] == {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}


@pytest.mark.parametrize("options", [
    {"messages": []}, {"messages": [{"role": "tool", "content": "no"}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "x"}}]}]},
    {"max_tokens": 0}, {"max_tokens": 2, "max_completion_tokens": 3},
    {"temperature": -1}, {"top_p": 0}, {"stop": ""}, {"stop": ["a"] * 5},
    {"stream": True}, {"n": 2}, {"tools": []}, {"max_tokens": True},
])
def test_reject_unsupported_and_invalid_requests(options):
    def unexpected(_request):
        pytest.fail("Invalid request reached the GPU backend")
    client = TestClient(create_app(SimpleNamespace(complete=unexpected), "test-model"))
    response = client.post("/v1/chat/completions", json=payload(**options))
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert client.post("/v1/chat/completions", json=payload(model="missing")).status_code == 404


def test_busy_and_failed_requests_release_lock():
    entered, release = threading.Event(), threading.Event()

    def complete(_request):
        entered.set()
        assert release.wait(5)
        raise RuntimeError("private filesystem details")

    backend = SimpleNamespace(complete=complete)
    client = TestClient(create_app(backend, "test-model"))
    responses = []
    worker = threading.Thread(target=lambda: responses.append(client.post("/v1/chat/completions", json=payload())))
    worker.start()
    try:
        assert entered.wait(5)
        assert client.get("/health").status_code == 200
        assert client.post("/v1/chat/completions", json=payload()).status_code == 429
    finally:
        release.set()
        worker.join(5)
    assert responses[0].status_code == 500
    assert "private filesystem" not in responses[0].text
    backend.complete = lambda request: ("Recovered", "length", 1, 1)
    assert client.post("/v1/chat/completions", json=payload()).status_code == 200


@pytest.mark.parametrize("eos", [99, [98, 99], (98, 99)])
def test_template_sampling_context_and_stop(monkeypatch, eos):
    from monolith import generate

    calls, prompts, decoded = [], [], []
    tokens = [10, 11, 99]

    def load(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(eos=eos, generate=lambda ids, n: SimpleNamespace(tokens=tokens[:n]))

    def decode(ids, **kwargs):
        decoded.append(ids)
        return "Hello END more" if 11 in ids else "Hello"

    def template(messages, **kwargs):
        prompts.append((messages, kwargs))
        return [1, 2, 3] if kwargs.get("return_dict") is False else {"input_ids": [1, 2, 3]}

    monkeypatch.setattr(generate, "load_session", load)
    backend = Backend.__new__(Backend)
    backend.model_dir, backend.pack_dir, backend.max_context = "model", "pack", 8
    backend.prefill_chunk_size = 256
    backend.session, backend.sampling = None, None
    backend.tokenizer = SimpleNamespace(apply_chat_template=template, decode=decode)
    client = TestClient(create_app(backend, "test-model"))
    response = client.post("/v1/chat/completions", json=payload(max_tokens=3, stop=" END")).json()
    assert response["choices"][0]["message"]["content"] == "Hello"
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["usage"]["completion_tokens"] == 3
    assert decoded[-1] == [10, 11]
    response = client.post("/v1/chat/completions", json=payload(max_tokens=3)).json()
    assert response["choices"][0]["finish_reason"] == "stop"
    assert prompts[0][1]["add_generation_prompt"] is True
    assert prompts[0][1]["enable_thinking"] is False
    client.post("/v1/chat/completions", json=payload(max_completion_tokens=2))
    assert len(calls) == 1
    assert calls[0]["prefill_chunk_size"] == 256
    response = client.post("/v1/chat/completions", json=payload(max_tokens=2, temperature=0.7, top_p=0.9, seed=42))
    assert response.json()["choices"][0]["finish_reason"] == "length"
    assert calls[-1]["temperature"] == 0.7 and calls[-1]["top_p"] == 0.9 and calls[-1]["seed"] == 42
    assert len(calls) == 2
    response = client.post("/v1/chat/completions", json=payload(max_tokens=7))
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_length_exceeded"
    assert len(calls) == 2
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    client.post("/v1/chat/completions", json=payload(max_tokens=2, messages=[{"role": "user", "content": parts}]))
    assert prompts[-1][0] == [{"role": "user", "content": "ab"}]
