"""Tests for the LLM provider abstraction (no network: httpx.MockTransport)."""

from __future__ import annotations

import json

import httpx
import pytest

from agent_smith.core.config import ModelConfig, build_model_config
from agent_smith.llm import (
    ChatMessage,
    ChatResponse,
    LLMAPIError,
    LLMConnectionError,
    LLMProvider,
    OpenAICompatProvider,
    create_provider,
)

MESSAGES = [ChatMessage(role="user", content="hi")]
OK_BODY = {
    "choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 7, "completion_tokens": 3},
}


def make_provider(handler, url="https://p.example/v1", api_key="k") -> OpenAICompatProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = ModelConfig(model_name="m/x", provider_url=url, temperature=0.3, max_tokens=50)
    return OpenAICompatProvider(config, api_key=api_key, client=client)


def test_request_shape_and_response_parsing():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    resp = make_provider(handler, url="https://p.example/v1/").chat(MESSAGES, stop=["<end_code>"])

    assert seen["url"] == "https://p.example/v1/chat/completions"
    assert seen["auth"] == "Bearer k"
    assert seen["body"] == {
        "model": "m/x",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.3,
        "max_tokens": 50,
        "stop": ["<end_code>"],
    }
    assert resp.content == "hello"
    assert (resp.input_tokens, resp.output_tokens) == (7, 3)
    assert resp.api_url == "https://p.example/v1/"
    assert resp.model_name == "m/x"
    assert resp.request_time_ms >= 0
    assert resp.finish_reason == "stop"


def test_no_api_key_omits_auth_header(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["headers"] = request.headers
        return httpx.Response(200, json=OK_BODY)

    make_provider(handler, api_key=None).chat(MESSAGES)
    assert "authorization" not in seen["headers"]


@pytest.mark.parametrize("status", [401, 429, 500])
def test_http_errors_raise_api_error(status):
    provider = make_provider(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(LLMAPIError) as exc_info:
        provider.chat(MESSAGES)
    assert exc_info.value.status_code == status


@pytest.mark.parametrize("body", [{"choices": []}, {"oops": 1}])
def test_malformed_payload_raises_api_error(body):
    provider = make_provider(lambda r: httpx.Response(200, json=body))
    with pytest.raises(LLMAPIError):
        provider.chat(MESSAGES)


def test_invalid_json_raises_api_error():
    provider = make_provider(lambda r: httpx.Response(200, text="<html>"))
    with pytest.raises(LLMAPIError):
        provider.chat(MESSAGES)


def test_network_failure_raises_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    with pytest.raises(LLMConnectionError):
        make_provider(handler).chat(MESSAGES)


def test_factory_switches_provider_at_runtime():
    a = create_provider(build_model_config("m/a", "https://a.example/v1"))
    b = create_provider(build_model_config("m/b", "https://b.example/v1"))
    assert (a.api_url, a.model_name) == ("https://a.example/v1", "m/a")
    assert (b.api_url, b.model_name) == ("https://b.example/v1", "m/b")


def test_factory_unknown_kind():
    with pytest.raises(ValueError, match="Unknown provider kind"):
        create_provider(build_model_config("m"), kind="nope")


def test_custom_backend_satisfies_interface():
    class Fake(LLMProvider):
        api_url = "fake://"
        model_name = "fake"

        def chat(self, messages, *, stop=None, temperature=None, max_tokens=None):
            return ChatResponse(content="x", api_url=self.api_url, model_name=self.model_name)

    assert Fake().chat(MESSAGES).content == "x"
