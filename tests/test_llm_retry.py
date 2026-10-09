"""Retry policy, error classification and request counting (no network, fake sleep)."""

from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from agent_smith.core.config import ModelConfig, RetryPolicy
from agent_smith.llm import (
    AllProvidersExhaustedError,
    ChatMessage,
    FailoverProvider,
    FailureKind,
    LLMAPIError,
    LLMError,
    LLMRetriesExhaustedError,
    ProviderChain,
)

MSGS = [ChatMessage(role="user", content="hi")]
OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def make(monkeypatch, handler, keys="k1,k2", **cfg):
    monkeypatch.setenv("RETRY_KEYS", keys)
    config = ModelConfig(
        model_name="m",
        provider_url="https://a",
        api_key_env_var="RETRY_KEYS",
        retry=RetryPolicy(max_retries=2, jitter=0.0, **cfg),
    )
    sleeps: list[float] = []
    chain = ProviderChain([config.primary_provider])
    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = FailoverProvider(config, chain, client=client, sleep=sleeps.append)
    return provider, sleeps


def test_backoff_grows_and_caps():
    policy = RetryPolicy(base_delay_s=1, backoff_factor=2, max_delay_s=5, jitter=0.0)
    assert [policy.delay(n) for n in (1, 2, 3, 4)] == [1, 2, 4, 5]


def test_backoff_jitter_bounds_and_retry_after():
    policy = RetryPolicy(base_delay_s=10, jitter=0.5, max_delay_s=100)
    assert policy.delay(1, rng=lambda: 0.0) == 5
    assert policy.delay(1, rng=lambda: 1.0) == 15
    assert policy.delay(1, retry_after=7) == 7
    assert policy.delay(1, retry_after=1000) == 100


def test_policy_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        RetryPolicy(nope=1)


def test_transient_errors_backoff_then_succeed_and_count_retries(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="down") if calls["n"] <= 2 else httpx.Response(200, json=OK)

    provider, sleeps = make(monkeypatch, handler, base_delay_s=1)
    resp = provider.chat(MSGS)
    assert (resp.retries, resp.attempts) == (2, 3)
    assert sleeps == [1.0, 2.0]
    assert provider.total_requests == 3


def test_rate_limit_rotates_without_sleeping(monkeypatch):
    def handler(req):
        if req.headers["authorization"].endswith("k1"):
            return httpx.Response(429, headers={"Retry-After": "30"}, text="slow")
        return httpx.Response(200, json=OK)

    provider, sleeps = make(monkeypatch, handler)
    resp = provider.chat(MSGS)
    assert resp.retries == 1 and sleeps == []


@pytest.mark.parametrize("status", [400, 404, 422])
def test_hard_errors_are_not_retried(monkeypatch, status):
    provider, sleeps = make(monkeypatch, lambda req: httpx.Response(status, text="bad"))
    with pytest.raises(LLMAPIError) as exc:
        provider.chat(MSGS)
    assert exc.value.attempts == 1 and exc.value.kind is FailureKind.FATAL
    assert sleeps == [] and provider.total_requests == 1


def test_auth_error_drops_key_and_uses_next(monkeypatch):
    seen = []

    def handler(req):
        key = req.headers["authorization"].split()[1]
        seen.append(key)
        return httpx.Response(401, text="no") if key == "k1" else httpx.Response(200, json=OK)

    provider, _ = make(monkeypatch, handler)
    assert provider.chat(MSGS).retries == 1
    provider.chat(MSGS)
    assert seen == ["k1", "k2", "k2"]  # k1 never retried


def test_persistent_transient_failure_raises_llm_error_with_attempts(monkeypatch):
    provider, _ = make(monkeypatch, lambda req: httpx.Response(500, text="boom"), keys="k1")
    provider._max_attempts = 4
    with pytest.raises(LLMError) as exc:
        provider.chat(MSGS)
    assert isinstance(exc.value, LLMRetriesExhaustedError)
    assert exc.value.attempts == 3 == provider.total_requests  # 3 tries, then cooling down
    assert exc.value.kind is FailureKind.TRANSIENT


def test_everything_exhausted_raises_llm_error_subclass(monkeypatch):
    provider, _ = make(monkeypatch, lambda req: httpx.Response(402, text="credits"), keys="k1")
    with pytest.raises(AllProvidersExhaustedError) as exc:
        provider.chat(MSGS)
    assert isinstance(exc.value, LLMError) and exc.value.attempts == 1


def test_unexpected_exception_is_wrapped(monkeypatch):
    def handler(req):
        raise RuntimeError("weird")

    provider, _ = make(monkeypatch, handler)
    with pytest.raises(LLMError) as exc:
        provider.chat(MSGS)
    assert exc.value.kind is FailureKind.FATAL and "weird" in str(exc.value)


def test_total_requests_accumulates_and_owned_client_closes(monkeypatch):
    monkeypatch.setenv("RETRY_KEYS", "k")
    config = ModelConfig(model_name="m", api_key_env_var="RETRY_KEYS", timeout_s=5, connect_timeout_s=1)
    provider = FailoverProvider(config)
    assert provider._client.timeout.read == 5 and provider._client.timeout.connect == 1
    provider.close()
    assert provider._client.is_closed
