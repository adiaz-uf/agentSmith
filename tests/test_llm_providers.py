"""Tests for token rotation and provider fallback (no network)."""

from __future__ import annotations

import time

import httpx
import pytest

from agent_smith.core.config import ModelConfig, ProviderConfig
from agent_smith.llm import (
    AllProvidersExhaustedError,
    ChatMessage,
    FailoverProvider,
    FailureKind,
    LLMAPIError,
    LLMConnectionError,
    ProviderChain,
    TokenPool,
    classify_failure,
    create_provider,
    load_provider_chain,
)
from agent_smith.llm.providers import mask_key


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def chain(clock, monkeypatch, **providers):
    cfgs = []
    for name, keys in providers.items():
        monkeypatch.setenv(f"{name.upper()}_KEYS", keys)
        cfgs.append(ProviderConfig(name=name, provider_url=f"https://{name}", api_key_env_vars=[f"{name.upper()}_KEYS"]))
    return ProviderChain(cfgs, clock=clock)


def test_round_robin_and_cooldown_expiry():
    clock = Clock()
    pool = TokenPool(["a", "b"], clock)
    assert [pool.acquire(), pool.acquire(), pool.acquire()] == ["a", "b", "a"]
    pool.mark_exhausted("a", 10)
    assert [pool.acquire(), pool.acquire()] == ["b", "b"]
    clock.t = 11
    assert "a" in {pool.acquire(), pool.acquire()}


def test_keys_split_and_deduplicated(monkeypatch):
    monkeypatch.setenv("K1", "x, y,,x")
    monkeypatch.setenv("K2", "z")
    cfg = ProviderConfig(name="p", provider_url="u", api_key_env_vars=["K1", "K2", "MISSING"])
    assert cfg.api_keys == ["x", "y", "z"]


def test_fallback_to_next_provider_then_exhausted(monkeypatch):
    clock = Clock()
    c = chain(clock, monkeypatch, a="k1", b="k2")
    pa, ka = c.acquire()
    c.report_failure(pa, ka, FailureKind.RATE_LIMIT, retry_after=60)
    pb, kb = c.acquire()
    assert (pb.name, kb) == ("b", "k2")
    c.report_failure(pb, kb, FailureKind.QUOTA)
    with pytest.raises(AllProvidersExhaustedError) as exc:
        c.acquire()
    assert exc.value.retry_after == 60
    clock.t = 61
    assert c.acquire()[0].name == "a"


def test_auth_failure_disables_key_permanently(monkeypatch):
    clock = Clock()
    c = chain(clock, monkeypatch, a="k1")
    p, k = c.acquire()
    c.report_failure(p, k, FailureKind.AUTH)
    clock.t = 10**6
    with pytest.raises(AllProvidersExhaustedError) as exc:
        c.acquire()
    assert exc.value.retry_after is None


@pytest.mark.parametrize(
    "status,body,kind",
    [
        (200, "", None),
        (429, "slow down", FailureKind.RATE_LIMIT),
        (429, "Daily quota exceeded", FailureKind.QUOTA),
        (402, "", FailureKind.QUOTA),
        (401, "", FailureKind.AUTH),
        (503, "", FailureKind.TRANSIENT),
        (400, "bad", FailureKind.FATAL),
    ],
)
def test_classify_failure(status, body, kind):
    assert classify_failure(status, {}, body) is kind


MSGS = [ChatMessage(role="user", content="hi")]
OK = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}


def failover(monkeypatch, handler, fallbacks=(), sleep=None, clock=None, **kw):
    monkeypatch.setenv("PRIMARY_KEYS", "k1,k2")
    cfg = ModelConfig(
        model_name="m",
        provider_url="https://a",
        api_key_env_var="PRIMARY_KEYS",
        fallbacks=list(fallbacks),
    )
    client = httpx.Client(transport=httpx.MockTransport(handler))
    chain = ProviderChain([cfg.primary_provider, *cfg.fallbacks], clock=clock or time.monotonic)
    return FailoverProvider(cfg, chain, client=client, sleep=sleep or (lambda s: None), **kw)


def test_failover_provider_rotates_on_429(monkeypatch):
    seen = []

    def handler(req):
        key = req.headers["authorization"].split()[1]
        seen.append(key)
        if key == "k1":
            return httpx.Response(429, headers={"Retry-After": "30"}, text="slow")
        return httpx.Response(200, json=OK)

    assert failover(monkeypatch, handler).chat(MSGS).content == "ok"
    assert seen == ["k1", "k2"]


def test_failover_provider_falls_back_and_records_real_provider(monkeypatch):
    monkeypatch.setenv("B_KEYS", "b1")
    fb = ProviderConfig(
        name="b", provider_url="https://b", api_key_env_vars=["B_KEYS"], model_name="m-b"
    )

    def handler(req):
        if req.url.host == "a":
            return httpx.Response(429, text="Daily quota exceeded")
        return httpx.Response(200, json=OK)

    provider = failover(monkeypatch, handler, fallbacks=[fb])
    resp = provider.chat(MSGS)
    assert (resp.api_url, resp.model_name) == ("https://b", "m-b")
    assert (provider.api_url, provider.model_name) == ("https://b", "m-b")


def test_failover_provider_fatal_error_not_rotated(monkeypatch):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(400, text="bad request")

    with pytest.raises(LLMAPIError) as exc:
        failover(monkeypatch, handler).chat(MSGS)
    assert exc.value.status_code == 400 and len(calls) == 1


def test_failover_provider_waits_when_all_cooling_down(monkeypatch):
    waits, n = [], {"i": 0}

    def handler(req):
        n["i"] += 1
        if n["i"] <= 2:
            return httpx.Response(429, headers={"Retry-After": "5"}, text="slow")
        return httpx.Response(200, json=OK)

    t = {"now": 0.0}

    def sleep(s):
        waits.append(s)
        t["now"] += s

    provider = failover(monkeypatch, handler, sleep=sleep, clock=lambda: t["now"])
    assert provider.chat(MSGS).content == "ok"
    assert waits == [5.0]


def test_failover_provider_exhausted_raises_when_wait_too_long(monkeypatch):
    provider = failover(
        monkeypatch,
        lambda req: httpx.Response(402, text="insufficient credits"),
    )
    with pytest.raises(AllProvidersExhaustedError):
        provider.chat(MSGS)


def test_failover_provider_connection_errors_rotate(monkeypatch):
    def handler(req):
        raise httpx.ConnectTimeout("slow", request=req)

    provider = failover(monkeypatch, handler, max_attempts=6)
    with pytest.raises(LLMConnectionError):
        provider.chat(MSGS)


def test_factory_creates_failover_provider(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert isinstance(create_provider(ModelConfig(model_name="m"), "failover"), FailoverProvider)


def test_model_config_backcompat_and_chain(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "p1")
    monkeypatch.setenv("EXTRA", "p2,p3")
    cfg = ModelConfig.model_validate_json(
        '{"model_name": "m", "api_key_env_vars": ["EXTRA"],'
        ' "fallbacks": [{"name": "g", "provider_url": "https://g", "api_key_env_vars": []}]}'
    )
    assert [p.name for p in load_provider_chain(cfg).providers] == ["primary", "g"]
    assert cfg.primary_provider.api_keys == ["p1", "p2", "p3"]
    assert ModelConfig(model_name="m").fallbacks == []


def test_mask_key():
    assert mask_key("sk-secret-1234") == "...1234"
