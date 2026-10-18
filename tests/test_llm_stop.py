"""Stop sequences: per-format defaults, overrides, provider limits, client-side cut."""

from __future__ import annotations

import json
import logging

import httpx
import pytest
from pydantic import ValidationError

from agent_smith.core.config import ModelConfig
from agent_smith.llm import (
    DEFAULT_STOP_SEQUENCES,
    ChatMessage,
    FailoverProvider,
    OpenAICompatProvider,
    PromptFormat,
    resolve_stop_sequences,
    truncate_at_stop,
)
from agent_smith.llm.stop import MAX_PROVIDER_STOP_SEQUENCES, normalize_for_provider

MSGS = [ChatMessage(role="user", content="hi")]


def provider(content="done", **cfg):
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    config = ModelConfig(model_name="m", **cfg)
    return OpenAICompatProvider(config, api_key="k", client=client), seen, config


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("python", ["<end_code>"]),
        ("xml", ["</tool_call>"]),
        ("hermes", ["</tool_call>"]),
        (PromptFormat.REACT, ["Observation:"]),
    ],
)
def test_defaults_per_format(fmt, expected):
    assert resolve_stop_sequences(fmt) == expected


def test_every_config_format_has_defaults():
    from typing import get_args

    from agent_smith.core.config import ModelConfig as MC

    literal = MC.model_fields["prompt_format"].annotation
    assert {f.value for f in DEFAULT_STOP_SEQUENCES} == set(get_args(literal))


def test_override_replaces_defaults_and_extra_is_appended_deduped():
    assert resolve_stop_sequences("python", override=["A", "", "A"]) == ["A"]
    assert resolve_stop_sequences("python", extra=["<end_code>", "X"]) == ["<end_code>", "X"]
    assert resolve_stop_sequences("python", override=[]) == []


def test_provider_cap_keeps_order_and_warns(caplog):
    seqs = [f"s{i}" for i in range(MAX_PROVIDER_STOP_SEQUENCES + 2)]
    with caplog.at_level(logging.WARNING):
        assert normalize_for_provider(seqs) == seqs[:MAX_PROVIDER_STOP_SEQUENCES]
    assert "Dropping stop sequences" in caplog.text


def test_truncate_at_earliest_match():
    assert truncate_at_stop("a<end_code>b</tool_call>c", ["</tool_call>", "<end_code>"]) == (
        "a",
        "<end_code>",
    )
    assert truncate_at_stop("plain", ["<end_code>"]) == ("plain", None)


def test_payload_uses_format_defaults_then_config_then_call_argument():
    p, seen, _ = provider(prompt_format="xml")
    p.chat(MSGS)
    assert seen["body"]["stop"] == ["</tool_call>"]

    p, seen, _ = provider(prompt_format="xml", stop_sequences=["STOP"])
    p.chat(MSGS)
    assert seen["body"]["stop"] == ["STOP"]
    p.chat(MSGS, stop=["CALL"])
    assert seen["body"]["stop"] == ["CALL"]


def test_empty_override_sends_no_stop():
    p, seen, _ = provider(stop_sequences=[])
    p.chat(MSGS)
    assert "stop" not in seen["body"]


def test_backend_that_ignores_stop_is_cut_client_side():
    p, _, _ = provider(content="print(1)\n<end_code>\nObservation: 42\nFinal answer")
    resp = p.chat(MSGS)
    assert resp.content == "print(1)\n"
    assert resp.stop_sequence == "<end_code>"


def test_no_stop_hit_leaves_content_and_none():
    resp = provider(content="print(1)")[0].chat(MSGS)
    assert (resp.content, resp.stop_sequence) == ("print(1)", None)


def test_invalid_prompt_format_rejected():
    with pytest.raises(ValidationError):
        ModelConfig(model_name="m", prompt_format="markdown")


def test_failover_forwards_per_call_stop_over_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    seen = {}

    def handler(req):
        seen["stop"] = json.loads(req.content).get("stop")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fo = FailoverProvider(ModelConfig(model_name="m", stop_sequences=["CFG"]), client=client)
    fo.chat(MSGS)
    assert seen["stop"] == ["CFG"]
    fo.chat(MSGS, stop=["CALL"])
    assert seen["stop"] == ["CALL"]
