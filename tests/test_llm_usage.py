"""Usage tracking: reasoning tokens, estimation fallback, aggregation (no network)."""

from __future__ import annotations

import httpx
import pytest

from agent_smith.core.config import ModelConfig
from agent_smith.core.models import SolutionOutput, StepMetrics
from agent_smith.llm import (
    ChatMessage,
    ChatResponse,
    LLMAPIError,
    OpenAICompatProvider,
    TrackedProvider,
    UsageTracker,
)

MSGS = [ChatMessage(role="user", content="a" * 40)]


def chat_with(usage, content="x" * 80, messages=MSGS) -> ChatResponse:
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)))
    provider = OpenAICompatProvider(ModelConfig(model_name="m"), api_key="k", client=client)
    return provider.chat(messages)


def test_real_usage_is_never_overridden():
    r = chat_with({"prompt_tokens": 7, "completion_tokens": 3})
    assert (r.input_tokens, r.output_tokens, r.reasoning_tokens, r.usage_estimated) == (7, 3, 0, False)


def test_reasoning_tokens_nested_is_subset_of_output():
    r = chat_with(
        {
            "prompt_tokens": 10,
            "completion_tokens": 50,
            "completion_tokens_details": {"reasoning_tokens": 30},
        }
    )
    assert (r.output_tokens, r.reasoning_tokens) == (50, 30)


def test_flat_reasoning_tokens_larger_than_completion_are_added():
    r = chat_with({"prompt_tokens": 10, "completion_tokens": 5, "reasoning_tokens": 20})
    assert r.reasoning_tokens == 20 and r.output_tokens == 25 >= r.reasoning_tokens


@pytest.mark.parametrize("usage", [None, {}, {"prompt_tokens": 0, "completion_tokens": 0}])
def test_missing_usage_is_estimated_and_flagged(usage):
    r = chat_with(usage)
    assert r.usage_estimated
    assert r.input_tokens == 40 // 4 + 4
    assert r.output_tokens == 80 // 4


def test_only_missing_side_is_estimated():
    r = chat_with({"prompt_tokens": 9, "completion_tokens": 0})
    assert (r.input_tokens, r.output_tokens, r.usage_estimated) == (9, 20, True)


def test_empty_reply_is_not_estimated():
    r = chat_with({"prompt_tokens": 9}, content="")
    assert (r.output_tokens, r.usage_estimated) == (0, False)


def resp(**kw) -> ChatResponse:
    base = {
        "content": "c",
        "api_url": "https://a",
        "model_name": "m",
        "input_tokens": 10,
        "output_tokens": 5,
    }
    return ChatResponse(**{**base, **kw})


def test_tracker_totals_include_retries_failures_and_reasoning():
    t = UsageTracker()
    t.record(resp(request_time_ms=100, retries=0))
    t.record(resp(request_time_ms=250, retries=2, reasoning_tokens=3, usage_estimated=True))
    t.record(resp(model_name="m2", input_tokens=1, output_tokens=1))
    failure = LLMAPIError("boom", 500)
    failure.attempts = 4
    t.record_failure(failure)

    totals = t.totals()
    assert totals.requests == 1 + 3 + 1 + 4
    assert totals.retries == 2 + 3
    assert (totals.input_tokens, totals.output_tokens) == (21, 11)
    assert totals.reasoning_tokens == 3 and totals.estimated
    assert totals.request_time_ms == 350
    assert totals.per_model["https://a|m"].requests == 4
    assert totals.per_model["https://a|m2"].input_tokens == 1


def test_step_metrics_and_solution_totals_validate_against_models():
    now = {"t": 100.0}
    t = UsageTracker(clock=lambda: now["t"])
    r = resp(request_time_ms=12.5, retries=1)
    t.record(r)
    now["t"] = 107.5

    step = t.step_metrics(1, r, llm_output="out")
    assert isinstance(step, StepMetrics)
    assert (step.retries, step.request_time_ms, step.api_url, step.llm_output) == (1, 12.5, "https://a", "out")

    out = SolutionOutput(
        task_id=1, benchmark="mbpp", success=True, iterations=1, steps=[step], **t.solution_totals()
    )
    assert (out.total_requests, out.total_input_tokens, out.total_output_tokens) == (2, 10, 5)
    assert out.total_time_seconds == 7.5


class Stub:
    api_url, model_name = "https://a", "m"

    def __init__(self, result):
        self.result = result

    def chat(self, messages, **kw):
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_tracked_provider_records_success_and_reraises_failures():
    tracked = TrackedProvider(Stub(resp(retries=1)))
    assert tracked.chat(MSGS).retries == 1
    err = LLMAPIError("x", 400)
    err.attempts = 1
    tracked._inner = Stub(err)
    with pytest.raises(LLMAPIError) as exc:
        tracked.chat(MSGS)
    assert exc.value is err
    assert tracked.tracker.totals().requests == 2 + 1
