"""Run-level usage tracking: tokens, requests, latency, retries (subject VI.1).

``UsageTracker`` aggregates what each ``ChatResponse`` reports into the numbers that go into
``StepMetrics`` (per step) and ``SolutionOutput`` (``total_*``). Reasoning tokens are part of
``output_tokens`` and are also broken out separately. ``request_time_ms`` is the latency of the
successful HTTP request; backoff sleeps between retries are not included.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agent_smith.core.models import StepMetrics
from agent_smith.llm.base import ChatMessage, ChatResponse, LLMError, LLMProvider

__all__ = ["ModelUsage", "TrackedProvider", "UsageTotals", "UsageTracker"]


@dataclass(frozen=True)
class ModelUsage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class UsageTotals:
    requests: int  # HTTP requests, retries and failed calls included
    retries: int
    input_tokens: int
    output_tokens: int  # includes reasoning_tokens
    reasoning_tokens: int
    request_time_ms: float
    estimated: bool  # some token counts were estimated (provider reported no usage)
    per_model: dict[str, ModelUsage] = field(default_factory=dict)  # keyed "api_url|model"


class UsageTracker:
    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._start = clock()
        self._requests = 0
        self._retries = 0
        self._input = 0
        self._output = 0
        self._reasoning = 0
        self._time_ms = 0.0
        self._estimated = False
        self._per_model: dict[str, ModelUsage] = {}

    def record(self, response: ChatResponse) -> None:
        """Account for one successful call (all of its attempts)."""
        self._requests += response.attempts
        self._retries += response.retries
        self._input += response.input_tokens
        self._output += response.output_tokens
        self._reasoning += response.reasoning_tokens
        self._time_ms += response.request_time_ms
        self._estimated = self._estimated or response.usage_estimated
        key = f"{response.api_url}|{response.model_name}"
        prev = self._per_model.get(key, ModelUsage())
        self._per_model[key] = ModelUsage(
            prev.requests + response.attempts,
            prev.input_tokens + response.input_tokens,
            prev.output_tokens + response.output_tokens,
            prev.reasoning_tokens + response.reasoning_tokens,
        )

    def record_failure(self, error: LLMError) -> None:
        """Account for a call that ended in an error: its attempts were still requests."""
        self._requests += error.attempts
        self._retries += max(error.attempts - 1, 0)

    def totals(self) -> UsageTotals:
        return UsageTotals(
            requests=self._requests,
            retries=self._retries,
            input_tokens=self._input,
            output_tokens=self._output,
            reasoning_tokens=self._reasoning,
            request_time_ms=self._time_ms,
            estimated=self._estimated,
            per_model=dict(self._per_model),
        )

    def elapsed_seconds(self) -> float:
        return max(0.0, self._clock() - self._start)

    def step_metrics(self, step: int, response: ChatResponse, **extra: Any) -> StepMetrics:
        """Build the ``StepMetrics`` for one LLM call; ``extra`` (llm_output, sandbox_*...) passes through."""
        return StepMetrics(
            step=step,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            request_time_ms=response.request_time_ms,
            api_url=response.api_url,
            model_name=response.model_name,
            retries=response.retries,
            **extra,
        )

    def solution_totals(self) -> dict[str, int | float]:
        """Keyword arguments for ``SolutionOutput``'s ``total_*`` fields."""
        totals = self.totals()
        return {
            "total_requests": totals.requests,
            "total_input_tokens": totals.input_tokens,
            "total_output_tokens": totals.output_tokens,
            "total_time_seconds": self.elapsed_seconds(),
        }


class TrackedProvider(LLMProvider):
    """Decorator recording every ``chat`` outcome into a ``UsageTracker``; errors re-raise as-is."""

    def __init__(self, inner: LLMProvider, tracker: UsageTracker | None = None) -> None:
        self._inner = inner
        self.tracker = tracker or UsageTracker()

    @property
    def api_url(self) -> str:
        return self._inner.api_url

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        try:
            response = self._inner.chat(
                messages, stop=stop, temperature=temperature, max_tokens=max_tokens
            )
        except LLMError as exc:
            self.tracker.record_failure(exc)
            raise
        self.tracker.record(response)
        return response
