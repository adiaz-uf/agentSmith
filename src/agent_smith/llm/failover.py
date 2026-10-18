"""LLMProvider that rotates API tokens and falls back across providers (subject V.6 / V.7)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import httpx

from agent_smith.core.config import ModelConfig, ProviderConfig
from agent_smith.llm.base import (
    ChatMessage,
    ChatResponse,
    FailureKind,
    LLMAPIError,
    LLMConnectionError,
    LLMError,
    LLMProvider,
    LLMRetriesExhaustedError,
)
from agent_smith.llm.openai_compat import OpenAICompatProvider, make_client
from agent_smith.llm.providers import (
    AllProvidersExhaustedError,
    ProviderChain,
    classify_failure,
    load_provider_chain,
    parse_retry_after,
)

__all__ = ["FailoverProvider"]

DEFAULT_MAX_ATTEMPTS = 20
DEFAULT_MAX_WAIT_S = 120.0


class FailoverProvider(LLMProvider):
    """Wraps a ``ProviderChain``: each ``chat`` uses the next healthy key, and moves on to the
    next key / provider when one is rate-limited, out of quota, invalid or unreachable.

    ``api_url`` and ``model_name`` reflect the provider that served the last request, so the
    values recorded in ``StepMetrics`` (via ``ChatResponse``) are always the real ones.
    """

    def __init__(
        self,
        config: ModelConfig,
        chain: ProviderChain | None = None,
        *,
        client: httpx.Client | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_wait: float = DEFAULT_MAX_WAIT_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._chain = chain or load_provider_chain(config)
        self._owns_client = client is None
        self._client = client or make_client(config)  # one shared client, configured timeouts
        self._total_requests = 0
        self._max_attempts = max_attempts
        self._max_wait = max_wait
        self._sleep = sleep
        self._last: tuple[str, str] = (config.provider_url, config.model_name)

    @property
    def api_url(self) -> str:
        return self._last[0]

    @property
    def model_name(self) -> str:
        return self._last[1]

    @property
    def chain(self) -> ProviderChain:
        return self._chain

    @property
    def total_requests(self) -> int:
        """HTTP requests made so far, retries included (feeds `SolutionOutput.total_requests`)."""
        return self._total_requests

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        policy = self._config.retry
        attempts = 0  # HTTP requests made by this call; retries = attempts - 1 on success
        transient_streak = 0
        last_error: LLMError | None = None
        for _ in range(self._max_attempts):
            try:
                provider, key = self._chain.acquire()
            except AllProvidersExhaustedError as exc:
                wait = exc.retry_after
                if wait is None or wait > self._max_wait:
                    raise _annotate(exc, attempts, FailureKind.RATE_LIMIT) from None
                self._sleep(wait)
                continue

            self._last = (provider.provider_url, provider.model_name or self._config.model_name)
            attempts += 1
            self._total_requests += 1
            retry_after = None
            try:
                response = self._build(provider, key).chat(
                    messages, stop=stop, temperature=temperature, max_tokens=max_tokens
                )
            except LLMAPIError as exc:
                last_error = exc
                kind = classify_failure(exc.status_code or 0, exc.headers, exc.body)
                if kind is None:  # 2xx with an unusable payload: retrying will not fix it
                    kind = FailureKind.FATAL
                if kind is FailureKind.FATAL:
                    raise _annotate(exc, attempts, kind) from None
                retry_after = parse_retry_after(exc.headers)
            except LLMConnectionError as exc:
                last_error, kind = exc, FailureKind.TRANSIENT
            except LLMError as exc:
                raise _annotate(exc, attempts, FailureKind.FATAL) from None
            except Exception as exc:  # never let a provider error crash the agent (IV.1)
                raise _annotate(
                    LLMError(f"Unexpected error from {provider.provider_url}: {exc}"),
                    attempts,
                    FailureKind.FATAL,
                ) from exc
            else:
                return response.model_copy(update={"retries": attempts - 1})

            if kind is FailureKind.TRANSIENT:
                transient_streak += 1
                if transient_streak <= policy.max_retries:
                    self._sleep(policy.delay(transient_streak, retry_after))
                    continue
                kind = FailureKind.RATE_LIMIT  # persistent failure: rotate away from this key
            transient_streak = 0
            self._chain.report_failure(provider, key, kind, retry_after)

        raise _annotate(
            LLMRetriesExhaustedError(f"Gave up after {attempts} attempts: {last_error}"),
            attempts,
            FailureKind.TRANSIENT,
        ) from last_error

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _build(self, provider: ProviderConfig, key: str) -> OpenAICompatProvider:
        config = self._config.model_copy(
            update={
                "provider_url": provider.provider_url,
                "model_name": provider.model_name or self._config.model_name,
            }
        )
        return OpenAICompatProvider(config, api_key=key, client=self._client)

    def __repr__(self) -> str:
        return f"FailoverProvider(providers={[p.name for p in self._chain.providers]})"


def _annotate(exc: LLMError, attempts: int, kind: FailureKind) -> LLMError:
    exc.attempts = attempts
    exc.kind = kind
    return exc
