"""LLMProvider that rotates API tokens and falls back across providers (subject V.6 / V.7)."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import httpx

from agent_smith.core.config import ModelConfig, ProviderConfig
from agent_smith.llm.base import (
    ChatMessage,
    ChatResponse,
    LLMAPIError,
    LLMConnectionError,
    LLMProvider,
)
from agent_smith.llm.openai_compat import OpenAICompatProvider
from agent_smith.llm.providers import (
    MAX_TRANSIENT_RETRIES,
    AllProvidersExhaustedError,
    FailureKind,
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
        self._client = client
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

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        transient_retries = 0
        last_error: LLMAPIError | LLMConnectionError | None = None
        for _ in range(self._max_attempts):
            try:
                provider, key = self._chain.acquire()
            except AllProvidersExhaustedError as exc:
                wait = exc.retry_after
                if wait is None or wait > self._max_wait:
                    raise
                self._sleep(wait)
                continue

            self._last = (provider.provider_url, provider.model_name or self._config.model_name)
            try:
                response = self._build(provider, key).chat(
                    messages, stop=stop, temperature=temperature, max_tokens=max_tokens
                )
            except LLMAPIError as exc:
                last_error = exc
                kind = classify_failure(exc.status_code or 0, exc.headers, exc.body)
                if kind is None or kind is FailureKind.FATAL:
                    raise
                retry_after = parse_retry_after(exc.headers)
            except LLMConnectionError as exc:
                last_error = exc
                kind, retry_after = FailureKind.TRANSIENT, None
            else:
                return response

            if kind is FailureKind.TRANSIENT:
                transient_retries += 1
                if transient_retries <= MAX_TRANSIENT_RETRIES:
                    continue
                kind = FailureKind.RATE_LIMIT  # persistent failure: rotate away from this key
            transient_retries = 0
            self._chain.report_failure(provider, key, kind, retry_after)

        raise last_error or AllProvidersExhaustedError(None)

    def close(self) -> None:
        if self._client is not None:
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


