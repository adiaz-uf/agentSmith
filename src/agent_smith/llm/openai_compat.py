"""Chat-completions backend for any server speaking the OpenAI-compatible protocol.

"OpenAI-compatible" names a wire protocol (served by OpenRouter, Groq, Mistral, Ollama, ...),
not a vendor dependency: no vendor SDK is used and callers only ever see `LLMProvider`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from types import TracebackType
from typing import Any

import httpx

from agent_smith.core.config import ModelConfig
from agent_smith.llm.base import (
    ChatMessage,
    ChatResponse,
    LLMAPIError,
    LLMConnectionError,
    LLMProvider,
)

__all__ = ["OpenAICompatProvider", "make_client"]


def make_client(config: ModelConfig) -> httpx.Client:
    return httpx.Client(timeout=_timeout(config))


def _timeout(config: ModelConfig, total: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(total or config.timeout_s, connect=config.connect_timeout_s)


class OpenAICompatProvider(LLMProvider):
    def __init__(
        self,
        config: ModelConfig,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        timeout: float | None = None,
    ) -> None:
        self._config = config
        self._api_key = api_key if api_key is not None else config.api_key
        self._owns_client = client is None
        self._client = client or httpx.Client(timeout=_timeout(config, timeout))

    @property
    def api_url(self) -> str:
        return self._config.provider_url

    @property
    def model_name(self) -> str:
        return self._config.model_name

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [m.model_dump() for m in messages],
            "temperature": self._config.temperature if temperature is None else temperature,
            "max_tokens": self._config.max_tokens if max_tokens is None else max_tokens,
        }
        if stop:
            payload["stop"] = list(stop)
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        url = f"{self.api_url.rstrip('/')}/chat/completions"

        start = time.perf_counter()
        try:
            response = self._client.post(url, json=payload, headers=headers)
        except httpx.TransportError as exc:
            raise LLMConnectionError(f"Cannot reach {self.api_url}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if response.status_code >= 400:
            raise LLMAPIError(
                f"{self.api_url} returned HTTP {response.status_code}: {response.text[:500]}",
                status_code=response.status_code,
                headers=response.headers,
                body=response.text,
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMAPIError(
                f"Malformed response from {self.api_url}: {response.text[:500]}",
                status_code=response.status_code,
            ) from exc

        usage = data.get("usage") or {}
        return ChatResponse(
            content=content,
            api_url=self.api_url,
            model_name=self.model_name,
            input_tokens=usage.get("prompt_tokens") or 0,
            output_tokens=usage.get("completion_tokens") or 0,
            request_time_ms=elapsed_ms,
            finish_reason=choice.get("finish_reason"),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OpenAICompatProvider:  # noqa: PYI034 (py3.10 has no typing.Self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
