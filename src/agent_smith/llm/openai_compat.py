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


CHARS_PER_TOKEN = 4  # rough rule of thumb; only used when the provider reports no usage
MESSAGE_OVERHEAD_TOKENS = 4  # chat framing per message


def _count(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else 0


def _parse_usage(
    usage: Any, messages: Sequence[ChatMessage], content: str
) -> tuple[int, int, int, bool]:
    """Return (input, output, reasoning, estimated) tokens for one response.

    ``output`` always includes reasoning tokens. Provider numbers are never overridden; the
    ~4 chars/token estimate fills in only what is missing or zero for a non-empty exchange.
    """
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = _count(usage.get("prompt_tokens"))
    output_tokens = _count(usage.get("completion_tokens"))
    details = usage.get("completion_tokens_details")
    reasoning = _count(details.get("reasoning_tokens") if isinstance(details, dict) else None)
    reasoning = reasoning or _count(usage.get("reasoning_tokens"))
    if reasoning > output_tokens:
        # Cannot be a subset of completion_tokens: the provider reports it separately.
        output_tokens += reasoning

    estimated = False
    if input_tokens == 0 and messages:
        chars = sum(len(m.content) for m in messages)
        input_tokens = chars // CHARS_PER_TOKEN + MESSAGE_OVERHEAD_TOKENS * len(messages)
        estimated = True
    if output_tokens == 0 and content:
        output_tokens = max(1, len(content) // CHARS_PER_TOKEN)
        estimated = True
    return input_tokens, output_tokens, reasoning, estimated


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

        input_tokens, output_tokens, reasoning_tokens, estimated = _parse_usage(
            data.get("usage"), messages, content
        )
        return ChatResponse(
            content=content,
            api_url=self.api_url,
            model_name=self.model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            usage_estimated=estimated,
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
