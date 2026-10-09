"""Provider-agnostic types and interface for chat completion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "LLMAPIError",
    "LLMConnectionError",
    "LLMError",
    "LLMProvider",
]


class LLMError(Exception):
    """Base class for all provider errors."""


class LLMAPIError(LLMError):
    """The provider answered with an error status or an unusable payload."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMConnectionError(LLMError):
    """The provider could not be reached (network failure, timeout)."""


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class ChatResponse(BaseModel):
    """Result of one chat completion, carrying what `StepMetrics` needs."""

    model_config = ConfigDict(extra="forbid")

    content: str
    api_url: str = Field(..., description="Base URL of the provider that served the request.")
    model_name: str = Field(..., description="Model identifier used for the request.")
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    request_time_ms: float = Field(default=0.0, ge=0.0)
    finish_reason: str | None = None


class LLMProvider(ABC):
    """One interface for chat completion, whatever the backend."""

    @property
    @abstractmethod
    def api_url(self) -> str: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        stop: Sequence[str] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResponse:
        """Send `messages` and return the model's reply.

        Raises:
            LLMConnectionError: the provider was unreachable.
            LLMAPIError: the provider returned an error or a malformed response.
        """
