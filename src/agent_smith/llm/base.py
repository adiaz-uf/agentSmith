"""Provider-agnostic types and interface for chat completion."""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChatMessage",
    "ChatResponse",
    "FailureKind",
    "LLMAPIError",
    "LLMConnectionError",
    "LLMError",
    "LLMProvider",
    "LLMRetriesExhaustedError",
]


class FailureKind(enum.Enum):
    RATE_LIMIT = "rate_limit"  # rotate, short cooldown
    QUOTA = "quota"  # rotate, long cooldown
    AUTH = "auth"  # key is invalid: drop it permanently
    TRANSIENT = "transient"  # retry with backoff, then rotate
    FATAL = "fatal"  # bad request: retrying will not help


class LLMError(Exception):
    """Base class for all provider errors.

    ``chat`` never lets anything else escape, so ``except LLMError`` is always enough to degrade
    gracefully (subject IV.1). ``attempts`` is the number of HTTP requests made before giving up
    (retries included) and ``kind`` says why, when known.
    """

    attempts: int = 0
    kind: FailureKind | None = None


class LLMAPIError(LLMError):
    """The provider answered with an error status or an unusable payload."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        body: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        # Kept so failover can read Retry-After and quota hints from the raw response.
        self.headers: Mapping[str, str] = headers or {}
        self.body = body


class LLMConnectionError(LLMError):
    """The provider could not be reached (network failure, timeout)."""


class LLMRetriesExhaustedError(LLMError):
    """The retry/rotation budget was spent without getting a successful response."""


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
    retries: int = Field(
        default=0,
        ge=0,
        description="Failed attempts before this response; feeds `StepMetrics.retries`.",
    )

    @property
    def attempts(self) -> int:
        """HTTP requests made for this response (retries included)."""
        return self.retries + 1


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
            LLMError: (or a subclass) on any failure, carrying ``attempts`` and ``kind``.
            LLMConnectionError: the provider was unreachable.
            LLMAPIError: the provider returned an error or a malformed response.
        """
