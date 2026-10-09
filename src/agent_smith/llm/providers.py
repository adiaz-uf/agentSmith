"""API token rotation and provider fallback (subject V.6 / V.7).

Transport-agnostic: a ``ProviderChain`` hands out ``(provider, api_key)`` pairs,
skipping keys that are cooling down after a rate-limit / quota error and falling
back to the next provider when every key of one provider is exhausted.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from agent_smith.core.config import ModelConfig, ProviderConfig
from agent_smith.llm.base import FailureKind, LLMError

__all__ = [
    "AllProvidersExhaustedError",
    "FailureKind",
    "ProviderChain",
    "TokenPool",
    "classify_failure",
    "load_provider_chain",
    "mask_key",
    "parse_retry_after",
]

logger = logging.getLogger(__name__)

DEFAULT_COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 3600.0
QUOTA_COOLDOWN_S = 3600.0
_QUOTA_MARKERS = ("quota", "insufficient", "daily", "credits", "billing")


def mask_key(key: str) -> str:
    """Return a log-safe form of an API key (last 4 characters only)."""
    return f"...{key[-4:]}" if len(key) > 4 else "***"


class AllProvidersExhaustedError(LLMError):
    """Every token of every provider is cooling down or disabled."""

    def __init__(self, retry_after: float | None):
        self.retry_after = retry_after
        when = f"; next token available in {retry_after:.0f}s" if retry_after is not None else ""
        super().__init__(f"All providers/tokens are exhausted{when}")


class _Token:
    __slots__ = ("available_at", "disabled", "failures", "key", "requests")

    def __init__(self, key: str):
        self.key = key
        self.available_at = 0.0
        self.disabled = False
        self.requests = 0
        self.failures = 0


class TokenPool:
    """Round-robin pool of API keys with per-key cooldowns."""

    def __init__(self, keys: list[str], clock: Callable[[], float] = time.monotonic):
        self._tokens = [_Token(k) for k in keys]
        self._clock = clock
        self._next = 0

    def __len__(self) -> int:
        return len(self._tokens)

    def acquire(self) -> str | None:
        """Return the next healthy key (round-robin), or None if none is available."""
        now = self._clock()
        n = len(self._tokens)
        for offset in range(n):
            idx = (self._next + offset) % n
            tok = self._tokens[idx]
            if not tok.disabled and tok.available_at <= now:
                self._next = (idx + 1) % n
                tok.requests += 1
                return tok.key
        return None

    def mark_exhausted(self, key: str, cooldown: float = DEFAULT_COOLDOWN_S) -> None:
        tok = self._find(key)
        tok.failures += 1
        tok.available_at = self._clock() + min(cooldown, MAX_COOLDOWN_S)
        logger.warning("Token %s cooling down for %.0fs", mask_key(key), cooldown)

    def disable(self, key: str) -> None:
        tok = self._find(key)
        tok.failures += 1
        tok.disabled = True
        logger.error("Token %s disabled (authentication failure)", mask_key(key))

    def seconds_until_available(self) -> float | None:
        """Seconds until the soonest non-disabled key is usable (None if all disabled)."""
        waits = [t.available_at for t in self._tokens if not t.disabled]
        if not waits:
            return None
        return max(0.0, min(waits) - self._clock())

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            mask_key(t.key): {"requests": t.requests, "failures": t.failures}
            for t in self._tokens
        }

    def _find(self, key: str) -> _Token:
        for tok in self._tokens:
            if tok.key == key:
                return tok
        raise KeyError("unknown token")


@dataclass
class _Entry:
    config: ProviderConfig
    pool: TokenPool


@dataclass
class ProviderChain:
    """Ordered providers (priority first), each with its own token pool."""

    providers: list[ProviderConfig]
    clock: Callable[[], float] = time.monotonic
    _entries: list[_Entry] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._entries = [
            _Entry(p, TokenPool(p.api_keys, self.clock)) for p in self.providers
        ]

    def acquire(self) -> tuple[ProviderConfig, str]:
        """Return the first available (provider, key); raise if everything is exhausted."""
        for entry in self._entries:
            key = entry.pool.acquire()
            if key is not None:
                return entry.config, key
        waits = [w for e in self._entries if (w := e.pool.seconds_until_available()) is not None]
        raise AllProvidersExhaustedError(min(waits) if waits else None)

    def report_failure(
        self,
        provider: ProviderConfig,
        key: str,
        kind: FailureKind,
        retry_after: float | None = None,
    ) -> None:
        """Apply the rotation policy for a failed request (no-op for TRANSIENT/FATAL)."""
        pool = self._pool_for(provider)
        if kind is FailureKind.AUTH:
            pool.disable(key)
        elif kind is FailureKind.QUOTA:
            pool.mark_exhausted(key, retry_after or QUOTA_COOLDOWN_S)
        elif kind is FailureKind.RATE_LIMIT:
            pool.mark_exhausted(key, retry_after or DEFAULT_COOLDOWN_S)

    def stats(self) -> dict[str, dict[str, dict[str, int]]]:
        return {e.config.name: e.pool.stats() for e in self._entries}

    def _pool_for(self, provider: ProviderConfig) -> TokenPool:
        for e in self._entries:
            if e.config is provider:
                return e.pool
        raise KeyError(provider.name)


def parse_retry_after(headers: Mapping[str, str]) -> float | None:
    for name, value in headers.items():
        if name.lower() == "retry-after":
            try:
                return max(0.0, float(value))
            except ValueError:
                return None
    return None


def classify_failure(status: int, headers: Mapping[str, str], body: str) -> FailureKind | None:
    """Map an HTTP response to a FailureKind; None means success (2xx/3xx)."""
    if status < 400:
        return None
    lowered = body.lower()
    if status == 402 or (status == 429 and any(m in lowered for m in _QUOTA_MARKERS)):
        return FailureKind.QUOTA
    if status == 429:
        return FailureKind.RATE_LIMIT
    if status in (401, 403):
        return FailureKind.AUTH
    if status in (408, 409, 425) or status >= 500:
        return FailureKind.TRANSIENT
    return FailureKind.FATAL


def load_provider_chain(config: ModelConfig) -> ProviderChain:
    """Build the ordered rotation/fallback chain: primary provider, then fallbacks."""
    return ProviderChain([config.primary_provider, *config.fallbacks])
