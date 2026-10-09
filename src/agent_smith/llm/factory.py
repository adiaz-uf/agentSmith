"""Single selection point for LLM backends."""

from __future__ import annotations

from collections.abc import Callable

from agent_smith.core.config import ModelConfig
from agent_smith.llm.base import LLMProvider
from agent_smith.llm.failover import FailoverProvider
from agent_smith.llm.openai_compat import OpenAICompatProvider

__all__ = ["PROVIDER_KINDS", "create_provider"]

DEFAULT_KIND = "openai"

# Adding a backend means adding one entry here.
PROVIDER_KINDS: dict[str, Callable[[ModelConfig], LLMProvider]] = {
    "openai": OpenAICompatProvider,
    "failover": FailoverProvider,  # token rotation + provider fallback (V.6)
}


def create_provider(config: ModelConfig, kind: str = DEFAULT_KIND) -> LLMProvider:
    """Build the provider for `config` (model name and base URL are chosen at runtime)."""
    try:
        factory = PROVIDER_KINDS[kind]
    except KeyError:
        raise ValueError(
            f"Unknown provider kind '{kind}'. Available: {', '.join(sorted(PROVIDER_KINDS))}"
        ) from None
    return factory(config)
