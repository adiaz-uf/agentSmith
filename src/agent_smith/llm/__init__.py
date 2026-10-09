"""LLM provider abstraction: multi-provider, multi-token, usage tracking."""

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
from agent_smith.llm.factory import create_provider
from agent_smith.llm.failover import FailoverProvider
from agent_smith.llm.openai_compat import OpenAICompatProvider
from agent_smith.llm.providers import (
    AllProvidersExhaustedError,
    ProviderChain,
    TokenPool,
    classify_failure,
    load_provider_chain,
)
from agent_smith.llm.usage import TrackedProvider, UsageTotals, UsageTracker

__all__ = [
    "AllProvidersExhaustedError",
    "ChatMessage",
    "ChatResponse",
    "FailoverProvider",
    "FailureKind",
    "LLMAPIError",
    "LLMConnectionError",
    "LLMError",
    "LLMProvider",
    "LLMRetriesExhaustedError",
    "OpenAICompatProvider",
    "ProviderChain",
    "TokenPool",
    "TrackedProvider",
    "UsageTotals",
    "UsageTracker",
    "classify_failure",
    "create_provider",
    "load_provider_chain",
]
