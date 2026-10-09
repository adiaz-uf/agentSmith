"""LLM provider abstraction: multi-provider, multi-token, usage tracking."""

from agent_smith.llm.base import (
    ChatMessage,
    ChatResponse,
    LLMAPIError,
    LLMConnectionError,
    LLMError,
    LLMProvider,
)
from agent_smith.llm.factory import create_provider
from agent_smith.llm.failover import FailoverProvider
from agent_smith.llm.openai_compat import OpenAICompatProvider
from agent_smith.llm.providers import (
    AllProvidersExhaustedError,
    FailureKind,
    ProviderChain,
    TokenPool,
    classify_failure,
    load_provider_chain,
)

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
    "OpenAICompatProvider",
    "ProviderChain",
    "TokenPool",
    "classify_failure",
    "create_provider",
    "load_provider_chain",
]
