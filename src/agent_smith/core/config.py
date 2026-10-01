"""Configuration management for Agent Smith.

Supports:
- Loading environment variables from .env and explicit --env-file CLI flags.
- Resolving API keys (e.g. OPENROUTER_API_KEY) without hardcoding secrets.
- JSON configuration loading and validation for sandbox (SandboxConfig) and models (ModelConfig).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from agent_smith.core.models import (
    DEFAULT_ALLOWED_DIRECTORIES,
    DEFAULT_AUTHORIZED_IMPORTS,
    SandboxConfig,
)

__all__ = [
    "DEFAULT_ALLOWED_DIRECTORIES",
    "DEFAULT_AUTHORIZED_IMPORTS",
    "ModelConfig",
    "SandboxConfig",
    "get_api_key",
    "load_env",
    "load_model_config",
    "load_sandbox_config",
]


class ModelConfig(BaseModel):
    """Configuration for an LLM model and API provider endpoint."""

    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(
        ...,
        description="Model identifier (e.g., 'qwen/qwen3-235b-a22b-2507', 'anthropic/claude-3.5-sonnet').",
    )
    provider_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL of the LLM API provider endpoint.",
    )
    api_key_env_var: str = Field(
        default="OPENROUTER_API_KEY",
        description="Name of the environment variable containing the API key.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for code generation.",
    )
    max_tokens: int = Field(
        default=4096,
        gt=0,
        description="Maximum tokens generated per request.",
    )

    @property
    def api_key(self) -> str | None:
        """Resolve the API key from environment."""
        return os.environ.get(self.api_key_env_var)

    @classmethod
    def from_file(cls, path: Path | str) -> ModelConfig:
        """Load and validate ModelConfig from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Model configuration file not found: {path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())


def load_env(env_file: Path | str | None = None, override: bool = False) -> bool:
    """Load environment variables from a .env file.

    Args:
        env_file: Specific path to a .env file (e.g. from --env-file).
                  If None, attempts to find and load a .env file automatically.
        override: Whether to override existing environment variables (default: False).

    Returns:
        True if an environment file was found and loaded, False otherwise.

    Raises:
        FileNotFoundError: If an explicit `env_file` path was given but does not exist.
    """
    if env_file is not None:
        path = Path(env_file)
        if not path.is_file():
            raise FileNotFoundError(f"Environment file not found: {env_file}")
        return load_dotenv(dotenv_path=path, override=override)

    # Restrict auto-discovery to the project root instead of walking
    # up the directory tree, so we never accidentally load a .env
    # from an unrelated parent directory.
    # NOTE: This assumes an editable (src-layout) install where __file__
    # is ``<root>/src/agent_smith/core/config.py``.  In a non-editable
    # install the path would differ and auto-discovery would silently
    # return False (safe default).
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    candidate = project_root / ".env"
    if candidate.is_file():
        return load_dotenv(dotenv_path=candidate, override=override)
    return False


def load_sandbox_config(path: Path | str | None = None) -> SandboxConfig:
    """Load sandbox configuration from JSON, or return default if path is None."""
    if path is None:
        return SandboxConfig()
    return SandboxConfig.from_file(path)


def load_model_config(path: Path | str) -> ModelConfig:
    """Load model configuration from a JSON file."""
    return ModelConfig.from_file(path)


def get_api_key(
    env_var: str = "OPENROUTER_API_KEY",
    required: bool = False,
) -> str | None:
    """Retrieve an API key from the environment.

    Args:
        env_var: Name of the environment variable (default: 'OPENROUTER_API_KEY').
        required: If True, raises ValueError when the variable is unset or empty.

    Returns:
        The API key string, or None if not set and not required.
    """
    key = os.environ.get(env_var)
    if required and (key is None or not key.strip()):
        raise ValueError(
            f"API key missing: environment variable '{env_var}' is not set. "
            f"Ensure it is defined in your environment or provided via --env-file."
        )
    return key
