"""Configuration management for Agent Smith.

Supports:
- Loading environment variables from .env and explicit --env-file CLI flags.
- Resolving API keys (e.g. OPENROUTER_API_KEY) without hardcoding secrets.
- JSON configuration loading and validation for sandbox (SandboxConfig) and models (ModelConfig).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# NOTE / TODO:
# SandboxConfig and its default imports/directories are defined here temporarily
# to make configuration loading (Issue #4) fully testable and autonomous in this
# branch without prematurely moving or copying files from outside.
#
# Once Issue #3 (repository layout & models) is addressed and models_public.py is
# officially placed in `agent_smith.core.models`, this class definition and its
# defaults will be removed from here and imported directly from `agent_smith.core.models`
# (e.g., `from agent_smith.core.models import SandboxConfig`) to maintain a single source
# of truth.
# -----------------------------------------------------------------------------

DEFAULT_AUTHORIZED_IMPORTS = [
    "math",
    "math.*",
    "collections",
    "collections.*",
    "itertools",
    "re",
    "json",
    "typing",
    "typing.*",
    "functools",
    "operator",
    "heapq",
    "bisect",
    "copy",
    "string",
    "random",
    "datetime",
    "datetime.*",
    "array",
    "cmath",
]

DEFAULT_ALLOWED_DIRECTORIES = [
    "/testbed",
    "/tmp/agent",
]


class SandboxConfig(BaseModel):
    """Configuration for the secure code execution sandbox.

    Matches the moulinette evaluation schema and defaults.
    """

    authorized_imports: list[str] = Field(
        default_factory=lambda: list(DEFAULT_AUTHORIZED_IMPORTS),
        description="List of allowed import names (e.g., ['math', 'json']). Glob patterns supported.",
    )
    allowed_directories: list[str] = Field(
        default_factory=lambda: list(DEFAULT_ALLOWED_DIRECTORIES),
        description="List of filesystem paths the sandbox can access (e.g., ['/testbed', '/tmp/agent']).",
    )
    max_execution_time_seconds: int = Field(
        default=30,
        description="Maximum wall-clock time in seconds for a single sandbox execution.",
    )
    max_memory_mb: int = Field(
        default=512,
        description="Maximum memory in megabytes for sandbox execution.",
    )

    @classmethod
    def from_file(cls, path: Path | str) -> SandboxConfig:
        """Load and validate SandboxConfig from a JSON file."""
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Sandbox configuration file not found: {path}")
        with open(file_path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to a formatted JSON string."""
        return self.model_dump_json(indent=indent)


class ModelConfig(BaseModel):
    """Configuration for an LLM model and API provider endpoint."""

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


def load_env(env_file: Path | str | None = None, override: bool = True) -> bool:
    """Load environment variables from a .env file.

    Args:
        env_file: Specific path to a .env file (e.g. from --env-file).
                  If None, attempts to find and load a .env file automatically.
        override: Whether to override existing environment variables (default: True).

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

    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        return load_dotenv(dotenv_path=dotenv_path, override=override)
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
