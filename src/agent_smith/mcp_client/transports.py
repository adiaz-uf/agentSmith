"""Transport configuration and process parameters for MCP connections."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlparse

# MCP SDK Dataclass to configure a subprocess communicating via standard input/output.
from mcp import StdioServerParameters

__all__ = [
    "build_stdio_params",
    "parse_stdio_command",
    "validate_http_url",
]


def parse_stdio_command(command: str | Sequence[str]) -> tuple[str, list[str]]:
    """Parse a command string or sequence into an executable and list of arguments."""
    if isinstance(command, str):
        cleaned = command.strip()
        if not cleaned:
            raise ValueError("MCP stdio command cannot be empty.")
        tokens = shlex.split(cleaned)
    else:
        tokens = [str(arg).strip() for arg in command if str(arg).strip()]
        if not tokens:
            raise ValueError("MCP stdio command sequence cannot be empty.")

    return tokens[0], list(tokens[1:])


def build_stdio_params(
    command: str | Sequence[str],
    *,
    env: Mapping[str, str | int | float | bool] | None = None,
    cwd: str | Path | None = None,
) -> StdioServerParameters:
    """Build and validate StdioServerParameters for an MCP server process."""
    executable, args = parse_stdio_command(command)

    merged_env = dict(os.environ)
    if env:
        for k, v in env.items():
            merged_env[str(k)] = str(v)

    resolved_cwd: Path | None = None
    if cwd is not None:
        resolved_cwd = Path(cwd).resolve()
        if not resolved_cwd.is_dir():
            raise FileNotFoundError(
                f"MCP server working directory does not exist or is not a directory: {resolved_cwd}"
            )

    return StdioServerParameters(
        command=executable,
        args=args,
        env=merged_env,
        cwd=resolved_cwd,
    )


def validate_http_url(url: str) -> str:
    """Validate and normalize an HTTP or HTTPS URL for MCP streamable HTTP transport."""
    if not isinstance(url, str):
        raise TypeError(f"MCP server URL must be a string, got {type(url).__name__}.")

    cleaned = url.strip()
    if not cleaned:
        raise ValueError("MCP server URL cannot be empty or whitespace-only.")

    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"Unsupported MCP URL scheme: '{parsed.scheme}'. Only 'http' and 'https' are supported."
        )

    if not parsed.netloc:
        raise ValueError(f"Invalid MCP URL: '{cleaned}' is missing host/network location.")

    return cleaned
