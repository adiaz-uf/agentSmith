"""Transport configuration and process parameters for MCP connections."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

# MCP SDK Dataclass to configure a subprocess communicating via standard input/output.
from mcp import StdioServerParameters

__all__ = [
    "build_stdio_params",
    "parse_stdio_command",
]


def parse_stdio_command(command: str | Sequence[str]) -> tuple[str, list[str]]:
    """Parse a command string or sequence into an executable and list of arguments.

    Handles shell quoting, whitespace normalization, and prevents empty invocations.

    Args:
        command: Either a raw command-line string (e.g. "python mcp_tools.py --flag")
            or a pre-tokenized sequence of strings.

    Returns:
        A tuple of (executable, list_of_arguments).

    Raises:
        ValueError: If command is empty or whitespace-only.
    """
    if isinstance(command, str):
        cleaned = command.strip()
        if not cleaned:
            raise ValueError("MCP stdio command cannot be empty.")
        tokens = shlex.split(cleaned)
    else:
        tokens = [str(arg).strip() for arg in command if str(arg).strip()]
        if not tokens:
            raise ValueError("MCP stdio command sequence cannot be empty.")

    executable = tokens[0]
    args = list(tokens[1:])
    return executable, args


def build_stdio_params(
    command: str | Sequence[str],
    *,
    env: Mapping[str, str | int | float | bool] | None = None,
    cwd: str | Path | None = None,
) -> StdioServerParameters:
    """Build and validate a StdioServerParameters instance for an MCP server process.

    Inherits the current OS environment variables, applies overrides (coercing values to strings),
    and verifies the target working directory if provided.

    Args:
        command: Command string or sequence specifying the server executable and arguments.
        env: Optional dictionary of environment variable overrides.
        cwd: Optional working directory for the server process.

    Returns:
        Validated StdioServerParameters ready for MCP transport.

    Raises:
        ValueError: If the command is empty.
        FileNotFoundError: If the specified cwd does not exist.
    """
    executable, args = parse_stdio_command(command)

    # Inherit current OS environment
    merged_env = dict(os.environ)
    if env:
        for k, v in env.items():
            merged_env[str(k)] = str(v)

    # Validate working directory early to avoid obscure subprocess spawn failures
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
