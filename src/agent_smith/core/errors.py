"""Error hierarchy and centralized CLI error formatting for Agent Smith."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from agent_smith.core.models import (
    SolutionOutput,
    current_iso_timestamp,
    write_solution_json
)

class AgentSmithError(Exception):
    """Base exception for all domain errors in Agent Smith."""


class ConfigurationError(AgentSmithError):
    """Raised when environment variables, CLI options or settings are invalid."""


class TaskLoadingError(AgentSmithError):
    """Raised when a benchmark task input file is missing or malformed."""


class LLMError(AgentSmithError):
    """Raised when LLM API requests fail or exhaust retry limits."""


class SandboxError(AgentSmithError):
    """Raised on sandbox isolation failures, timeouts or permission violations."""


def format_cli_error(exc: Exception, verbose: bool = False) -> str:
    """Format an exception into a concise CLI error message"""
    if isinstance(exc, ConfigurationError):
        return f"Configuration Error: {exc}"
    if isinstance(exc, TaskLoadingError):
        return f"Task Error: {exc}"
    if isinstance(exc, AgentSmithError):
        return f"Error: {exc}"
    if verbose:
        return f"Unexpected Error ({type(exc).__name__}): {exc}"
    return f"Unexpected Error: {exc}"

def handle_cli_error(
    exc: Exception, command_name: str = "agent-smith", verbose: bool = False
) -> int:
    """Print formatted error to stderr and return exit code 1."""
    msg = format_cli_error(exc, verbose=verbose)
    print(f"[{command_name}] {msg}", file=sys.stderr)
    return 1

def record_evaluation_crash(
    task_id: str,
    benchmark: str,
    error: Exception,
    output_path: Path | str | None = None,
    total_time_seconds: float = 0.0,
) -> SolutionOutput:
    """Construct a failed SolutionOutput upon unhandled exception and write solution.json."""
    sol = SolutionOutput(
        task_id=task_id,
        benchmark=benchmark,  # type: ignore[arg-type]
        success=False,
        solution="",
        iterations=0,
        total_requests=0,
        total_input_tokens=0,
        total_output_tokens=0,
        total_time_seconds=total_time_seconds,
        error=f"{type(error).__name__}: {error}",
        timestamp=current_iso_timestamp(),
    )
    if output_path is not None:
        write_solution_json(sol, output_path)
    return sol