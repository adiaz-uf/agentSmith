"""Exceptions for the sandbox subsystem."""

from __future__ import annotations

__all__ = [
    "SandboxCrashedError",
    "SandboxError",
    "SandboxIPCError",
    "SandboxTimeoutError",
]

from agent_smith.core.errors import SandboxError


class SandboxTimeoutError(SandboxError):
    """Code execution exceeded the allowed wall-clock time; the worker was killed."""

    def __init__(self, message: str, timeout: float, elapsed: float) -> None:
        super().__init__(message)
        self.timeout = timeout
        self.elapsed = elapsed


class SandboxCrashedError(SandboxError):
    """The worker subprocess terminated unexpectedly (e.g. ``os._exit``, segfault, OOM)."""

    def __init__(self, message: str, exit_code: int | None = None) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class SandboxIPCError(SandboxError):
    """Communication with the worker subprocess failed or returned a malformed message."""
