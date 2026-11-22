"""Host-side manager for sandboxed code execution in an isolated subprocess."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, NoReturn

import agent_smith
from agent_smith.core.models import SandboxConfig
from agent_smith.sandbox.exceptions import (
    SandboxCrashedError,
    SandboxIPCError,
    SandboxTimeoutError,
)
from agent_smith.sandbox.protocol import (
    ExecutionResult,
    decode_ipc_message,
    encode_ipc_message,
)

WORKER_MODULE = "agent_smith.sandbox.worker"

# Only these host variables are forwarded to the worker. Everything else
# (API keys, tokens, credentials...) is dropped.
SANDBOX_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMPDIR",
)

_CONTROL_TIMEOUT_SECONDS = 2.0
_KILL_GRACE_SECONDS = 0.5


def build_sandbox_env(host_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build the minimal environment passed to the worker subprocess."""
    source = os.environ if host_env is None else host_env
    env = {key: source[key] for key in SANDBOX_ENV_ALLOWLIST if key in source}
    # Make the worker module importable without relying on the host's PYTHONPATH.
    env["PYTHONPATH"] = str(Path(agent_smith.__file__).resolve().parent.parent)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class ProcessSandbox:
    """Manages an isolated child worker process running sandboxed code."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._proc: subprocess.Popen[bytes] | None = None
        self._buffer = b""

    @property
    def is_alive(self) -> bool:
        """Return True if the worker subprocess is currently running."""
        return self._proc is not None and self._proc.poll() is None

    def start(self) -> None:
        """Start the worker subprocess if not already running."""
        if self.is_alive:
            return

        self._cleanup_process()
        config_json = json.dumps(self.config.model_dump())
        try:
            self._proc = subprocess.Popen(
                [sys.executable, "-m", WORKER_MODULE, config_json],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=build_sandbox_env(),
                close_fds=True,
            )
        except OSError as exc:
            raise SandboxCrashedError(f"Failed to start sandbox worker: {exc}") from exc

    def _kill_worker(self) -> None:
        """Forcefully terminate and clean up the worker process."""
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=_KILL_GRACE_SECONDS)
        self._cleanup_process()

    def _cleanup_process(self) -> None:
        """Close pipes and clear process references."""
        if self._proc is not None:
            for stream in (self._proc.stdin, self._proc.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        self._proc = None
        self._buffer = b""

    def _send(self, message: dict[str, Any]) -> None:
        """Send one framed message to the worker."""
        if self._proc is None or self._proc.stdin is None:
            raise SandboxIPCError("Sandbox worker is not running")
        try:
            self._proc.stdin.write(encode_ipc_message(message))
            self._proc.stdin.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            exit_code = self._proc.poll()
            self._kill_worker()
            raise SandboxCrashedError(
                f"Failed to transmit message to sandbox worker: {exc}", exit_code=exit_code
            ) from exc

    def _recv(self, timeout: float) -> dict[str, Any] | None:
        """Read one framed message from the worker.

        Returns ``None`` if no complete message arrived within ``timeout``.
        Raises :class:`EOFError` if the worker closed its end of the pipe.
        """
        if self._proc is None or self._proc.stdout is None:
            raise SandboxIPCError("Sandbox worker is not running")
        fd = self._proc.stdout.fileno()
        deadline = time.monotonic() + timeout
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            ready, _, _ = select.select([fd], [], [], remaining)
            if not ready:
                return None
            chunk = os.read(fd, 65536)
            if not chunk:
                raise EOFError
            self._buffer += chunk

        line, _, self._buffer = self._buffer.partition(b"\n")
        try:
            message = decode_ipc_message(line)
        except (ValueError, TypeError) as exc:
            self._kill_worker()
            raise SandboxIPCError(f"Failed to decode worker response: {exc}") from exc
        if message.get("status") == "error":
            raise SandboxIPCError(str(message.get("message", "Unknown worker error")))
        return message

    def _raise_crashed(self) -> NoReturn:
        """Collect the worker exit code, clean up and raise SandboxCrashedError."""
        exit_code: int | None = None
        if self._proc is not None:
            try:
                exit_code = self._proc.wait(timeout=_KILL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                exit_code = None
        self._kill_worker()
        raise SandboxCrashedError(
            f"Sandbox worker process terminated unexpectedly (exit code: {exit_code})",
            exit_code=exit_code,
        )

    def close(self) -> None:
        """Gracefully shut down the sandbox worker process."""
        if self.is_alive and self._proc is not None and self._proc.stdin is not None:
            try:
                self._proc.stdin.write(encode_ipc_message({"cmd": "shutdown"}))
                self._proc.stdin.flush()
                self._proc.wait(timeout=_KILL_GRACE_SECONDS)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._kill_worker()

    def reset(self) -> None:
        """Reset the execution namespace of the worker process, or restart it if unhealthy."""
        if not self.is_alive:
            self.start()
            return

        try:
            self._send({"cmd": "reset"})
            response = self._recv(_CONTROL_TIMEOUT_SECONDS)
        except (EOFError, SandboxCrashedError, SandboxIPCError):
            response = None
        if response is None:
            self._kill_worker()
            self.start()

    def execute(self, code: str, timeout: float | None = None) -> ExecutionResult:
        """Execute Python code in the sandbox subprocess.

        Errors raised by the user code are reported in the returned
        :class:`ExecutionResult`. Failures of the sandbox itself raise:

        Raises:
            SandboxTimeoutError: execution exceeded ``timeout`` (worker is killed
                and transparently restarted on the next call).
            SandboxCrashedError: the worker process died during execution.
            SandboxIPCError: the worker returned a malformed or error response.
        """
        self.start()
        effective_timeout = (
            timeout if timeout is not None else float(self.config.max_execution_time_seconds)
        )

        start_time = time.perf_counter()
        self._send({"cmd": "execute", "code": code})

        try:
            response = self._recv(effective_timeout)
        except EOFError:
            self._raise_crashed()
        elapsed = time.perf_counter() - start_time

        if response is None:
            self._kill_worker()
            raise SandboxTimeoutError(
                f"Execution exceeded timeout of {effective_timeout:.1f}s",
                timeout=effective_timeout,
                elapsed=elapsed,
            )

        result_data = response.get("result")
        if not isinstance(result_data, dict):
            self._kill_worker()
            raise SandboxIPCError("Worker response is missing the execution result")
        result = ExecutionResult.from_dict(result_data)

        if result.terminated:
            self._kill_worker()

        return result

    def __enter__(self) -> ProcessSandbox:  # noqa: PYI034
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
