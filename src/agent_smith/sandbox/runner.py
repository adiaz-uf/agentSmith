"""Host-side manager for sandboxed code execution in an isolated subprocess."""

from __future__ import annotations

import multiprocessing as mp
import time
from multiprocessing.connection import Connection
from types import TracebackType

from agent_smith.core.models import SandboxConfig
from agent_smith.sandbox.protocol import (
    ExecutionResult,
    decode_ipc_message,
    encode_ipc_message,
)
from agent_smith.sandbox.worker import worker_entrypoint


class ProcessSandbox:
    """Manages an isolated child worker process running sandboxed code."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._ctx = mp.get_context("spawn")
        self._parent_conn: Connection | None = None
        self._child_conn: Connection | None = None
        self._proc: mp.Process | None = None

    @property
    def is_alive(self) -> bool:
        """Return True if the worker subprocess is currently alive and healthy."""
        return self._proc is not None and self._proc.is_alive()

    def start(self) -> None:
        """Start the worker subprocess if not already running."""
        if self.is_alive:
            return

        self._cleanup_process()

        self._parent_conn, self._child_conn = self._ctx.Pipe(duplex=True)
        config_dict = self.config.model_dump() if self.config else {}

        self._proc = self._ctx.Process(
            target=worker_entrypoint,
            args=(self._child_conn, config_dict),
            daemon=True,
        )
        self._proc.start()

        # Close child connection end in host so EOF is detected if child terminates
        if self._child_conn is not None:
            self._child_conn.close()
            self._child_conn = None

    def _kill_worker(self) -> None:
        """Forcefully terminate and clean up the worker process."""
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=0.5)
            if self._proc.is_alive():
                self._proc.kill()
                self._proc.join(timeout=0.5)
        self._cleanup_process()

    def _cleanup_process(self) -> None:
        """Close pipes and clear process references."""
        if self._parent_conn is not None:
            try:
                self._parent_conn.close()
            except Exception:  # noqa: BLE001, S110
                pass
            self._parent_conn = None

        if self._child_conn is not None:
            try:
                self._child_conn.close()
            except Exception:  # noqa: BLE001, S110
                pass
            self._child_conn = None

        self._proc = None

    def close(self) -> None:
        """Gracefully shut down the sandbox worker process."""
        if self.is_alive and self._parent_conn is not None:
            try:
                self._parent_conn.send_bytes(encode_ipc_message({"cmd": "shutdown"}))
                if self._proc is not None:
                    self._proc.join(timeout=0.5)
            except Exception:  # noqa: BLE001, S110
                pass

        self._kill_worker()

    def reset(self) -> None:
        """Reset the execution namespace of the worker process, or restart if dead."""
        if not self.is_alive or self._parent_conn is None:
            self.start()
            return

        try:
            self._parent_conn.send_bytes(encode_ipc_message({"cmd": "reset"}))
            if self._parent_conn.poll(2.0):
                self._parent_conn.recv_bytes()
            else:
                self._kill_worker()
                self.start()
        except Exception:  # noqa: BLE001
            self._kill_worker()
            self.start()

    def execute(self, code: str, timeout: float | None = None) -> ExecutionResult:
        """Execute Python code in the sandbox subprocess and wait for results with timeout."""
        self.start()
        if self._parent_conn is None:
            return ExecutionResult(
                error="Failed to establish IPC connection with sandbox worker",
                error_type="SandboxError",
                terminated=True,
            )

        effective_timeout = (
            timeout if timeout is not None else float(self.config.max_execution_time_seconds)
        )

        start_time = time.perf_counter()

        try:
            self._parent_conn.send_bytes(encode_ipc_message({"cmd": "execute", "code": code}))
        except (BrokenPipeError, ConnectionResetError, OSError) as exc:
            self._kill_worker()
            return ExecutionResult(
                error=f"Failed to transmit code to sandbox worker: {exc}",
                error_type="IPCError",
                terminated=True,
            )

        # Wait for worker response within timeout
        has_response = self._parent_conn.poll(effective_timeout)
        elapsed = time.perf_counter() - start_time

        if not has_response:
            # Execution exceeded timeout - hard kill worker
            self._kill_worker()
            return ExecutionResult(
                error=f"Execution exceeded timeout of {effective_timeout:.1f}s",
                error_type="TimeoutError",
                execution_time=elapsed,
                is_timeout=True,
                terminated=True,
            )

        try:
            payload = self._parent_conn.recv_bytes()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            exit_code = self._proc.exitcode if self._proc else None
            self._kill_worker()
            return ExecutionResult(
                error=f"Sandbox worker process terminated unexpectedly (exit code: {exit_code})",
                error_type="WorkerCrashed",
                execution_time=elapsed,
                terminated=True,
            )

        try:
            data = decode_ipc_message(payload)
            result = ExecutionResult.from_dict(data)
        except Exception as exc:  # noqa: BLE001
            return ExecutionResult(
                error=f"Failed to decode worker response: {exc}",
                error_type="IPCError",
                execution_time=elapsed,
            )

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
