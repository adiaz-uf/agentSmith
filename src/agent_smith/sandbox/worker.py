"""Subprocess worker loop for executing sandboxed Python code."""

from __future__ import annotations

import ast
import builtins
import io
import sys
import time
import traceback
from multiprocessing.connection import Connection
from typing import Any

from agent_smith.core.models import DEFAULT_AUTHORIZED_IMPORTS
from agent_smith.sandbox.imports import create_safe_import
from agent_smith.sandbox.protocol import (
    ExecutionResult,
    decode_ipc_message,
    encode_ipc_message,
)


class SandboxWorker:
    """Worker maintaining state and executing code inside an isolated child process."""

    def __init__(self, connection: Connection, config: dict[str, Any] | None = None) -> None:
        self.connection = connection
        self.config = config or {}
        self.namespace: dict[str, Any] = self._create_initial_namespace()

    def _create_initial_namespace(self) -> dict[str, Any]:
        """Create a fresh execution namespace."""
        authorized = self.config.get("authorized_imports", DEFAULT_AUTHORIZED_IMPORTS)

        real_builtins = (
            __builtins__ if isinstance(__builtins__, dict) else builtins.__dict__
        )
        safe_builtins = dict(real_builtins)
        safe_builtins["__import__"] = create_safe_import(authorized)

        return {
            "__name__": "__main__",
            "__doc__": None,
            "__builtins__": safe_builtins,
        }

    def reset_namespace(self) -> None:
        """Reset the execution namespace to a clean initial state."""
        self.namespace = self._create_initial_namespace()

    def execute_code(self, code: str) -> ExecutionResult:
        """Execute Python code within the persistent namespace and capture all outputs."""
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        sys.stdout = stdout_buf
        sys.stderr = stderr_buf

        start_time = time.perf_counter()
        return_value_repr: str | None = None
        error_msg: str | None = None
        error_type: str | None = None
        terminated = False

        try:
            parsed = ast.parse(code)
            # If code ends with an expression (e.g. 2 + 2), evaluate it to capture its value (REPL style)
            if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                last_expr = parsed.body.pop()
                if parsed.body:
                    statements_code = compile(
                        ast.Module(body=parsed.body, type_ignores=[]),
                        filename="<sandbox>",
                        mode="exec",
                    )
                    exec(statements_code, self.namespace)  # noqa: S102 (exec is intentional: core mechanism of the sandbox)

                expr_code = compile(
                    ast.Expression(body=last_expr.value),
                    filename="<sandbox>",
                    mode="eval",
                )
                val = eval(expr_code, self.namespace)
                if val is not None:
                    return_value_repr = repr(val)
            else:
                compiled = compile(code, filename="<sandbox>", mode="exec")
                exec(compiled, self.namespace)  # noqa: S102 (exec is intentional: core mechanism of the sandbox)

        except SystemExit as exc:
            terminated = True
            error_type = "SystemExit"
            error_msg = f"SystemExit({exc.code})"
        except KeyboardInterrupt:
            terminated = True
            error_type = "KeyboardInterrupt"
            error_msg = "Execution interrupted by KeyboardInterrupt"
        except BaseException as exc:  # noqa: BLE001 (catch all runtime exceptions to return traceback to agent)
            error_type = type(exc).__name__
            tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
            error_msg = "".join(tb_lines).strip()
        finally:
            elapsed = time.perf_counter() - start_time
            sys.stdout = original_stdout
            sys.stderr = original_stderr

        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            return_value=return_value_repr,
            error=error_msg,
            error_type=error_type,
            execution_time=elapsed,
            is_timeout=False,
            terminated=terminated,
        )

    def run_loop(self) -> None:
        """Main IPC loop handling incoming execution commands."""
        while True:
            try:
                payload = self.connection.recv_bytes()
            except (EOFError, BrokenPipeError, ConnectionResetError):
                break

            try:
                message = decode_ipc_message(payload)
            except Exception as exc:  # noqa: BLE001 (catch all runtime exceptions to return traceback to agent)
                err_result = ExecutionResult(
                    error=f"Malformed IPC payload: {exc}",
                    error_type="IPCError",
                )
                self.connection.send_bytes(encode_ipc_message(err_result.to_dict()))
                continue

            cmd = message.get("cmd")
            if cmd == "shutdown":
                break
            elif cmd == "reset":
                self.reset_namespace()
                self.connection.send_bytes(encode_ipc_message({"status": "ok"}))
            elif cmd == "execute":
                code = message.get("code", "")
                result = self.execute_code(code)
                self.connection.send_bytes(encode_ipc_message(result.to_dict()))
                if result.terminated:
                    break
            else:
                err_result = ExecutionResult(
                    error=f"Unknown IPC command: {cmd}",
                    error_type="IPCError",
                )
                self.connection.send_bytes(encode_ipc_message(err_result.to_dict()))


def worker_entrypoint(connection: Connection, config: dict[str, Any] | None = None) -> None:
    """Entrypoint executed inside the spawned worker subprocess."""
    worker = SandboxWorker(connection=connection, config=config)
    worker.run_loop()
