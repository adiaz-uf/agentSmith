"""Subprocess worker loop for executing sandboxed Python code."""

from __future__ import annotations

import ast
import builtins
import io
import json
import os
import sys
import time
import traceback
from typing import Any, BinaryIO

from agent_smith.core.models import DEFAULT_AUTHORIZED_IMPORTS
from agent_smith.sandbox.imports import create_safe_import
from agent_smith.sandbox.protocol import (
    ExecutionResult,
    decode_ipc_message,
    encode_ipc_message,
)

# Builtins entries that give access to the import machinery itself.
_STRIPPED_BUILTINS = frozenset({"__loader__", "__spec__"})


class SandboxWorker:
    """Worker maintaining state and executing code inside an isolated child process."""

    def __init__(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.config = config or {}
        self.namespace: dict[str, Any] = self._create_initial_namespace()

    def _create_initial_namespace(self) -> dict[str, Any]:
        """Create a fresh execution namespace.

        ``__builtins__`` is a per-namespace *copy* of the builtins module dict, so
        user code doing ``del __builtins__['isinstance']`` neither breaks the
        worker's own code (json, traceback, run_loop) nor survives a reset.
        ``__import__`` is replaced with an allowlist-enforcing import, and
        ``__loader__`` / ``__spec__`` (the ``BuiltinImporter``) are dropped so
        built-in modules cannot be loaded without going through it.
        """
        authorized = self.config.get("authorized_imports", DEFAULT_AUTHORIZED_IMPORTS)

        safe_builtins = {
            name: value
            for name, value in builtins.__dict__.items()
            if name not in _STRIPPED_BUILTINS
        }
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
            terminated=terminated,
        )

    def _send(self, message: dict[str, Any]) -> None:
        self.writer.write(encode_ipc_message(message))
        self.writer.flush()

    def _send_error(self, message: str) -> None:
        self._send({"status": "error", "message": message})

    def run_loop(self) -> None:
        """Main IPC loop handling incoming commands until shutdown or EOF."""
        while True:
            try:
                payload = self.reader.readline()
            except OSError:
                break
            if not payload:
                break

            try:
                message = decode_ipc_message(payload)
            except (ValueError, TypeError) as exc:
                self._send_error(f"Malformed IPC payload: {exc}")
                continue

            cmd = message.get("cmd")
            if cmd == "shutdown":
                break
            if cmd == "reset":
                self.reset_namespace()
                self._send({"status": "ok"})
            elif cmd == "execute":
                result = self.execute_code(message.get("code", ""))
                self._send({"status": "ok", "result": result.to_dict()})
                if result.terminated:
                    break
            else:
                self._send_error(f"Unknown IPC command: {cmd!r}")


def _detach_ipc_channel() -> tuple[BinaryIO, BinaryIO]:
    """Move the host pipes off fds 0/1 and point those fds at /dev/null.

    ``os.dup`` returns non-inheritable descriptors, so grandchildren spawned by
    user code do not receive the IPC channel either.
    """
    ipc_in = os.dup(0)
    ipc_out = os.dup(1)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(devnull, 1)
    os.close(devnull)
    return os.fdopen(ipc_in, "rb"), os.fdopen(ipc_out, "wb")


def main(argv: list[str] | None = None) -> int:
    """Entrypoint executed inside the spawned worker subprocess."""
    args = sys.argv[1:] if argv is None else argv
    config = json.loads(args[0]) if args else {}
    reader, writer = _detach_ipc_channel()
    try:
        SandboxWorker(reader=reader, writer=writer, config=config).run_loop()
    finally:
        reader.close()
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
