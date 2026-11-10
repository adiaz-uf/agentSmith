"""Tests for subprocess-based sandbox isolation strategy (Issue #18)."""

from __future__ import annotations

import time

from agent_smith.core.models import SandboxConfig
from agent_smith.sandbox.runner import ProcessSandbox


def test_sandbox_basic_execution() -> None:
    """Test running basic Python code that prints to stdout."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("print('Hello from sandbox')")
        assert res.success is True
        assert res.stdout == "Hello from sandbox\n"
        assert res.stderr == ""
        assert res.error is None
        assert res.is_timeout is False
        assert res.terminated is False


def test_sandbox_expression_return_value() -> None:
    """Test that evaluating a final expression captures its representation."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("x = 20\nx + 22")
        assert res.success is True
        assert res.return_value == "42"


def test_sandbox_stderr_capture() -> None:
    """Test capturing output written to sys.stderr."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("import sys\nsys.stderr.write('Warning message\\n')")
        assert res.success is True
        assert res.stderr == "Warning message\n"


def test_sandbox_state_persistence_across_executions() -> None:
    """Test that variables and functions persist across separate execute calls."""
    with ProcessSandbox() as sandbox:
        res1 = sandbox.execute(
            "def greet(name):\n    return f'Hello, {name}!'\nmessage = greet('Alice')"
        )
        assert res1.success is True

        res2 = sandbox.execute("print(message)")
        assert res2.success is True
        assert res2.stdout == "Hello, Alice!\n"


def test_sandbox_reset_clears_state() -> None:
    """Test that resetting the sandbox clears previous variables."""
    with ProcessSandbox() as sandbox:
        res1 = sandbox.execute("persistent_var = 999")
        assert res1.success is True

        sandbox.reset()

        res2 = sandbox.execute("print(persistent_var)")
        assert res2.success is False
        assert res2.error_type == "NameError"
        assert "persistent_var" in (res2.error or "")


def test_sandbox_standard_exception_handling() -> None:
    """Test that standard exceptions are captured without terminating the worker."""
    with ProcessSandbox() as sandbox:
        res1 = sandbox.execute("1 / 0")
        assert res1.success is False
        assert res1.error_type == "ZeroDivisionError"
        assert res1.is_timeout is False
        assert res1.terminated is False
        assert sandbox.is_alive is True

        # Worker should remain alive for subsequent executions
        res2 = sandbox.execute("print('Alive after error')")
        assert res2.success is True
        assert res2.stdout == "Alive after error\n"


def test_sandbox_timeout_enforcement() -> None:
    """Test that an infinite loop is killed after the specified timeout."""
    config = SandboxConfig(max_execution_time_seconds=1)
    sandbox = ProcessSandbox(config=config)
    try:
        start = time.perf_counter()
        res = sandbox.execute("while True: pass", timeout=0.5)
        duration = time.perf_counter() - start

        assert res.is_timeout is True
        assert res.terminated is True
        assert res.success is False
        assert "timeout" in (res.error or "").lower()
        assert 0.4 <= duration < 2.0

        # Worker should restart cleanly on next call
        res_after = sandbox.execute("print('Restarted')")
        assert res_after.success is True
        assert res_after.stdout == "Restarted\n"
    finally:
        sandbox.close()


def test_sandbox_crash_recovery() -> None:
    """Test that hard process exits (e.g. os._exit) are handled and recovered from."""
    with ProcessSandbox() as sandbox:
        res1 = sandbox.execute("import os\nos._exit(7)")
        assert res1.success is False
        assert res1.terminated is True
        assert "terminated unexpectedly" in (res1.error or "")

        # Worker should transparently recover on subsequent call
        res2 = sandbox.execute("print('Recovered after crash')")
        assert res2.success is True
        assert res2.stdout == "Recovered after crash\n"


def test_sandbox_system_exit_handling() -> None:
    """Test handling of sys.exit() calls without crashing host orchestrator."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("import sys\nsys.exit(42)")
        assert res.success is False
        assert res.terminated is True
        assert res.error_type == "SystemExit"

