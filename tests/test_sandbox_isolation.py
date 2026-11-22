"""Tests for subprocess-based sandbox isolation strategy (Issue #18)."""

from __future__ import annotations

import time

import pytest

from agent_smith.core.errors import SandboxError
from agent_smith.core.models import SandboxConfig
from agent_smith.sandbox.exceptions import SandboxCrashedError, SandboxTimeoutError
from agent_smith.sandbox.runner import ProcessSandbox, build_sandbox_env


def test_sandbox_basic_execution() -> None:
    """Test running basic Python code that prints to stdout."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute("print('Hello from sandbox')")
        assert res.success is True
        assert res.stdout == "Hello from sandbox\n"
        assert res.stderr == ""
        assert res.error is None
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
        with pytest.raises(SandboxTimeoutError) as exc_info:
            sandbox.execute("while True: pass", timeout=0.5)
        duration = time.perf_counter() - start

        assert isinstance(exc_info.value, SandboxError)
        assert "timeout" in str(exc_info.value).lower()
        assert exc_info.value.timeout == 0.5
        assert sandbox.is_alive is False
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
        with pytest.raises(SandboxCrashedError) as exc_info:
            sandbox.execute("import os\nos._exit(7)")
        assert isinstance(exc_info.value, SandboxError)
        assert exc_info.value.exit_code == 7
        assert "terminated unexpectedly" in str(exc_info.value)

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


def test_sandbox_builtins_tampering_does_not_break_worker_or_survive_reset() -> None:
    """Mutating __builtins__ must not crash the worker nor persist after reset()."""
    with ProcessSandbox() as sandbox:
        res1 = sandbox.execute("del __builtins__['isinstance']\n__builtins__['print'] = None")
        assert res1.success is True

        # Worker internals (json, traceback, run_loop) keep working.
        res2 = sandbox.execute("1 / 0")
        assert res2.error_type == "ZeroDivisionError"
        assert sandbox.is_alive is True

        sandbox.reset()

        res3 = sandbox.execute("print(isinstance(1, int))")
        assert res3.success is True
        assert res3.stdout == "True\n"


def test_sandbox_worker_does_not_inherit_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Host environment variables such as API keys must not reach the worker."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    monkeypatch.setenv("AGENT_SMITH_TEST_TOKEN", "token-value")
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(
            "import os\n"
            "[k for k in ('OPENAI_API_KEY', 'AGENT_SMITH_TEST_TOKEN') if k in os.environ]"
        )
        assert res.success is True
        assert res.return_value == "[]"


def test_build_sandbox_env_only_forwards_allowlisted_variables() -> None:
    env = build_sandbox_env({"PATH": "/usr/bin", "ANTHROPIC_API_KEY": "secret", "HOME": "/root"})
    assert env["PATH"] == "/usr/bin"
    assert "ANTHROPIC_API_KEY" not in env
    assert "HOME" not in env
    assert "PYTHONPATH" in env


def test_sandbox_user_writes_to_real_stdout_do_not_corrupt_ipc() -> None:
    """Writing directly to fd 1 / sys.__stdout__ must not break the IPC channel."""
    with ProcessSandbox() as sandbox:
        res = sandbox.execute(
            "import os, sys\nos.write(1, b'garbage\\n')\nsys.__stdout__.write('more\\n')\n"
            "sys.__stdout__.flush()\n'ok'"
        )
        assert res.success is True
        assert res.return_value == "'ok'"
