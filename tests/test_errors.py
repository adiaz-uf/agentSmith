"""Tests for domain error hierarchy, CLI error formatting, and evaluation crash recording."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_smith.core.errors import (
    AgentSmithError,
    ConfigurationError,
    LLMError,
    SandboxError,
    TaskLoadingError,
    format_cli_error,
    handle_cli_error,
    record_evaluation_crash,
)
from agent_smith.core.models import SolutionOutput
from agent_smith.mcp_client.exceptions import MCPError


class TestErrorHierarchy:
    def test_custom_exceptions_inherit_from_agent_smith_error(self):
        assert issubclass(ConfigurationError, AgentSmithError)
        assert issubclass(TaskLoadingError, AgentSmithError)
        assert issubclass(LLMError, AgentSmithError)
        assert issubclass(SandboxError, AgentSmithError)
        assert issubclass(MCPError, AgentSmithError)
        assert issubclass(AgentSmithError, Exception)

    def test_exceptions_can_be_raised_and_caught(self):
        with pytest.raises(AgentSmithError):
            raise ConfigurationError("Invalid provider URL")

        with pytest.raises(AgentSmithError):
            raise TaskLoadingError("Corrupted task input")

        with pytest.raises(AgentSmithError):
            raise LLMError("Rate limit exceeded")

        with pytest.raises(AgentSmithError):
            raise SandboxError("Disallowed import attempt")


class TestCliErrorFormatting:
    def test_format_configuration_error(self):
        exc = ConfigurationError("Missing OPENROUTER_API_KEY")
        msg = format_cli_error(exc)
        assert msg == "Configuration Error: Missing OPENROUTER_API_KEY"

    def test_format_task_loading_error(self):
        exc = TaskLoadingError("JSON decode failure")
        msg = format_cli_error(exc)
        assert msg == "Task Error: JSON decode failure"

    def test_format_base_domain_error(self):
        exc = AgentSmithError("Generic failure")
        msg = format_cli_error(exc)
        assert msg == "Error: Generic failure"

    def test_format_file_not_found_error(self):
        exc = FileNotFoundError("File not found: /path/to/missing")
        msg = format_cli_error(exc)
        assert msg == "File not found: /path/to/missing"

    def test_format_unexpected_error_standard(self):
        exc = RuntimeError("Null pointer inside generator")
        msg = format_cli_error(exc, verbose=False)
        assert msg == "Unexpected Error: Null pointer inside generator"

    def test_format_unexpected_error_verbose(self):
        exc = ZeroDivisionError("division by zero")
        msg = format_cli_error(exc, verbose=True)
        assert msg == "Unexpected Error (ZeroDivisionError): division by zero"

    def test_handle_cli_error_outputs_to_stderr_and_returns_one(self, capsys):
        exc = ConfigurationError("Bad config value")
        code = handle_cli_error(exc, command_name="test_cmd")
        assert code == 1
        captured = capsys.readouterr()
        assert "test_cmd: Configuration Error: Bad config value\n" == captured.err


class TestRecordEvaluationCrash:
    def test_record_crash_creates_failed_solution_output(self):
        sol = record_evaluation_crash(
            task_id="42",
            benchmark="mbpp",
            error=ValueError("Evaluation failed"),
            total_time_seconds=3.5,
        )
        assert isinstance(sol, SolutionOutput)
        assert sol.task_id == "42"
        assert sol.benchmark == "mbpp"
        assert sol.success is False
        assert sol.solution == ""
        assert sol.error == "ValueError: Evaluation failed"
        assert sol.total_time_seconds == 3.5

    def test_record_crash_with_normalized_benchmark(self):
        sol = record_evaluation_crash(
            task_id="sympy-123",
            benchmark="SWEBENCH",
            error="String error representation",
        )
        assert sol.benchmark == "swebench"
        assert sol.error == "String error representation"

    def test_record_crash_invalid_benchmark_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid benchmark"):
            record_evaluation_crash(
                task_id="1",
                benchmark="unknown_benchmark",
                error=RuntimeError("crash"),
            )

    def test_record_crash_writes_solution_json_atomically(self, tmp_path: Path):
        output_file = tmp_path / "output" / "solution.json"
        record_evaluation_crash(
            task_id="101",
            benchmark="mbpp",
            error=TimeoutError("Sandbox exceeded 120s"),
            output_path=output_file,
            total_time_seconds=120.0,
        )
        assert output_file.is_file()
        loaded = SolutionOutput.model_validate_json(output_file.read_text(encoding="utf-8"))
        assert loaded.task_id == "101"
        assert loaded.success is False
        assert loaded.error == "TimeoutError: Sandbox exceeded 120s"
        assert loaded.total_time_seconds == 120.0
