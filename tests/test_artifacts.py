"""Tests for evaluation artifact paths, atomic writing, and stdio log capturing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from agent_smith.core.artifacts import EvaluationArtifacts, sanitize_task_id
from agent_smith.core.models import SolutionOutput


class TestSanitizeTaskId:
    def test_replaces_slashes_and_colons(self):
        assert sanitize_task_id("MBPP/11") == "MBPP_11"
        assert sanitize_task_id("task:foo/bar") == "task_foo_bar"
        assert sanitize_task_id("foo\\bar") == "foo_bar"

    def test_preserves_valid_filename_characters(self):
        assert sanitize_task_id("django__django-11099") == "django__django-11099"
        assert sanitize_task_id("test.task-1") == "test.task-1"
        assert sanitize_task_id(42) == "42"


class TestEvaluationArtifacts:
    def test_directory_structure_resolution(self, tmp_path: Path):
        artifacts = EvaluationArtifacts(
            eval_type="MBPP",
            task_id="MBPP/10",
            base_dir=tmp_path,
            timestamp_str="2026-10-11_12-30-00",
        )
        expected_dir = tmp_path / "mbpp" / "2026-10-11_12-30-00" / "MBPP_10"
        assert artifacts.run_dir == expected_dir
        assert artifacts.task_json_path == expected_dir / "task.json"
        assert artifacts.solution_json_path == expected_dir / "solution.json"
        assert artifacts.stdout_log_path == expected_dir / "stdout.log"
        assert artifacts.stderr_log_path == expected_dir / "stderr.log"

    def test_save_task_json_from_dict(self, tmp_path: Path):
        artifacts = EvaluationArtifacts("mbpp", "1", base_dir=tmp_path)
        task_data = {"task_id": 1, "prompt": "Solve this"}
        saved_path = artifacts.save_task_json(task_data)

        assert saved_path == artifacts.task_json_path
        assert saved_path.is_file()
        content = json.loads(saved_path.read_text(encoding="utf-8"))
        assert content == task_data

    def test_save_task_json_from_string(self, tmp_path: Path):
        artifacts = EvaluationArtifacts("swebench", "django-1", base_dir=tmp_path)
        task_str = json.dumps({"instance_id": "django-1", "problem": "bug"})
        saved_path = artifacts.save_task_json(task_str)

        assert saved_path.is_file()
        content = json.loads(saved_path.read_text(encoding="utf-8"))
        assert content["instance_id"] == "django-1"

    def test_save_solution_json_atomic(self, tmp_path: Path):
        artifacts = EvaluationArtifacts("mbpp", "42", base_dir=tmp_path)
        sol = SolutionOutput(
            task_id="42",
            benchmark="mbpp",
            success=True,
            solution="def solution(): pass",
            iterations=1,
            total_requests=1,
            total_input_tokens=100,
            total_output_tokens=50,
            total_time_seconds=1.2,
        )
        saved_path = artifacts.save_solution_json(sol)

        assert saved_path == artifacts.solution_json_path
        assert saved_path.is_file()
        loaded = SolutionOutput.model_validate_json(saved_path.read_text(encoding="utf-8"))
        assert loaded.task_id == "42"
        assert loaded.success is True
        assert loaded.solution == "def solution(): pass"

    def test_capture_stdio_logs_and_redact_secrets(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv("SECRET_EVAL_KEY", "secret_value_in_stdio_123456")
        artifacts = EvaluationArtifacts("mbpp", "100", base_dir=tmp_path)

        with artifacts.capture_stdio_logs():
            print("Step 1 started")
            print("Calling model with key secret_value_in_stdio_123456 in stdout")
            print("Warning: internal error", file=sys.stderr)

        assert artifacts.stdout_log_path.is_file()
        assert artifacts.stderr_log_path.is_file()

        stdout_content = artifacts.stdout_log_path.read_text(encoding="utf-8")
        stderr_content = artifacts.stderr_log_path.read_text(encoding="utf-8")

        assert "Step 1 started" in stdout_content
        assert "secret_value_in_stdio_123456" not in stdout_content
        assert "[REDACTED]" in stdout_content
        assert "Warning: internal error" in stderr_content
