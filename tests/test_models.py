"""Tests for Pydantic data models (StepMetrics, SolutionOutput, SandboxConfig, TaskInputs)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_smith.core.models import (
    DEFAULT_ALLOWED_DIRECTORIES,
    DEFAULT_AUTHORIZED_IMPORTS,
    MBPPTaskInput,
    SandboxConfig,
    SolutionOutput,
    StepMetrics,
    SWEBenchTaskInput,
    load_task_input,
    write_solution_json,
)


class TestStepMetrics:
    def test_valid_step_metrics(self):
        step = StepMetrics(
            step=1,
            input_tokens=150,
            output_tokens=42,
            request_time_ms=350.5,
            api_url="https://openrouter.ai/api/v1",
            model_name="qwen/qwen-2.5-coder-32b",
            llm_output="```python\nprint('hello')\n```",
            sandbox_input="print('hello')",
            sandbox_output="hello\n",
            retries=0,
        )
        assert step.step == 1
        assert step.input_tokens == 150
        assert step.output_tokens == 42
        assert step.request_time_ms == 350.5
        assert step.timestamp is not None
        assert "T" in step.timestamp

    def test_step_metrics_defaults_for_empty_strings(self):
        step = StepMetrics(
            step=1,
            input_tokens=0,
            output_tokens=0,
            request_time_ms=0.0,
        )
        assert step.api_url == ""
        assert step.model_name == ""
        assert step.llm_output == ""
        assert step.sandbox_input == ""
        assert step.sandbox_output == ""
        assert step.retries == 0

    def test_step_metrics_validation_constraints(self):
        with pytest.raises(ValidationError, match="step"):
            StepMetrics(step=0, input_tokens=0, output_tokens=0, request_time_ms=0.0)

        with pytest.raises(ValidationError, match="input_tokens"):
            StepMetrics(step=1, input_tokens=-1, output_tokens=0, request_time_ms=0.0)

        with pytest.raises(ValidationError, match="output_tokens"):
            StepMetrics(step=1, input_tokens=0, output_tokens=-1, request_time_ms=0.0)

        with pytest.raises(ValidationError, match="request_time_ms"):
            StepMetrics(step=1, input_tokens=0, output_tokens=0, request_time_ms=-0.5)

        with pytest.raises(ValidationError, match="retries"):
            StepMetrics(step=1, input_tokens=0, output_tokens=0, request_time_ms=0.0, retries=-1)

    def test_step_metrics_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra"):
            StepMetrics(
                step=1,
                input_tokens=10,
                output_tokens=10,
                request_time_ms=10.0,
                unknown_field="invalid",
            )


class TestSolutionOutput:
    def test_valid_solution_output_mbpp(self):
        step = StepMetrics(
            step=1,
            input_tokens=100,
            output_tokens=50,
            request_time_ms=200.0,
            sandbox_output="OK",
        )
        sol = SolutionOutput(
            task_id="11",
            benchmark="mbpp",
            success=True,
            solution="def remove_Occ(s, ch):\n    return s.replace(ch, '', 1)",
            iterations=1,
            total_requests=1,
            total_input_tokens=100,
            total_output_tokens=50,
            total_time_seconds=1.25,
            steps=[step],
            system_prompt="You are an autonomous code agent.",
        )
        assert sol.task_id == "11"
        assert sol.benchmark == "mbpp"
        assert sol.success is True
        assert len(sol.steps) == 1
        assert sol.error is None
        assert sol.timestamp is not None

    def test_valid_solution_output_swebench_failure(self):
        sol = SolutionOutput(
            task_id="sympy__sympy-14711",
            benchmark="swebench",
            success=False,
            solution="",
            iterations=10,
            total_requests=10,
            total_input_tokens=50000,
            total_output_tokens=4000,
            total_time_seconds=120.0,
            error="Iteration budget exhausted without passing tests",
        )
        assert sol.success is False
        assert sol.solution == ""
        assert sol.error == "Iteration budget exhausted without passing tests"

    def test_solution_output_rejects_invalid_benchmark(self):
        with pytest.raises(ValidationError, match="benchmark"):
            SolutionOutput(
                task_id="1",
                benchmark="invalid_benchmark",
                success=True,
                iterations=0,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_time_seconds=0.0,
            )

    def test_solution_output_rejects_extra_fields(self):
        with pytest.raises(ValidationError, match="extra"):
            SolutionOutput(
                task_id="1",
                benchmark="mbpp",
                success=True,
                iterations=0,
                total_requests=0,
                total_input_tokens=0,
                total_output_tokens=0,
                total_time_seconds=0.0,
                foo="bar",
            )

    def test_write_solution_json_atomic(self, tmp_path: Path):
        output_path = tmp_path / "nested" / "evaluations" / "solution.json"
        sol = SolutionOutput(
            task_id="11",
            benchmark="mbpp",
            success=True,
            solution="def foo(): pass",
            iterations=1,
            total_requests=1,
            total_input_tokens=50,
            total_output_tokens=25,
            total_time_seconds=0.5,
        )

        written_path = write_solution_json(sol, output_path)
        assert written_path == output_path.resolve()
        assert output_path.is_file()

        raw_data = json.loads(output_path.read_text(encoding="utf-8"))
        assert raw_data["task_id"] == "11"
        assert raw_data["benchmark"] == "mbpp"
        assert raw_data["success"] is True

        # Ensure it can be re-loaded into SolutionOutput
        reloaded = SolutionOutput.model_validate(raw_data)
        assert reloaded.task_id == sol.task_id

    def test_solution_output_write_to_file_method(self, tmp_path: Path):
        output_path = tmp_path / "solution.json"
        sol = SolutionOutput(
            task_id="sympy__sympy-14711",
            benchmark="swebench",
            success=False,
            solution="",
            iterations=5,
            total_requests=5,
            total_input_tokens=2000,
            total_output_tokens=500,
            total_time_seconds=15.0,
            error="Aborted by user",
        )
        sol.write_to_file(output_path)
        assert output_path.is_file()

        loaded = json.loads(output_path.read_text())
        assert loaded["error"] == "Aborted by user"


class TestSandboxConfigModels:
    def test_sandbox_config_defaults(self):
        config = SandboxConfig()
        assert config.max_execution_time_seconds == 30
        assert config.max_memory_mb == 512
        assert config.authorized_imports == DEFAULT_AUTHORIZED_IMPORTS
        assert config.allowed_directories == DEFAULT_ALLOWED_DIRECTORIES

    def test_sandbox_config_loads_template_file(self):
        root = Path(__file__).resolve().parent.parent
        template_file = root / "sandbox_template.json"
        config = SandboxConfig.from_file(template_file)
        assert config.max_execution_time_seconds == 30
        assert config.max_memory_mb == 512
        assert "math" in config.authorized_imports
        assert "/testbed" in config.allowed_directories

    def test_sandbox_config_from_file_missing_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="Sandbox configuration file not found"):
            SandboxConfig.from_file(tmp_path / "missing.json")


class TestTaskInputLoaders:
    def test_mbpp_task_input_from_dict(self):
        data = {
            "task_id": 11,
            "task_definition": "Write a python function to remove first occurrence of a given character from a string.",
            "function_definition": "def remove_Occ(s, ch):",
            "test_imports": [],
            "test_list": ["assert remove_Occ('hello', 'l') == 'heo'"],
        }
        task = MBPPTaskInput.model_validate(data)
        assert task.task_id == 11
        assert task.task_definition.startswith("Write a python function")
        assert len(task.test_list) == 1

    def test_mbpp_task_input_from_file(self, tmp_path: Path):
        data = {
            "task_id": "mbpp_42",
            "task_definition": "Sort a list",
            "function_definition": "def sort_list(lst):",
            "test_imports": ["import math"],
            "test_list": ["assert sort_list([3, 1, 2]) == [1, 2, 3]"],
        }
        file_path = tmp_path / "mbpp_task.json"
        file_path.write_text(json.dumps(data))

        task = MBPPTaskInput.from_file(file_path)
        assert task.task_id == "mbpp_42"
        assert task.test_imports == ["import math"]

    def test_mbpp_task_input_missing_field_raises(self):
        data = {
            "task_id": 11,
            # missing task_definition
            "function_definition": "def foo():",
            "test_list": ["assert foo() == 1"],
        }
        with pytest.raises(ValidationError, match="task_definition"):
            MBPPTaskInput.model_validate(data)

    def test_mbpp_task_input_empty_test_list_raises(self):
        data = {
            "task_id": 11,
            "task_definition": "Do something",
            "function_definition": "def foo():",
            "test_list": [],
        }
        with pytest.raises(ValidationError, match="test_list"):
            MBPPTaskInput.model_validate(data)

    def test_swebench_task_input_from_dict(self):
        data = {
            "instance_id": "sympy__sympy-14711",
            "problem_statement": "Vector add error with non-trivial matrix",
            "docker_image": "swebench/sweb.eval.x86_64.sympy_14711:v1",
            "eval_script": "./eval.sh",
            "hints_text": "Check sympy/physics/vector/vector.py",
            "repo": "sympy/sympy",
        }
        task = SWEBenchTaskInput.model_validate(data)
        assert task.instance_id == "sympy__sympy-14711"
        assert task.repo == "sympy/sympy"
        assert task.hints_text == "Check sympy/physics/vector/vector.py"

    def test_swebench_task_input_from_file(self, tmp_path: Path):
        data = {
            "instance_id": "pydata__xarray-4629",
            "problem_statement": "merge attrs broken",
            "docker_image": "swebench/xarray:v1",
            "eval_script": "pytest",
            "repo": "pydata/xarray",
        }
        file_path = tmp_path / "swebench_task.json"
        file_path.write_text(json.dumps(data))

        task = SWEBenchTaskInput.from_file(file_path)
        assert task.instance_id == "pydata__xarray-4629"
        assert task.hints_text == ""

    def test_load_task_input_helper(self, tmp_path: Path):
        mbpp_file = tmp_path / "mbpp.json"
        mbpp_file.write_text(
            json.dumps(
                {
                    "task_id": 1,
                    "task_definition": "desc",
                    "function_definition": "def f():",
                    "test_list": ["assert f() == 0"],
                }
            )
        )
        task_mbpp = load_task_input(mbpp_file, "mbpp")
        assert isinstance(task_mbpp, MBPPTaskInput)

        swe_file = tmp_path / "swe.json"
        swe_file.write_text(
            json.dumps(
                {
                    "instance_id": "id1",
                    "problem_statement": "stmt",
                    "docker_image": "img",
                    "eval_script": "run",
                    "repo": "owner/repo",
                }
            )
        )
        task_swe = load_task_input(swe_file, "swebench")
        assert isinstance(task_swe, SWEBenchTaskInput)

        with pytest.raises(ValueError, match="Unknown benchmark"):
            load_task_input(mbpp_file, "unknown")
