"""Tests for configuration loading, .env handling, and JSON schema validation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_mbpp.__main__ import parse_args as parse_mbpp_args
from agent_smith.core.config import (
    DEFAULT_ALLOWED_DIRECTORIES,
    DEFAULT_AUTHORIZED_IMPORTS,
    ModelConfig,
    SandboxConfig,
    get_api_key,
    load_env,
    load_model_config,
    load_sandbox_config,
)
from agent_smith.sandbox.cli import parse_args as parse_sandbox_args
from agent_swebench.__main__ import parse_args as parse_swebench_args


class TestEnvLoading:
    def test_load_env_missing_file_raises_error(self, tmp_path: Path):
        non_existent = tmp_path / ".non_existent_env"
        with pytest.raises(FileNotFoundError, match="Environment file not found"):
            load_env(non_existent)

    def test_load_env_custom_file(self, tmp_path: Path, monkeypatch):
        test_env_file = tmp_path / "custom.env"
        test_env_file.write_text("TEST_API_KEY=secret_12345\nANOTHER_VAR=hello\n")

        monkeypatch.delenv("TEST_API_KEY", raising=False)
        monkeypatch.delenv("ANOTHER_VAR", raising=False)

        loaded = load_env(test_env_file)
        assert loaded is True
        assert os.environ.get("TEST_API_KEY") == "secret_12345"
        assert os.environ.get("ANOTHER_VAR") == "hello"

    def test_load_env_default_when_none(self, monkeypatch):
        # Even without .env, should return a bool and not raise error
        result = load_env(None)
        assert isinstance(result, bool)

    def test_get_api_key_success(self, monkeypatch):
        monkeypatch.setenv("TEST_KEY_VAR", "my-valid-api-key")
        key = get_api_key("TEST_KEY_VAR", required=True)
        assert key == "my-valid-api-key"

    def test_get_api_key_missing_not_required(self, monkeypatch):
        monkeypatch.delenv("UNSET_KEY_VAR", raising=False)
        assert get_api_key("UNSET_KEY_VAR", required=False) is None

    def test_get_api_key_missing_required_raises(self, monkeypatch):
        monkeypatch.delenv("UNSET_KEY_VAR", raising=False)
        with pytest.raises(ValueError, match="API key missing"):
            get_api_key("UNSET_KEY_VAR", required=True)

    def test_get_api_key_empty_required_raises(self, monkeypatch):
        monkeypatch.setenv("EMPTY_KEY_VAR", "   ")
        with pytest.raises(ValueError, match="API key missing"):
            get_api_key("EMPTY_KEY_VAR", required=True)


class TestSandboxConfig:
    def test_sandbox_config_defaults(self):
        config = SandboxConfig()
        assert config.authorized_imports == DEFAULT_AUTHORIZED_IMPORTS
        assert config.allowed_directories == DEFAULT_ALLOWED_DIRECTORIES
        assert config.max_execution_time_seconds == 30
        assert config.max_memory_mb == 512

    def test_sandbox_config_from_file(self, tmp_path: Path):
        data = {
            "authorized_imports": ["math", "json"],
            "allowed_directories": ["/tmp/test"],
            "max_execution_time_seconds": 15,
            "max_memory_mb": 256,
        }
        config_file = tmp_path / "sandbox.json"
        config_file.write_text(json.dumps(data))

        config = SandboxConfig.from_file(config_file)
        assert config.authorized_imports == ["math", "json"]
        assert config.allowed_directories == ["/tmp/test"]
        assert config.max_execution_time_seconds == 15
        assert config.max_memory_mb == 256

    def test_sandbox_config_file_not_found(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            SandboxConfig.from_file(tmp_path / "does_not_exist.json")

    def test_sandbox_template_json_matches_schema(self):
        root_dir = Path(__file__).resolve().parent.parent
        template_file = root_dir / "sandbox_template.json"
        assert template_file.is_file(), "sandbox_template.json must exist at project root"

        config = load_sandbox_config(template_file)
        assert "math" in config.authorized_imports
        assert "/testbed" in config.allowed_directories
        assert config.max_execution_time_seconds == 30
        assert config.max_memory_mb == 512

    def test_sandbox_config_to_json(self):
        config = SandboxConfig()
        json_str = config.to_json()
        parsed = json.loads(json_str)
        assert parsed["max_execution_time_seconds"] == 30
        assert parsed["max_memory_mb"] == 512
        assert "math" in parsed["authorized_imports"]


class TestModelConfig:
    def test_model_config_initialization(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-mock-123")
        config = ModelConfig(
            model_name="qwen/qwen3-235b-a22b-2507",
            provider_url="https://openrouter.ai/api/v1",
        )
        assert config.model_name == "qwen/qwen3-235b-a22b-2507"
        assert config.provider_url == "https://openrouter.ai/api/v1"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.api_key == "sk-mock-123"

    def test_model_config_from_file(self, tmp_path: Path):
        data = {
            "model_name": "anthropic/claude-3.5-sonnet",
            "provider_url": "https://api.anthropic.com/v1",
            "api_key_env_var": "ANTHROPIC_API_KEY",
            "temperature": 0.2,
            "max_tokens": 8192,
        }
        config_file = tmp_path / "model.json"
        config_file.write_text(json.dumps(data))

        config = load_model_config(config_file)
        assert config.model_name == "anthropic/claude-3.5-sonnet"
        assert config.temperature == 0.2
        assert config.max_tokens == 8192

    def test_model_config_invalid_temperature(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                model_name="test-model",
                temperature=3.5,  # must be <= 2.0
            )


class TestCliEnvFileParsing:
    def test_agent_mbpp_args_parsing(self):
        args = parse_mbpp_args([
            "--task-file", "task.json",
            "--output", "sol.json",
            "--model-name", "my-model",
            "--provider-url", "https://api.test",
            "--env-file", "/custom/.env",
        ])
        assert args.task_file == "task.json"
        assert args.output == "sol.json"
        assert args.model_name == "my-model"
        assert args.provider_url == "https://api.test"
        assert args.env_file == "/custom/.env"

    def test_agent_swebench_args_parsing(self):
        args = parse_swebench_args([
            "--task-file", "task.json",
            "--output", "sol.json",
            "--model-name", "my-model",
            "--provider-url", "https://api.test",
            "--env-file", "/custom/.env",
        ])
        assert args.task_file == "task.json"
        assert args.output == "sol.json"
        assert args.env_file == "/custom/.env"

    def test_sandbox_args_parsing(self):
        args = parse_sandbox_args([
            "sandbox_template.json",
            "--env-file", ".env.local",
            "--mcp-stdio", "python tools.py",
        ])
        assert args.config_file == "sandbox_template.json"
        assert args.env_file == ".env.local"
        assert args.mcp_stdio == "python tools.py"


class TestSecurityHygiene:
    def test_env_example_has_no_real_keys(self):
        root_dir = Path(__file__).resolve().parent.parent
        env_example = root_dir / ".env.example"
        assert env_example.is_file(), ".env.example must exist"
        content = env_example.read_text()
        assert "sk-" not in content, "Found potential live secret in .env.example!"
        assert "OPENROUTER_API_KEY=" in content

    def test_gitignore_ignores_env_and_caches(self):
        root_dir = Path(__file__).resolve().parent.parent
        gitignore = (root_dir / ".gitignore").read_text()
        assert ".env" in gitignore
        assert "cache/" in gitignore
        assert "evaluations/" in gitignore
