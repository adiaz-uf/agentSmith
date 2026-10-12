"""Tests for secret redaction in logging and structured JSON formatting."""

from __future__ import annotations

import json
import logging
from io import StringIO

import pytest

from agent_smith.core.logging import (
    JsonLogFormatter,
    SecretMaskingFilter,
    get_logger,
    setup_logging,
)


class TestSecretMaskingFilter:
    def test_redact_env_var_secrets(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-live-secret-key-1234567890")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-aws-key-9999")

        redactor = SecretMaskingFilter()
        text = "Calling API with sk-live-secret-key-1234567890 and super-secret-aws-key-9999"
        redacted = redactor.redact(text)

        assert "sk-live-secret-key-1234567890" not in redacted
        assert "super-secret-aws-key-9999" not in redacted
        assert redacted == "Calling API with [REDACTED] and [REDACTED]"

    def test_redact_regex_sk_patterns(self):
        redactor = SecretMaskingFilter()
        sample = "Key leaked: sk-abcdef1234567890123456 in debug dump"
        assert redactor.redact(sample) == "Key leaked: [REDACTED] in debug dump"

    def test_redact_bearer_patterns(self):
        redactor = SecretMaskingFilter()
        sample = "Header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token"
        assert redactor.redact(sample) == "Header: [REDACTED]"

    def test_redact_additional_secrets(self):
        redactor = SecretMaskingFilter(additional_secrets=["custom_vault_pass_xyz"])
        sample = "Decrypted using custom_vault_pass_xyz password"
        assert redactor.redact(sample) == "Decrypted using [REDACTED] password"

    def test_filter_modifies_log_record(self):
        redactor = SecretMaskingFilter(additional_secrets=["my_super_secret_token"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Token is: my_super_secret_token",
            args=(),
            exc_info=None,
        )
        redactor.filter(record)
        assert record.msg == "Token is: [REDACTED]"

    def test_filter_modifies_log_record_args_tuple(self):
        redactor = SecretMaskingFilter(additional_secrets=["secret_arg_value_123"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Secret %s",
            args=("secret_arg_value_123",),
            exc_info=None,
        )
        redactor.filter(record)
        assert record.args == ("[REDACTED]",)

    def test_filter_modifies_log_record_args_dict(self):
        redactor = SecretMaskingFilter(additional_secrets=["secret_arg_value_123"])
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=10,
            msg="Secret %(k)s",
            args=({"k": "secret_arg_value_123"},),
            exc_info=None,
        )
        redactor.filter(record)
        assert record.args == {"k": "[REDACTED]"}


class TestJsonLogFormatter:
    def test_formatter_produces_valid_json_with_metadata(self):
        formatter = JsonLogFormatter()
        record = logging.LogRecord(
            name="agent_smith.test",
            level=logging.WARNING,
            pathname=__file__,
            lineno=20,
            msg="Iteration timeout occurred",
            args=(),
            exc_info=None,
        )
        record.task_id = "MBPP_42"
        record.step = 3

        output = formatter.format(record)
        parsed = json.loads(output)

        assert parsed["logger"] == "agent_smith.test"
        assert parsed["level"] == "WARNING"
        assert parsed["message"] == "Iteration timeout occurred"
        assert parsed["task_id"] == "MBPP_42"
        assert parsed["step"] == 3
        assert "timestamp" in parsed

    def test_formatter_redacts_exceptions_with_secrets(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-topsecret-exception-key-12345")
        formatter = JsonLogFormatter()

        try:
            raise ValueError("Error with sk-topsecret-exception-key-12345 in stack")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="agent_smith",
            level=logging.ERROR,
            pathname=__file__,
            lineno=30,
            msg="Failure in LLM call",
            args=(),
            exc_info=exc_info,
        )

        output = formatter.format(record)
        parsed = json.loads(output)

        assert "sk-topsecret-exception-key-12345" not in parsed["exception"]
        assert "[REDACTED]" in parsed["exception"]


class TestLoggerHelpers:
    def test_get_logger_has_secret_masking_filter(self):
        logger = get_logger("test_unit_logger")
        has_filter = any(isinstance(f, SecretMaskingFilter) for f in logger.filters)
        assert has_filter is True

    def test_setup_logging_stream_redaction(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("CUSTOM_API_KEY", "secret_stream_key_987654")
        stream = StringIO()
        logger = setup_logging(level=logging.INFO, json_format=True, stream=stream)

        logger.info("Connecting using secret_stream_key_987654 now")
        log_line = stream.getvalue()

        assert "secret_stream_key_987654" not in log_line
        assert "[REDACTED]" in log_line
        parsed = json.loads(log_line)
        assert parsed["message"] == "Connecting using [REDACTED] now"
