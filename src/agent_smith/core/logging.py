from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

__all__ = [
    "JsonLogFormatter",
    "SecretMaskingFilter",
    "get_logger",
    "setup_logging",
]

_SECRET_PATTERNS = [
    re.compile(r"sk-[a-zA-Z0-9_-]{20,}", re.IGNORECASE),
    re.compile(r"Bearer\s+[a-zA-Z0-9_\-\.]{20,}", re.IGNORECASE),
]


class SecretMaskingFilter(logging.Filter):
    """Logging filter that redacts API keys and sensitive secrets from all log records."""

    def __init__(self, additional_secrets: list[str] | None = None) -> None:
        super().__init__()
        self._additional_secrets: set[str] = set()
        if additional_secrets:
            for s in additional_secrets:
                if s and len(s.strip()) >= 6:
                    self._additional_secrets.add(s.strip())

    def _get_secrets(self) -> set[str]:
        """Dynamically collect known secrets from environment variables and configured secrets."""
        secrets = set(self._additional_secrets)
        for key, val in os.environ.items():
            key_upper = key.upper()
            if any(term in key_upper for term in ("API_KEY", "SECRET", "TOKEN", "PASSWORD")):
                val_clean = val.strip()
                if len(val_clean) >= 8:
                    secrets.add(val_clean)
        return secrets

    def redact(self, text: str) -> str:
        """Redact known secrets and patterns from a string."""
        if not text:
            return text

        # Sort secrets by length descending to prevent partial replacements of nested tokens
        secrets = sorted(self._get_secrets(), key=len, reverse=True)
        for secret in secrets:
            if secret in text:
                text = text.replace(secret, "[REDACTED]")

        for pattern in _SECRET_PATTERNS:
            text = pattern.sub("[REDACTED]", text)

        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self.redact(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: self.redact(str(val)) if isinstance(val, str) else val
                    for key, val in record.args.items()
                }
            elif isinstance(record.args, tuple):
                redacted_args: list[Any] = []
                for arg in record.args:
                    if isinstance(arg, dict):
                        redacted_args.append(
                            {
                                key: self.redact(str(val)) if isinstance(val, str) else val
                                for key, val in arg.items()
                            }
                        )
                    elif isinstance(arg, str):
                        redacted_args.append(self.redact(arg))
                    else:
                        redacted_args.append(arg)
                record.args = tuple(redacted_args)

        return True


class JsonLogFormatter(logging.Formatter):
    """Structured JSON formatter for production evaluation logs."""

    def __init__(
        self,
        fmt: str | None = None,
        datefmt: str | None = None,
        redactor: SecretMaskingFilter | None = None,
    ) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.redactor = redactor or SecretMaskingFilter()

    def format(self, record: logging.LogRecord) -> str:
        raw_message = record.getMessage()
        redacted_message = self.redactor.redact(raw_message)

        data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": redacted_message,
        }
        if hasattr(record, "task_id"):
            data["task_id"] = str(record.task_id)
        if hasattr(record, "step"):
            data["step"] = record.step
        if record.exc_info:
            formatted_exc = self.formatException(record.exc_info)
            data["exception"] = self.redactor.redact(formatted_exc)
        return json.dumps(data)


def get_logger(name: str = "agent_smith") -> logging.Logger:
    """Retrieve logger with attached SecretMaskingFilter."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, SecretMaskingFilter) for f in logger.filters):
        logger.addFilter(SecretMaskingFilter())
    return logger


def setup_logging(
    level: int = logging.INFO,
    json_format: bool = False,
    stream: Any = None,
) -> logging.Logger:
    """Configure the agent_smith root logger with redacting formatter and filter."""
    logger = get_logger("agent_smith")
    logger.setLevel(level)
    handler = logging.StreamHandler(stream)
    redactor = SecretMaskingFilter()
    handler.addFilter(redactor)
    if json_format:
        handler.setFormatter(JsonLogFormatter(redactor=redactor))
    else:
        fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
    logger.handlers.clear()
    logger.addHandler(handler)
    return logger
