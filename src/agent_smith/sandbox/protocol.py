"""Data model and wire format shared by the sandbox host and worker."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExecutionResult:
    """Result of executing code within the sandbox."""

    stdout: str = ""
    stderr: str = ""
    return_value: str | None = None
    error: str | None = None
    error_type: str | None = None
    execution_time: float = 0.0
    terminated: bool = False

    @property
    def success(self) -> bool:
        """Return True if execution succeeded without errors."""
        return self.error is None and not self.terminated

    def to_dict(self) -> dict[str, Any]:
        """Convert execution result to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExecutionResult:
        """Create execution result from dictionary."""
        return cls(
            stdout=data.get("stdout", ""),
            stderr=data.get("stderr", ""),
            return_value=data.get("return_value"),
            error=data.get("error"),
            error_type=data.get("error_type"),
            execution_time=float(data.get("execution_time", 0.0)),
            terminated=bool(data.get("terminated", False)),
        )


def encode_ipc_message(message: dict[str, Any]) -> bytes:
    """Encode a dictionary as a single newline-terminated UTF-8 JSON frame."""
    return json.dumps(message, ensure_ascii=False).encode("utf-8") + b"\n"


def decode_ipc_message(payload: bytes) -> dict[str, Any]:
    """Decode a UTF-8 JSON frame (with or without trailing newline) into a dictionary."""
    message = json.loads(payload.decode("utf-8"))
    if not isinstance(message, dict):
        raise TypeError(f"IPC message must be a JSON object, got {type(message).__name__}")
    return message
