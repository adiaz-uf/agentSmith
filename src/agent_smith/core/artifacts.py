"""Evaluation run directory and artifact lifecycle management (Subject VI.5)."""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from agent_smith.core.logging import SecretMaskingFilter
from agent_smith.core.models import SolutionOutput, write_solution_json

__all__ = [
    "EvaluationArtifacts",
    "sanitize_task_id",
]


def sanitize_task_id(task_id: str | int) -> str:
    """Sanitize task_id for filesystem path safety (e.g. 'MBPP/1' -> 'MBPP_1')."""
    return re.sub(r"[^\w\.-]", "_", str(task_id))


class _RedactingTee(io.TextIOBase):
    """Tee stream that duplicates output to a file stream with secret redaction."""

    def __init__(
        self,
        original_stream: Any,
        file_stream: io.TextIOBase,
        redactor: SecretMaskingFilter,
    ) -> None:
        self.original_stream = original_stream
        self.file_stream = file_stream
        self.redactor = redactor

    def write(self, s: str) -> int:
        if hasattr(self.original_stream, "write"):
            self.original_stream.write(s)
        redacted = self.redactor.redact(s)
        self.file_stream.write(redacted)
        self.file_stream.flush()
        return len(s)

    def flush(self) -> None:
        if hasattr(self.original_stream, "flush"):
            self.original_stream.flush()
        self.file_stream.flush()

    def isatty(self) -> bool:
        if hasattr(self.original_stream, "isatty"):
            return bool(self.original_stream.isatty())
        return False


class EvaluationArtifacts:
    """Manages evaluation directory structure and artifacts according to Subject VI.5.

    Directory structure:
    ./evaluations/EVAL_TYPE/YYYY-MM-DD_HH-MM-SS/task_id/{task.json, solution.json, stdout.log, stderr.log}
    """

    def __init__(
        self,
        eval_type: str,
        task_id: str | int,
        base_dir: Path | str = "./evaluations",
        timestamp_str: str | None = None,
    ) -> None:
        self.eval_type = str(eval_type).lower()
        self.raw_task_id = str(task_id)
        self.task_id = sanitize_task_id(task_id)

        if timestamp_str is None:
            now = datetime.now(timezone.utc)
            self.timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
        else:
            self.timestamp_str = timestamp_str

        self.run_dir = Path(base_dir).resolve() / self.eval_type / self.timestamp_str / self.task_id
        self.task_json_path = self.run_dir / "task.json"
        self.solution_json_path = self.run_dir / "solution.json"
        self.stdout_log_path = self.run_dir / "stdout.log"
        self.stderr_log_path = self.run_dir / "stderr.log"

    def initialize(self) -> Path:
        """Create the run directory structure on disk."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        return self.run_dir

    def save_task_json(self, task_data: dict[str, Any] | str | BaseModel) -> Path:
        """Save task.json copy into the evaluation directory."""
        self.initialize()
        if isinstance(task_data, BaseModel):
            data = task_data.model_dump()
        elif isinstance(task_data, str):
            data = json.loads(task_data)
        else:
            data = task_data

        with open(self.task_json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return self.task_json_path

    def save_solution_json(self, solution: SolutionOutput) -> Path:
        """Atomically write solution.json into the evaluation directory."""
        self.initialize()
        return write_solution_json(solution, self.solution_json_path)

    @contextlib.contextmanager
    def capture_stdio_logs(self) -> Iterator[None]:
        """Tee stdout and stderr into stdout.log and stderr.log with secret redaction."""
        self.initialize()
        redactor = SecretMaskingFilter()

        with contextlib.ExitStack() as stack:
            stdout_file = stack.enter_context(open(self.stdout_log_path, "a", encoding="utf-8"))
            stderr_file = stack.enter_context(open(self.stderr_log_path, "a", encoding="utf-8"))

            old_stdout = sys.stdout
            old_stderr = sys.stderr

            sys.stdout = _RedactingTee(old_stdout, stdout_file, redactor)  # type: ignore[assignment]
            sys.stderr = _RedactingTee(old_stderr, stderr_file, redactor)  # type: ignore[assignment]

            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
