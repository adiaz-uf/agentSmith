"""Entry point for `uv run python -m agent_swebench`.

    uv run python -m agent_swebench --task-file <task.json> \
        --output <solution.json> \
        --model-name "model/name" --provider-url "https://provider.api/v1" \
        [--env-file <path_to_.env>]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from agent_smith.core.config import load_env


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SWE-bench agent on a single task.")
    parser.add_argument("--task-file", type=str, help="Path to the SWE-bench task input JSON file.")
    parser.add_argument("--output", type=str, help="Path to write the solution JSON output.")
    parser.add_argument("--model-name", type=str, help="Model identifier (e.g. 'model/name').")
    parser.add_argument("--provider-url", type=str, help="Base URL of the LLM API provider.")
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file for loading environment variables and API keys.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SWE-bench agent on a single task."""
    args = parse_args(argv)
    try:
        load_env(args.env_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"agent_swebench: {exc}", file=sys.stderr)
        return 1
    print("agent_swebench: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
