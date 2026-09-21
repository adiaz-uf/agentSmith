"""Entry point for `uv run python -m agent_swebench`.

    uv run python -m agent_swebench --task-file <task.json> \
        --output <solution.json> \
        --model-name "model/name" --provider-url "https://provider.api/v1"
"""

import sys


def main() -> int:
    """Run the SWE-bench agent on a single task."""
    print("agent_swebench: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
