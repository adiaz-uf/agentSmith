"""Entry point for `uv run python -m agent_mbpp`.

    uv run python -m agent_mbpp --task-file <task.json> \
        --output <solution.json> \
        --model-name "model/name" --provider-url "https://provider.api/v1"
"""

import sys


def main() -> int:
    """Run the MBPP agent on a single task."""
    print("agent_mbpp: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
