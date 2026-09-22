"""Entry point for `uv run sandbox`.

The full CLI is specified in subject V.2.1 and implemented in the sandbox epic:

    uv run sandbox
    uv run sandbox sandbox_template.json
    uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py" sandbox_template.json
    uv run sandbox --mcp-server <URL>
"""

import sys


def main() -> int:
    """Launch the interactive sandbox REPL."""
    print("sandbox: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
