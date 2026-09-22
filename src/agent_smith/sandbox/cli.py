"""Entry point for `uv run sandbox`.

The full CLI is specified in subject V.2.1 and implemented in the sandbox epic:

    uv run sandbox
    uv run sandbox sandbox_template.json
    uv run sandbox --mcp-stdio "python mcp_tools_mbpp.py" sandbox_template.json
    uv run sandbox --mcp-server <URL>
    uv run sandbox --env-file <path_to_.env>
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from agent_smith.core.config import load_env, load_sandbox_config


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive sandbox REPL CLI.")
    parser.add_argument(
        "config_file",
        nargs="?",
        default=None,
        help="Optional path to sandbox JSON config file (e.g. sandbox_template.json).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=None,
        help="Path to .env file for loading environment variables and API keys.",
    )
    parser.add_argument(
        "--mcp-stdio",
        type=str,
        default=None,
        help="Launch MCP tool server over stdio via command.",
    )
    parser.add_argument(
        "--mcp-server",
        type=str,
        default=None,
        help="Connect to running MCP server at URL.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Launch the interactive sandbox REPL."""
    args = parse_args(argv)
    load_env(args.env_file)
    if args.config_file:
        load_sandbox_config(args.config_file)
    print("sandbox: not implemented yet", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
