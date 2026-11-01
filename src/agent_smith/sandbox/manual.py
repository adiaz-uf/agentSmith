"""Sandbox execution manual generation from live MCP tool schemas and configuration."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from mcp.types import Tool

from agent_smith.core.models import SandboxConfig
from agent_smith.mcp_client.client import MCPClient
from agent_smith.mcp_client.tool_wrappers import (
    create_tool_signature,
    sanitize_identifier,
    tool_namespace_names,
)

__all__ = [
    "format_tool_signature",
    "format_tools_manual",
    "generate_sandbox_manual",
]

logger = logging.getLogger(__name__)


def format_tool_signature(tool: Tool, name: str | None = None) -> str:
    """Format an MCP tool as the Python function signature + docstring the sandbox exposes."""
    fn_name = name or sanitize_identifier(tool.name)
    doc = tool.description or "No description provided."
    return f'def {fn_name}{create_tool_signature(tool)} -> Any:\n    """{doc}"""'


def format_tools_manual(tools: Sequence[Tool] | None) -> str:
    """Format all tool signatures into a unified documentation block."""
    if not tools:
        return "No external tools available."
    return "\n\n".join(format_tool_signature(t, name) for name, t in tool_namespace_names(tools))


def _context_block(client: MCPClient) -> str:
    """Document the server's resources and prompts and the helpers that reach them."""
    sections: list[str] = []
    try:
        resources = client.list_resources()
    except Exception as exc:  # noqa: BLE001 - servers may not support resources at all
        logger.debug("Server exposes no resources: %s", exc)
        resources = []
    if resources:
        lines = "\n".join(f"- `{r.uri}`: {r.description or r.name}" for r in resources)
        sections.append(
            "### Available Resources\n"
            "Read them with `read_resource(uri: str) -> str` "
            "(or `list_resources()` to refresh the list):\n" + lines
        )

    try:
        prompts = client.list_prompts()
    except Exception as exc:  # noqa: BLE001 - servers may not support prompts at all
        logger.debug("Server exposes no prompts: %s", exc)
        prompts = []
    if prompts:
        lines = "\n".join(
            f"- `{p.name}({', '.join(a.name for a in p.arguments or [])})`: {p.description or ''}"
            for p in prompts
        )
        sections.append(
            "### Available Prompts\n"
            "Render them with `get_prompt(name: str, **arguments: str) -> str` "
            "(or `list_prompts()` to refresh the list):\n" + lines
        )
    return "".join(f"\n{s}\n" for s in sections)


def generate_sandbox_manual(
    tools: Sequence[Tool] | None = None,
    config: SandboxConfig | None = None,
    *,
    client: MCPClient | None = None,
    include_context: bool = True,
) -> str:
    """Generate the Markdown sandbox manual for the LLM system prompt.

    With a ``client``, tools (and resources/prompts) are read live from the connected server,
    so switching servers changes the manual.
    """
    if tools is None and client is not None:
        tools = client.list_tools()
    cfg = config if config is not None else SandboxConfig()

    context_block = _context_block(client) if client is not None and include_context else ""
    allowed_imports = ", ".join(sorted(cfg.authorized_imports)) or "None"
    allowed_dirs = ", ".join(sorted(cfg.allowed_directories)) or "None"

    return f"""# Sandbox Execution Manual

## Protocol
Write executable Python code in each step to solve the problem.
Your code runs inside an isolated sandbox with the following built-in functions:

### Completion
- `final_answer(answer: Any) -> None`: Call this function to submit your final answer and terminate the solution.

### Available Tools
```python
{format_tools_manual(tools)}
```
{context_block}
## Security & Constraints
- **Execution Timeout**: {cfg.max_execution_time_seconds} seconds per step.
- **Memory Limit**: {cfg.max_memory_mb} MB.
- **Tool Output Limit**: {cfg.max_tool_output_chars} characters per tool/resource call; longer output ends with an `[output truncated: ...]` notice.
- **Allowed Imports**: {allowed_imports}
- **Allowed Directories**: {allowed_dirs}
- **Network**: Strict isolation (no inbound or outbound connections).
"""
