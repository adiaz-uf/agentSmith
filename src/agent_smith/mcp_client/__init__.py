"""Model Context Protocol (MCP) client: stdio and streamable HTTP transports."""

from __future__ import annotations

from agent_smith.mcp_client.client import MCPClient
from agent_smith.mcp_client.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPServerCrashedError,
    MCPTimeoutError,
    MCPToolCallError,
)
from agent_smith.mcp_client.tool_wrappers import (
    ToolRegistry,
    create_tool_signature,
    create_tool_wrapper,
    discover_tools,
    sanitize_identifier,
)
from agent_smith.mcp_client.transports import (
    build_stdio_params,
    parse_stdio_command,
    validate_http_url,
)

__all__ = [
    "MCPClient",
    "MCPConnectionError",
    "MCPError",
    "MCPServerCrashedError",
    "MCPTimeoutError",
    "MCPToolCallError",
    "ToolRegistry",
    "build_stdio_params",
    "create_tool_signature",
    "create_tool_wrapper",
    "discover_tools",
    "parse_stdio_command",
    "sanitize_identifier",
    "validate_http_url",
]
