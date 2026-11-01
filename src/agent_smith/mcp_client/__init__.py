"""Model Context Protocol (MCP) client: transports, tool wrappers, resources and prompts."""

from __future__ import annotations

from agent_smith.mcp_client.client import MCPClient
from agent_smith.mcp_client.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPPromptError,
    MCPResourceError,
    MCPServerCrashedError,
    MCPTimeoutError,
    MCPToolCallError,
)
from agent_smith.mcp_client.resources_prompts import (
    MCPContextBridge,
    read_resource_content,
    render_prompt,
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
    "MCPContextBridge",
    "MCPError",
    "MCPPromptError",
    "MCPResourceError",
    "MCPServerCrashedError",
    "MCPTimeoutError",
    "MCPToolCallError",
    "ToolRegistry",
    "build_stdio_params",
    "create_tool_signature",
    "create_tool_wrapper",
    "discover_tools",
    "parse_stdio_command",
    "read_resource_content",
    "render_prompt",
    "sanitize_identifier",
    "validate_http_url",
]
