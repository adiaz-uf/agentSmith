"""Exceptions for the Model Context Protocol (MCP) client subsystem."""

from __future__ import annotations

__all__ = [
    "MCPConnectionError",
    "MCPError",
    "MCPServerCrashedError",
    "MCPTimeoutError",
    "MCPToolCallError",
]


from agent_smith.core.errors import AgentSmithError


class MCPError(AgentSmithError):
    """Base exception for all MCP client errors."""


class MCPConnectionError(MCPError):
    """The MCP server could not be spawned or the connection handshake failed."""


class MCPServerCrashedError(MCPError):
    """The MCP server subprocess terminated unexpectedly."""


class MCPTimeoutError(MCPError):
    """An operation with the MCP server exceeded the configured timeout."""


class MCPToolCallError(MCPError):
    """The MCP server returned an error response when executing a tool."""
