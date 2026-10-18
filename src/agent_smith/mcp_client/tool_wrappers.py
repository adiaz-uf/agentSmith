"""Dynamic tool discovery and Python wrapper generation for MCP tool servers."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

from mcp.types import Tool

from agent_smith.mcp_client.client import MCPClient
from agent_smith.mcp_client.exceptions import MCPToolCallError

__all__ = [
    "ToolRegistry",
    "create_tool_signature",
    "create_tool_wrapper",
    "discover_tools",
    "sanitize_identifier",
]

logger = logging.getLogger(__name__)

JSON_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def sanitize_identifier(name: str) -> str:
    """Normalize a tool or parameter name into a valid Python identifier."""
    cleaned = name.replace("-", "_").replace(" ", "_").strip()
    return cleaned if cleaned.isidentifier() else f"tool_{cleaned}"


def create_tool_signature(tool: Tool) -> inspect.Signature:
    """Generate an inspect.Signature from a Tool's JSON input schema."""
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    properties: Mapping[str, Any] = schema.get("properties", {}) or {}
    required_set = set(schema.get("required", []) or [])

    sorted_props = sorted(properties.keys(), key=lambda p: (p not in required_set, p))
    parameters: list[inspect.Parameter] = []

    for prop in sorted_props:
        meta = properties[prop] or {}
        py_type = JSON_SCHEMA_TYPE_MAP.get(str(meta.get("type", "any")).lower(), Any)
        default = inspect.Parameter.empty if prop in required_set else meta.get("default", None)
        parameters.append(
            inspect.Parameter(
                name=sanitize_identifier(prop),
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default,
                annotation=py_type,
            )
        )

    return inspect.Signature(parameters=parameters)


def create_tool_wrapper(
    client: MCPClient,
    tool: Tool,
    *,
    unwrap_text: bool = False,
) -> Callable[..., Any]:
    """Create a callable Python function that delegates to client.call_tool()."""
    sig = create_tool_signature(tool)
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    properties: Mapping[str, Any] = schema.get("properties", {}) or {}
    name_map = {sanitize_identifier(k): k for k in properties}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"Invalid call to tool '{tool.name}': {exc}") from exc

        bound.apply_defaults()
        server_args = {name_map.get(k, k): v for k, v in bound.arguments.items()}
        result = client.call_tool(tool.name, server_args)

        if not unwrap_text:
            return result

        if result.is_error:
            err_msg = ""
            if result.content:
                err_msg = "\n".join(item.text for item in result.content if hasattr(item, "text"))
            raise MCPToolCallError(
                f"Tool '{tool.name}' failed on MCP server: {err_msg or 'Unknown error'}"
            )

        if result.content:
            texts = [item.text for item in result.content if hasattr(item, "text")]
            if texts:
                return "\n".join(texts)

        return result

    wrapper.__name__ = sanitize_identifier(tool.name)
    wrapper.__doc__ = tool.description or f"MCP tool: {tool.name}"
    wrapper.__signature__ = sig
    wrapper.__tool__ = tool  # type: ignore[attr-defined]
    wrapper.original_name = tool.name  # type: ignore[attr-defined]
    return wrapper


def discover_tools(
    client: MCPClient,
    *,
    unwrap_text: bool = False,
) -> dict[str, Callable[..., Any]]:
    """Discover all tools from the MCP server and return callable wrappers."""
    return {
        sanitize_identifier(t.name): create_tool_wrapper(client, t, unwrap_text=unwrap_text)
        for t in client.list_tools()
    }


class ToolRegistry:
    """Manages dynamic MCP tool wrappers within a sandbox namespace."""

    def __init__(
        self,
        namespace: dict[str, Any] | None = None,
        *,
        client: MCPClient | None = None,
        unwrap_text: bool = False,
    ) -> None:
        self._namespace = namespace if namespace is not None else {}
        self._unwrap_text = unwrap_text
        self._registered_tools: set[str] = set()
        if client is not None:
            self.sync_tools(client)

    @property
    def namespace(self) -> dict[str, Any]:
        return self._namespace

    @property
    def registered_tools(self) -> set[str]:
        return set(self._registered_tools)

    def sync_tools(self, client: MCPClient) -> dict[str, Callable[..., Any]]:
        """Discover tools from client, remove outdated wrappers, and inject new ones."""
        self.clear_tools()
        new_tools = discover_tools(client, unwrap_text=self._unwrap_text)
        for name, wrapper in new_tools.items():
            self._namespace[name] = wrapper
            self._registered_tools.add(name)
        return new_tools

    def clear_tools(self) -> None:
        """Remove previously registered MCP tools from namespace."""
        for name in self._registered_tools:
            self._namespace.pop(name, None)
        self._registered_tools.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._registered_tools

    def __repr__(self) -> str:
        return f"ToolRegistry(registered_tools={sorted(self._registered_tools)!r})"
