"""Dynamic tool discovery and Python wrapper generation for MCP tool servers."""

from __future__ import annotations

import builtins
import inspect
import keyword
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from mcp.types import Tool

from agent_smith.mcp_client.client import MCPClient
from agent_smith.mcp_client.exceptions import MCPError
from agent_smith.mcp_client.resources_prompts import (
    DEFAULT_MAX_OUTPUT_CHARS,
    MCPContextBridge,
    truncate_output,
)

__all__ = [
    "ToolRegistry",
    "create_tool_signature",
    "create_tool_wrapper",
    "discover_tools",
    "sanitize_identifier",
    "tool_namespace_names",
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
    """Normalize a tool or parameter name into a valid, non-keyword Python identifier."""
    cleaned = re.sub(r"\W", "_", name.strip())
    if not cleaned or not cleaned.isidentifier():
        cleaned = f"tool_{cleaned}"
    if keyword.iskeyword(cleaned) or keyword.issoftkeyword(cleaned):
        cleaned = f"{cleaned}_"
    return cleaned


def _unique(name: str, taken: set[str]) -> str:
    """Append underscores until name doesn't collide with taken."""
    while name in taken:
        name = f"{name}_"
    return name


def _schema_params(tool: Tool) -> list[tuple[str, str, Mapping[str, Any], bool]]:
    """Return (python_name, server_name, meta, required) per schema property, required first."""
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None) or {}
    properties: Mapping[str, Any] = schema.get("properties", {}) or {}
    required_set = set(schema.get("required", []) or [])

    params: list[tuple[str, str, Mapping[str, Any], bool]] = []
    taken: set[str] = set()
    for prop in sorted(properties, key=lambda p: (p not in required_set, p)):
        py_name = _unique(sanitize_identifier(prop), taken)
        taken.add(py_name)
        params.append((py_name, prop, properties[prop] or {}, prop in required_set))
    return params


def create_tool_signature(tool: Tool) -> inspect.Signature:
    """Generate an inspect.Signature from a Tool's JSON input schema."""
    return inspect.Signature(
        parameters=[
            inspect.Parameter(
                name=py_name,
                kind=inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=inspect.Parameter.empty if required else meta.get("default", None),
                annotation=JSON_SCHEMA_TYPE_MAP.get(str(meta.get("type", "any")).lower(), Any),
            )
            for py_name, _, meta, required in _schema_params(tool)
        ]
    )


def create_tool_wrapper(
    client: MCPClient,
    tool: Tool,
    *,
    unwrap_text: bool = False,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> Callable[..., Any]:
    """Create a callable Python function that delegates to client.call_tool()."""
    sig = create_tool_signature(tool)
    name_map = {py_name: server_name for py_name, server_name, _, _ in _schema_params(tool)}

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = sig.bind(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"Invalid call to tool '{tool.name}': {exc}") from exc

        # Only forward what the caller passed; the server applies its own defaults.
        server_args = {name_map[k]: v for k, v in bound.arguments.items()}
        try:
            result = client.call_tool(tool.name, server_args)
        except MCPError as exc:
            if not unwrap_text:
                raise
            # Sandbox mode: failures are observations for the LLM, never crashes.
            return truncate_output(f"Error: tool '{tool.name}' failed: {exc}", max_output_chars)

        if not unwrap_text:
            return result

        text = "\n".join(item.text for item in result.content or [] if hasattr(item, "text"))
        if result.is_error:
            text = f"Error: tool '{tool.name}' failed: {text or 'Unknown error'}"
        return truncate_output(text, max_output_chars) if text else result

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
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> dict[str, Callable[..., Any]]:
    """Discover all tools from the MCP server and return callable wrappers.

    Tools whose sanitized names collide get underscores appended (``a.b`` / ``a_b`` -> ``a_b_``).
    """
    return {
        name: create_tool_wrapper(
            client, t, unwrap_text=unwrap_text, max_output_chars=max_output_chars
        )
        for name, t in tool_namespace_names(client.list_tools())
    }


def tool_namespace_names(tools: Iterable[Tool]) -> list[tuple[str, Tool]]:
    """Pair each tool with the unique Python name it gets in the sandbox namespace."""
    named: list[tuple[str, Tool]] = []
    taken: set[str] = set()
    for t in tools:
        name = _unique(sanitize_identifier(t.name), taken)
        taken.add(name)
        named.append((name, t))
    return named


class ToolRegistry:
    """Manages dynamic MCP tool wrappers within a sandbox namespace."""

    def __init__(
        self,
        namespace: dict[str, Any] | None = None,
        *,
        client: MCPClient | None = None,
        unwrap_text: bool = False,
        expose_context: bool = False,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self._namespace = namespace if namespace is not None else {}
        self._unwrap_text = unwrap_text
        self._expose_context = expose_context
        self._max_output_chars = max_output_chars
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
        """Discover tools from client, remove outdated wrappers, and inject new ones.

        Discovery runs first, so a failing server leaves the current tools in place.
        With ``expose_context`` the resource/prompt helpers (``read_resource``,
        ``get_prompt``...) are injected too and win over same-named MCP tools.
        Anything that would shadow a non-MCP namespace entry (e.g. ``final_answer``) or a
        Python builtin is skipped with a warning.
        """
        new_tools = discover_tools(
            client, unwrap_text=self._unwrap_text, max_output_chars=self._max_output_chars
        )
        helpers: dict[str, Callable[..., Any]] = {}
        if self._expose_context:
            helpers = MCPContextBridge(
                client, max_output_chars=self._max_output_chars
            ).namespace_functions()
        self.clear_tools()
        injected: dict[str, Callable[..., Any]] = {}
        for name, wrapper in [*helpers.items(), *new_tools.items()]:
            if name in self._namespace or hasattr(builtins, name):
                logger.warning("Skipping MCP tool %r: name is reserved in the sandbox", name)
                continue
            self._namespace[name] = wrapper
            self._registered_tools.add(name)
            injected[name] = wrapper
        return injected

    def clear_tools(self) -> None:
        """Remove previously registered MCP tools from namespace."""
        for name in self._registered_tools:
            self._namespace.pop(name, None)
        self._registered_tools.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._registered_tools

    def __repr__(self) -> str:
        return f"ToolRegistry(registered_tools={sorted(self._registered_tools)!r})"
