"""MCP resources and prompts handling and sandbox namespace exposure."""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

from mcp.types import Prompt, Resource

from agent_smith.mcp_client.client import MCPClient
from agent_smith.mcp_client.exceptions import MCPPromptError, MCPResourceError

__all__ = [
    "MCPContextBridge",
    "read_resource_content",
    "render_prompt",
]


def read_resource_content(client: MCPClient, uri: str) -> str:
    """Read resource by URI and return its text content."""
    try:
        result = client.read_resource(uri)
    except Exception as exc:
        raise MCPResourceError(f"Failed to read resource '{uri}': {exc}") from exc

    chunks: list[str] = []
    for item in result.contents:
        if hasattr(item, "text") and item.text is not None:
            chunks.append(item.text)
        elif hasattr(item, "blob") and item.blob:
            try:
                decoded = base64.b64decode(item.blob).decode("utf-8")
                chunks.append(decoded)
            except (ValueError, UnicodeDecodeError):
                chunks.append(f"[Binary data: {len(item.blob)} base64 chars]")
    return "\n".join(chunks)


def render_prompt(
    client: MCPClient,
    name: str,
    arguments: dict[str, str] | None = None,
    **kwargs: str,
) -> list[dict[str, Any]]:
    """Retrieve and render prompt messages ready for LLM consumption."""
    merged_args = dict(arguments or {})
    merged_args.update(kwargs)
    try:
        result = client.get_prompt(name, merged_args)
    except Exception as exc:
        raise MCPPromptError(f"Failed to get prompt '{name}': {exc}") from exc

    messages: list[dict[str, Any]] = []
    for msg in result.messages:
        content = msg.content
        text = getattr(content, "text", None)
        if text is None:
            # Embedded text resources carry their text one level down; anything else is binary.
            text = getattr(getattr(content, "resource", None), "text", None)
        if text is None:
            text = f"[{content.type} content omitted]"
        messages.append({"role": str(msg.role), "content": text})
    return messages


class MCPContextBridge:
    """Ergonomic bridge to query and expose MCP resources and prompts."""

    def __init__(self, client: MCPClient) -> None:
        self._client = client

    @property
    def client(self) -> MCPClient:
        return self._client

    def list_resources(self) -> list[Resource]:
        """List all resources exposed by the MCP server."""
        try:
            return self._client.list_resources()
        except Exception as exc:
            raise MCPResourceError(f"Failed to list resources: {exc}") from exc

    def read_resource(self, uri: str) -> str:
        """Read a resource by URI and return its text content."""
        return read_resource_content(self._client, uri)

    def list_prompts(self) -> list[Prompt]:
        """List all prompt templates exposed by the MCP server."""
        try:
            return self._client.list_prompts()
        except Exception as exc:
            raise MCPPromptError(f"Failed to list prompts: {exc}") from exc

    def render_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        **kwargs: str,
    ) -> list[dict[str, Any]]:
        """Render prompt template to standard message dictionaries."""
        return render_prompt(self._client, name, arguments=arguments, **kwargs)

    def get_prompt_text(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        **kwargs: str,
    ) -> str:
        """Render prompt template and concatenate message contents into a single string."""
        msgs = self.render_prompt(name, arguments=arguments, **kwargs)
        return "\n\n".join(m["content"] for m in msgs)

    def namespace_functions(self) -> dict[str, Callable[..., Any]]:
        """Sandbox-facing helpers for resources and prompts, keyed by namespace name."""
        return {
            "list_resources": lambda: [
                {"uri": str(r.uri), "name": r.name, "description": r.description}
                for r in self.list_resources()
            ],
            "read_resource": self.read_resource,
            "list_prompts": lambda: [
                {
                    "name": p.name,
                    "description": p.description,
                    "arguments": [a.name for a in p.arguments or []],
                }
                for p in self.list_prompts()
            ],
            "get_prompt": self.get_prompt_text,
        }
