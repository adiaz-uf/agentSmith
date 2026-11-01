"""Tests for MCP resources and prompts exposure."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from mcp.types import GetPromptResult, ImageContent, PromptMessage, TextContent

from agent_smith.mcp_client import (
    MCPClient,
    MCPContextBridge,
    MCPPromptError,
    MCPResourceError,
    ToolRegistry,
    read_resource_content,
    render_prompt,
)

SERVER_CODE = """
import base64
from mcp.server.mcpserver import MCPServer

server = MCPServer("context-server")

@server.resource("docs://architecture")
def get_architecture() -> str:
    return "Microkernel and sandbox architecture documentation."

@server.resource("binary://logo")
def get_logo() -> bytes:
    return b"SmithLogoPNG"

@server.prompt()
def code_review(language: str, code: str) -> str:
    return f"Reviewing {language} code:\\n{code}"

@server.prompt()
def system_instructions() -> str:
    return "You are an autonomous AI coding agent."

@server.tool()
def echo(text: str) -> str:
    return text

@server.tool()
def read_resource(uri: str) -> str:
    return "shadowing tool"

if __name__ == "__main__":
    server.run(transport="stdio")
"""


@pytest.fixture
def context_server_path(tmp_path: Path) -> Path:
    script = tmp_path / "context_server.py"
    script.write_text(SERVER_CODE, encoding="utf-8")
    return script


class TestMCPResources:
    def test_list_and_read_text_resource(self, context_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client:
            bridge = MCPContextBridge(client)
            resources = bridge.list_resources()
            assert any(str(r.uri) == "docs://architecture" for r in resources)

            text = bridge.read_resource("docs://architecture")
            assert "Microkernel and sandbox" in text

    def test_read_binary_resource(self, context_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client:
            content = read_resource_content(client, "binary://logo")
            assert "SmithLogoPNG" in content

    def test_read_nonexistent_resource_raises(self, context_server_path: Path):
        with (
            MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client,
            pytest.raises(MCPResourceError),
        ):
            MCPContextBridge(client).read_resource("docs://nonexistent")


class TestMCPPrompts:
    def test_list_prompts(self, context_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client:
            bridge = MCPContextBridge(client)
            prompts = {p.name: p for p in bridge.list_prompts()}
            assert "code_review" in prompts and "system_instructions" in prompts

    def test_render_prompt_messages(self, context_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client:
            bridge = MCPContextBridge(client)
            msgs = bridge.render_prompt(
                "code_review",
                {"language": "python", "code": "def foo(): pass"},
            )
            assert len(msgs) == 1
            assert "Reviewing python code" in msgs[0]["content"]

    def test_get_prompt_text(self, context_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client:
            bridge = MCPContextBridge(client)
            text = bridge.get_prompt_text("system_instructions")
            assert "autonomous AI coding agent" in text

    def test_nonexistent_prompt_raises(self, context_server_path: Path):
        with (
            MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client,
            pytest.raises(MCPPromptError),
        ):
            render_prompt(client, "missing_prompt")


class TestNamespaceExposure:
    def test_namespace_functions_in_sandbox(self, context_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client:
            ns: dict[str, Any] = dict(MCPContextBridge(client).namespace_functions())
            code = (
                "doc = read_resource('docs://architecture')\n"
                "res_list = list_resources()\n"
                "prompts = list_prompts()\n"
                "review = get_prompt('code_review', language='go', code='x := 1')\n"
            )
            exec(code, ns)  # noqa: S102
            assert "Microkernel" in ns["doc"]
            assert any(r["uri"] == "docs://architecture" for r in ns["res_list"])
            by_name = {p["name"]: p for p in ns["prompts"]}
            assert by_name["code_review"]["arguments"] == ["language", "code"]
            assert ns["review"].startswith("Reviewing go code")

    def test_tool_registry_with_expose_context(self, context_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(context_server_path)]) as client:
            ns: dict[str, Any] = {"final_answer": lambda x: x, "get_prompt": "user-defined"}
            registry = ToolRegistry(namespace=ns, expose_context=True)
            registry.sync_tools(client)

            assert {"echo", "read_resource", "list_resources", "list_prompts"} <= ns.keys()
            # Pre-existing names are never overwritten.
            assert ns["get_prompt"] == "user-defined" and "get_prompt" not in registry

            # The helper wins over the server's same-named tool.
            exec("doc = read_resource('docs://architecture')", ns)  # noqa: S102
            assert "Microkernel" in ns["doc"]

            registry.clear_tools()
            assert "echo" not in ns and "read_resource" not in ns
            assert "final_answer" in ns and ns["get_prompt"] == "user-defined"


def test_render_prompt_non_text_content():
    class FakeClient:
        def get_prompt(self, name: str, arguments: dict[str, str]) -> GetPromptResult:
            return GetPromptResult(
                messages=[
                    PromptMessage(role="user", content=TextContent(type="text", text="hi")),
                    PromptMessage(
                        role="user",
                        content=ImageContent(type="image", data="AA==", mimeType="image/png"),
                    ),
                ]
            )

    msgs = render_prompt(FakeClient(), "p")  # type: ignore[arg-type]
    assert [m["content"] for m in msgs] == ["hi", "[image content omitted]"]
