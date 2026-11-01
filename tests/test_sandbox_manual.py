"""Tests for sandbox manual generation from live MCP tools and configuration."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from mcp.types import Tool

from agent_smith.core.models import SandboxConfig
from agent_smith.mcp_client import MCPClient
from agent_smith.sandbox import (
    format_tool_signature,
    format_tools_manual,
    generate_sandbox_manual,
)

SERVER_A = """
from mcp.server.mcpserver import MCPServer
server = MCPServer("server-a")

@server.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two numbers.\"\"\"
    return a + b

@server.tool()
def echo(text: str) -> str:
    \"\"\"Echo text.\"\"\"
    return text

if __name__ == "__main__":
    server.run(transport="stdio")
"""

SERVER_B = """
from mcp.server.mcpserver import MCPServer
server = MCPServer("server-b")

@server.tool()
def multiply(x: int, y: int) -> int:
    \"\"\"Multiply two numbers.\"\"\"
    return x * y

@server.resource("docs://info")
def get_info() -> str:
    return "Server B info."

if __name__ == "__main__":
    server.run(transport="stdio")
"""


@pytest.fixture
def server_a_path(tmp_path: Path) -> Path:
    f = tmp_path / "server_a.py"
    f.write_text(SERVER_A, encoding="utf-8")
    return f


@pytest.fixture
def server_b_path(tmp_path: Path) -> Path:
    f = tmp_path / "server_b.py"
    f.write_text(SERVER_B, encoding="utf-8")
    return f


class TestToolSignatureFormatting:
    def test_format_signature_with_defaults_and_types(self):
        tool = Tool(
            name="query_data",
            description="Query database.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 20},
                    "query": {"type": "string"},
                },
                "required": ["query"],
            },
        )
        sig = format_tool_signature(tool)
        assert "def query_data(query: str, limit: int = 20) -> Any:" in sig
        assert '"""Query database."""' in sig

    def test_format_signature_empty_params(self):
        tool = Tool(name="ping", inputSchema={"type": "object"})
        sig = format_tool_signature(tool)
        assert "def ping() -> Any:" in sig

    def test_format_tools_manual_empty(self):
        assert format_tools_manual([]) == "No external tools available."
        assert format_tools_manual(None) == "No external tools available."


class TestSandboxManualGeneration:
    def test_manual_contains_protocol_and_final_answer(self):
        manual = generate_sandbox_manual()
        assert "# Sandbox Execution Manual" in manual
        assert "final_answer(answer: Any) -> None" in manual
        assert "No external tools available." in manual
        assert "Execution Timeout" in manual
        assert "Memory Limit" in manual

    def test_manual_with_custom_config(self):
        cfg = SandboxConfig(
            max_execution_time_seconds=60,
            max_memory_mb=1024,
            authorized_imports=["json", "math", "re"],
            allowed_directories=["/testbed", "/tmp/agent"],
        )
        manual = generate_sandbox_manual(config=cfg)
        assert "60 seconds per step." in manual
        assert "1024 MB." in manual
        assert "json, math, re" in manual
        assert "/testbed, /tmp/agent" in manual

    def test_dynamic_server_switching(self, server_a_path: Path, server_b_path: Path):
        with MCPClient.from_stdio([sys.executable, str(server_a_path)]) as client_a:
            manual_a = generate_sandbox_manual(client=client_a)
            assert "def add(a: int, b: int) -> Any:" in manual_a
            assert "def echo(text: str) -> Any:" in manual_a
            assert "def multiply" not in manual_a

        with MCPClient.from_stdio([sys.executable, str(server_b_path)]) as client_b:
            manual_b = generate_sandbox_manual(client=client_b)
            assert "def multiply(x: int, y: int) -> Any:" in manual_b
            assert "def add" not in manual_b
            assert "def echo" not in manual_b
            assert "docs://info" in manual_b

    def test_manual_documents_prompts(self, tmp_path: Path):
        server = tmp_path / "prompt_server.py"
        server.write_text(
            "from mcp.server.mcpserver import MCPServer\n"
            "server = MCPServer('p')\n"
            "@server.prompt()\n"
            "def review(language: str) -> str:\n"
            '    """Review code."""\n'
            "    return language\n"
            "if __name__ == '__main__':\n"
            "    server.run(transport='stdio')\n",
            encoding="utf-8",
        )
        with MCPClient.from_stdio([sys.executable, str(server)]) as client:
            manual = generate_sandbox_manual(client=client)
        assert "### Available Prompts" in manual
        assert "`review(language)`: Review code." in manual
        assert "### Available Resources" not in manual


class TestManualMatchesNamespace:
    def test_names_match_sandbox_wrappers(self):
        tools = [
            Tool(name="search.code", inputSchema={"type": "object"}),
            Tool(name="search_code", inputSchema={"type": "object"}),
            Tool(
                name="q",
                inputSchema={"type": "object", "properties": {"from": {"type": "string"}}},
            ),
        ]
        manual = format_tools_manual(tools)
        assert "def search_code() -> Any:" in manual
        assert "def search_code_() -> Any:" in manual
        assert "def q(from_: str = None) -> Any:" in manual
