"""Tests for MCP client stdio transport."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_smith.mcp_client import (
    MCPClient,
    MCPConnectionError,
    MCPServerCrashedError,
    MCPTimeoutError,
    build_stdio_params,
    parse_stdio_command,
)

SERVER_CODE = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("dummy-server")

@server.tool()
def echo(text: str) -> str:
    return text

@server.tool()
def add(a: int, b: int) -> int:
    return a + b

@server.prompt()
def greet(name: str) -> str:
    return f"Hello, {name}!"

@server.resource("test://info")
def get_info() -> str:
    return "Sample resource data"

if __name__ == "__main__":
    server.run(transport="stdio")
"""


@pytest.fixture
def dummy_server_path(tmp_path: Path) -> Path:
    script = tmp_path / "dummy_server.py"
    script.write_text(SERVER_CODE, encoding="utf-8")
    return script


class TestCommandParsing:
    def test_parse_string_command(self):
        assert parse_stdio_command("python mcp_tools_mbpp.py --port 5000") == (
            "python",
            ["mcp_tools_mbpp.py", "--port", "5000"],
        )

    def test_parse_string_with_quotes(self):
        assert parse_stdio_command('node server.js --config "my folder/config.json"') == (
            "node",
            ["server.js", "--config", "my folder/config.json"],
        )

    def test_parse_sequence_command(self):
        assert parse_stdio_command(["python", "-u", "tools.py"]) == ("python", ["-u", "tools.py"])

    def test_parse_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_stdio_command("   ")
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_stdio_command([])

    def test_build_stdio_params(self, monkeypatch):
        monkeypatch.setenv("EXISTING_VAR", "original")
        params = build_stdio_params("python server.py", env={"PORT": 8080}, cwd=".")
        assert params.command == "python"
        assert params.args == ["server.py"]
        assert params.env.get("EXISTING_VAR") == "original"
        assert params.env.get("PORT") == "8080"
        assert params.cwd == Path.cwd()

    def test_build_stdio_params_invalid_cwd(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            build_stdio_params("python server.py", cwd="/nonexistent/directory/xyz")


class TestStdioClientLifecycle:
    def test_connect_and_initialize(self, dummy_server_path: Path):
        client = MCPClient.from_stdio([sys.executable, str(dummy_server_path)])
        assert not client.is_connected
        with client:
            assert client.is_connected
            assert client.server_name == "dummy-server"
            assert client.ping() is True
        assert not client.is_connected

    def test_operations_before_connect_raise(self, dummy_server_path: Path):
        client = MCPClient.from_stdio([sys.executable, str(dummy_server_path)])
        with pytest.raises(MCPConnectionError, match="not connected"):
            client.list_tools()
        with pytest.raises(MCPConnectionError, match="not connected"):
            client.call_tool("echo")

    def test_server_not_found(self):
        with pytest.raises(MCPConnectionError, match="not found"):
            MCPClient.from_stdio("non_existent_executable_123456789").connect()

    def test_server_crash_on_startup(self, tmp_path: Path):
        crash_script = tmp_path / "crasher.py"
        crash_script.write_text("import sys; sys.exit(42)\n", encoding="utf-8")
        with pytest.raises((MCPServerCrashedError, MCPConnectionError)):
            MCPClient.from_stdio([sys.executable, str(crash_script)]).connect()

    def test_translate_timeout_error(self):
        translated = MCPClient.from_stdio("python server.py")._translate_error(
            TimeoutError("timed out")
        )
        assert isinstance(translated, MCPTimeoutError)


class TestStdioClientProtocol:
    def test_list_and_call_tools(self, dummy_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(dummy_server_path)]) as client:
            tools = {t.name: t for t in client.list_tools()}
            assert "echo" in tools and "add" in tools
            assert "a" in tools["add"].input_schema["properties"]
            assert client.call_tool("add", {"a": 10, "b": 32}).content[0].text == "42"
            assert client.call_tool("echo", {"text": "hello"}).content[0].text == "hello"

    def test_resources_and_prompts(self, dummy_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(dummy_server_path)]) as client:
            assert any(str(r.uri) == "test://info" for r in client.list_resources())
            assert client.read_resource("test://info").contents[0].text == "Sample resource data"
            assert any(p.name == "greet" for p in client.list_prompts())
            assert "Neo" in client.get_prompt("greet", {"name": "Neo"}).messages[0].content.text

    @pytest.mark.anyio
    async def test_async_interface(self, dummy_server_path: Path):
        async with MCPClient.from_stdio([sys.executable, str(dummy_server_path)]) as client:
            assert client.is_connected
            assert any(t.name == "add" for t in await client.list_tools_async())
            res = await client.call_tool_async("add", {"a": 100, "b": 200})
            assert res.content[0].text == "300"
