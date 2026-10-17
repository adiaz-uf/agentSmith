"""Tests for MCP client stdio transport (no external network dependencies)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_smith.mcp_client import (
    MCPClient,
    MCPConnectionError,
    MCPServerCrashedError,
    build_stdio_params,
    parse_stdio_command,
)

SERVER_CODE = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("dummy-server")

@server.tool()
def echo(text: str) -> str:
    \"\"\"Return the provided text unaltered.\"\"\"
    return text

@server.tool()
def add(a: int, b: int) -> int:
    \"\"\"Add two integers together.\"\"\"
    return a + b

@server.prompt()
def greet(name: str) -> str:
    \"\"\"Greeting prompt.\"\"\"
    return f"Hello, {name}!"

@server.resource("test://info")
def get_info() -> str:
    \"\"\"Static test resource.\"\"\"
    return "Sample resource data"

if __name__ == "__main__":
    server.run(transport="stdio")
"""


@pytest.fixture
def dummy_server_path(tmp_path: Path) -> Path:
    """Write a standalone stdio MCP server script and return its path."""
    script_file = tmp_path / "dummy_server.py"
    script_file.write_text(SERVER_CODE, encoding="utf-8")
    return script_file


class TestCommandParsing:
    def test_parse_string_command(self):
        cmd, args = parse_stdio_command("python mcp_tools_mbpp.py --port 5000")
        assert cmd == "python"
        assert args == ["mcp_tools_mbpp.py", "--port", "5000"]

    def test_parse_string_with_quotes(self):
        cmd, args = parse_stdio_command('node server.js --config "my folder/config.json"')
        assert cmd == "node"
        assert args == ["server.js", "--config", "my folder/config.json"]

    def test_parse_sequence_command(self):
        cmd, args = parse_stdio_command(["python", "-u", "tools.py"])
        assert cmd == "python"
        assert args == ["-u", "tools.py"]

    def test_parse_empty_string_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_stdio_command("   ")

    def test_parse_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            parse_stdio_command([])

    def test_build_stdio_params(self, monkeypatch):
        monkeypatch.setenv("EXISTING_VAR", "original")
        params = build_stdio_params(
            "python server.py",
            env={"EXTRA_VAR": "custom", "PORT": 8080},
            cwd=".",
        )
        assert params.command == "python"
        assert params.args == ["server.py"]
        assert params.env.get("EXISTING_VAR") == "original"
        assert params.env.get("EXTRA_VAR") == "custom"
        assert params.env.get("PORT") == "8080"
        assert params.cwd == Path.cwd()

    def test_build_stdio_params_invalid_cwd(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            build_stdio_params("python server.py", cwd="/nonexistent/directory/xyz")


class TestStdioClientLifecycle:
    def test_connect_and_initialize(self, dummy_server_path: Path):
        client = MCPClient.from_stdio([sys.executable, str(dummy_server_path)])
        assert not client.is_connected

        client.connect()
        try:
            assert client.is_connected
            assert client.server_name == "dummy-server"
            assert client.ping() is True
        finally:
            client.close()

        assert not client.is_connected

    def test_context_manager(self, dummy_server_path: Path):
        cmd = f'"{sys.executable}" "{dummy_server_path}"'
        with MCPClient.from_stdio(cmd) as client:
            assert client.is_connected
            assert client.server_name == "dummy-server"

        assert not client.is_connected

    def test_operations_before_connect_raise(self, dummy_server_path: Path):
        client = MCPClient.from_stdio([sys.executable, str(dummy_server_path)])
        with pytest.raises(MCPConnectionError, match="not connected"):
            client.list_tools()
        with pytest.raises(MCPConnectionError, match="not connected"):
            client.call_tool("echo")

    def test_server_not_found(self):
        client = MCPClient.from_stdio("non_existent_executable_123456789")
        with pytest.raises(MCPConnectionError, match="not found"):
            client.connect()

    def test_server_crash_on_startup(self, tmp_path: Path):
        crash_script = tmp_path / "crasher.py"
        crash_script.write_text("import sys; sys.exit(42)\n", encoding="utf-8")

        client = MCPClient.from_stdio([sys.executable, str(crash_script)])
        with pytest.raises((MCPServerCrashedError, MCPConnectionError)):
            client.connect()


class TestStdioClientProtocol:
    def test_list_and_call_tools(self, dummy_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(dummy_server_path)]) as client:
            tools = client.list_tools()
            tool_names = {t.name for t in tools}
            assert "echo" in tool_names
            assert "add" in tool_names

            # Check input_schema
            add_tool = next(t for t in tools if t.name == "add")
            assert "properties" in add_tool.input_schema
            assert "a" in add_tool.input_schema["properties"]
            assert "b" in add_tool.input_schema["properties"]

            # Call add tool
            res_add = client.call_tool("add", {"a": 10, "b": 32})
            assert res_add.is_error is False
            assert res_add.content[0].text == "42"

            # Call echo tool
            res_echo = client.call_tool("echo", {"text": "Agent Smith MCP Test"})
            assert res_echo.is_error is False
            assert res_echo.content[0].text == "Agent Smith MCP Test"

    def test_resources_and_prompts(self, dummy_server_path: Path):
        with MCPClient.from_stdio([sys.executable, str(dummy_server_path)]) as client:
            resources = client.list_resources()
            assert any(str(r.uri) == "test://info" for r in resources)

            read_res = client.read_resource("test://info")
            assert read_res.contents[0].text == "Sample resource data"

            prompts = client.list_prompts()
            assert any(p.name == "greet" for p in prompts)

            prompt_res = client.get_prompt("greet", {"name": "Neo"})
            assert "Neo" in prompt_res.messages[0].content.text

    @pytest.mark.anyio
    async def test_async_interface(self, dummy_server_path: Path):
        async with MCPClient.from_stdio([sys.executable, str(dummy_server_path)]) as client:
            assert client.is_connected
            tools = await client.list_tools_async()
            assert any(t.name == "add" for t in tools)

            res = await client.call_tool_async("add", {"a": 100, "b": 200})
            assert res.content[0].text == "300"
