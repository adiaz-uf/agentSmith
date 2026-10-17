"""Tests for MCP client streamable HTTP transport."""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Generator

import anyio
import httpx2
import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings

from agent_smith.mcp_client import (
    MCPClient,
    MCPConnectionError,
    MCPTimeoutError,
    validate_http_url,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_server_url() -> Generator[str, None, None]:
    """Spin up a live in-process MCP server over streamable HTTP on an ephemeral port."""
    server = MCPServer("http-test-server")

    @server.tool()
    def echo(text: str) -> str:
        """Return text."""
        return text

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.prompt()
    def greet(name: str) -> str:
        """Greet prompt."""
        return f"Hello, {name}!"

    @server.resource("test://info")
    def get_info() -> str:
        """Resource info."""
        return "Resource data over HTTP"

    port = _find_free_port()
    sec = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = server.streamable_http_app(transport_security=sec)

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="error",
        access_log=False,
    )
    srv = uvicorn.Server(config)
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    # Wait for server to bind and start listening
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)

    endpoint_url = f"http://127.0.0.1:{port}/mcp"
    yield endpoint_url

    srv.should_exit = True
    thread.join(timeout=3)


class TestURLValidation:
    def test_valid_http_urls(self):
        assert validate_http_url("http://localhost:8000/mcp") == "http://localhost:8000/mcp"
        assert (
            validate_http_url("  https://api.example.com/v1/mcp  ")
            == "https://api.example.com/v1/mcp"
        )
        assert validate_http_url("http://127.0.0.1:3000") == "http://127.0.0.1:3000"

    def test_invalid_scheme(self):
        with pytest.raises(ValueError, match="Unsupported MCP URL scheme"):
            validate_http_url("ftp://example.com/mcp")

        with pytest.raises(ValueError, match="Unsupported MCP URL scheme"):
            validate_http_url("file:///tmp/mcp")

    def test_empty_or_whitespace_url(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_http_url("")

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_http_url("   \t  ")

    def test_non_string_url(self):
        with pytest.raises(TypeError, match="must be a string"):
            validate_http_url(123)  # type: ignore[arg-type]

    def test_missing_host(self):
        with pytest.raises(ValueError, match="missing host/network location"):
            validate_http_url("http://")


class TestHttpClientConstruction:
    def test_constructor_options(self):
        client = MCPClient.from_http("http://localhost:8000/mcp")
        assert client.transport == "http"
        assert client.url == "http://localhost:8000/mcp"
        assert client.params is None
        assert not client.is_connected
        assert "transport='http'" in repr(client)
        assert "http://localhost:8000/mcp" in repr(client)

    def test_from_url_alias(self):
        client = MCPClient.from_url("https://example.com/mcp")
        assert client.transport == "http"
        assert client.url == "https://example.com/mcp"

    def test_constructor_mutual_exclusion(self):
        with pytest.raises(ValueError, match="Either 'params' .* or 'url'"):
            MCPClient()

        from mcp import StdioServerParameters

        params = StdioServerParameters(command="python", args=[])
        with pytest.raises(ValueError, match="Cannot specify both"):
            MCPClient(params=params, url="http://localhost:8000")


class TestHttpClientOperations:
    def test_unreachable_server_raises(self):
        # Pick an unused port
        dead_port = _find_free_port()
        client = MCPClient.from_http(f"http://127.0.0.1:{dead_port}/mcp")
        with pytest.raises(MCPConnectionError):
            client.connect(timeout=2.0)

    def test_sync_operations(self, http_server_url: str):
        client = MCPClient.from_http(http_server_url)
        with client:
            assert client.is_connected
            assert client.server_name == "http-test-server"
            assert client.transport == "http"
            assert client.url == http_server_url

            # Tools
            tools = client.list_tools()
            tool_names = [t.name for t in tools]
            assert "echo" in tool_names
            assert "add" in tool_names

            # Call echo
            echo_res = client.call_tool("echo", {"text": "streamable test"})
            assert echo_res.content[0].text == "streamable test"

            # Call add
            add_res = client.call_tool("add", {"a": 25, "b": 17})
            assert add_res.content[0].text == "42"

            # Resources
            resources = client.list_resources()
            assert any(r.uri == "test://info" for r in resources)
            res_content = client.read_resource("test://info")
            assert "Resource data over HTTP" in str(res_content)

            # Prompts
            prompts = client.list_prompts()
            assert any(p.name == "greet" for p in prompts)
            prompt_res = client.get_prompt("greet", {"name": "Smith"})
            assert any("Hello, Smith!" in str(m) for m in prompt_res.messages)

            # Ping
            assert client.ping() is True

        assert not client.is_connected

    def test_custom_headers(self, http_server_url: str):
        client = MCPClient.from_http(
            http_server_url,
            headers={"X-Test-Header": "agent-smith"},
        )
        with client:
            assert client.is_connected
            res = client.call_tool("echo", {"text": "with headers"})
            assert res.content[0].text == "with headers"

    @pytest.mark.anyio
    async def test_async_operations(self, http_server_url: str):
        client = MCPClient.from_http(http_server_url)
        async with client:
            assert client.is_connected
            assert client.server_name == "http-test-server"
            tools = await client.list_tools_async()
            tool_names = [t.name for t in tools]
            assert "echo" in tool_names

            res = await client.call_tool_async("add", {"a": 100, "b": 200})
            assert res.content[0].text == "300"

        assert not client.is_connected

    def test_timeout_handling(self):
        async def slow_app(scope, receive, send):
            await anyio.sleep(5)

        transport = httpx2.ASGITransport(app=slow_app)
        slow_http_client = httpx2.AsyncClient(transport=transport)
        client = MCPClient.from_http("http://testserver/mcp", http_client=slow_http_client)
        with pytest.raises(MCPTimeoutError, match="Timed out after 0.1s connecting"):
            client.connect(timeout=0.1)
