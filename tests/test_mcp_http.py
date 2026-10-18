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
    server = MCPServer("http-test-server")

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
        return "Resource data over HTTP"

    port = _find_free_port()
    sec = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = server.streamable_http_app(transport_security=sec)

    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False)
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()

    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)

    yield f"http://127.0.0.1:{port}/mcp"
    srv.should_exit = True
    thread.join(timeout=3)


class TestURLValidation:
    def test_valid_http_urls(self):
        assert validate_http_url("http://localhost:8000/mcp") == "http://localhost:8000/mcp"
        assert validate_http_url("  https://api.example.com/mcp  ") == "https://api.example.com/mcp"
        assert validate_http_url("http://127.0.0.1:3000") == "http://127.0.0.1:3000"

    def test_invalid_urls(self):
        for invalid in ["ftp://example.com/mcp", "file:///tmp/mcp"]:
            with pytest.raises(ValueError, match="Unsupported MCP URL scheme"):
                validate_http_url(invalid)
        for empty in ["", "   \t  "]:
            with pytest.raises(ValueError, match="cannot be empty"):
                validate_http_url(empty)
        with pytest.raises(TypeError, match="must be a string"):
            validate_http_url(123)  # type: ignore[arg-type]
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

    def test_from_url_alias(self):
        client = MCPClient.from_url("https://example.com/mcp")
        assert client.transport == "http"
        assert client.url == "https://example.com/mcp"

    def test_constructor_mutual_exclusion(self):
        from mcp import StdioServerParameters

        with pytest.raises(ValueError, match="Either 'params' .* or 'url'"):
            MCPClient()
        with pytest.raises(ValueError, match="Cannot specify both"):
            MCPClient(
                params=StdioServerParameters(command="python", args=[]),
                url="http://localhost:8000",
            )


class TestHttpClientOperations:
    def test_unreachable_server_raises(self):
        client = MCPClient.from_http(f"http://127.0.0.1:{_find_free_port()}/mcp")
        with pytest.raises(MCPConnectionError):
            client.connect(timeout=2.0)

    def test_sync_operations(self, http_server_url: str):
        with MCPClient.from_http(http_server_url) as client:
            assert client.is_connected
            assert client.server_name == "http-test-server"
            assert client.transport == "http"
            assert {t.name for t in client.list_tools()} >= {"echo", "add"}
            assert client.call_tool("echo", {"text": "msg"}).content[0].text == "msg"
            assert client.call_tool("add", {"a": 25, "b": 17}).content[0].text == "42"
            assert any(r.uri == "test://info" for r in client.list_resources())
            assert "Resource data" in str(client.read_resource("test://info"))
            assert any(p.name == "greet" for p in client.list_prompts())
            prompt_res = client.get_prompt("greet", {"name": "Smith"})
            assert any("Hello, Smith!" in str(m) for m in prompt_res.messages)
            assert client.ping() is True

    def test_custom_headers(self, http_server_url: str):
        with MCPClient.from_http(http_server_url, headers={"X-Test": "1"}) as client:
            assert client.call_tool("echo", {"text": "hi"}).content[0].text == "hi"

    @pytest.mark.anyio
    async def test_async_operations(self, http_server_url: str):
        async with MCPClient.from_http(http_server_url) as client:
            assert client.is_connected
            assert any(t.name == "echo" for t in await client.list_tools_async())
            res = await client.call_tool_async("add", {"a": 100, "b": 200})
            assert res.content[0].text == "300"

    def test_timeout_handling(self):
        async def slow_app(scope, receive, send):
            await anyio.sleep(5)

        slow_client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=slow_app))
        client = MCPClient.from_http("http://testserver/mcp", http_client=slow_client)
        with pytest.raises(MCPTimeoutError, match="Timed out after 0.1s connecting"):
            client.connect(timeout=0.1)

    def test_owned_http_client_closed_on_sync_close(self, http_server_url: str):
        client = MCPClient.from_http(http_server_url, headers={"X-Test": "1"})
        assert client._owns_http_client is True
        internal_client = client._http_client
        assert internal_client is not None
        assert not internal_client.is_closed

        with client:
            assert client.is_connected
            assert client.call_tool("echo", {"text": "check"}).content[0].text == "check"

        assert internal_client.is_closed

    def test_unconnected_owned_http_client_closed(self):
        client = MCPClient.from_http("http://127.0.0.1:8000/mcp", headers={"X-Test": "1"})
        assert client._owns_http_client is True
        internal_client = client._http_client
        assert not internal_client.is_closed
        client.close()
        assert internal_client.is_closed

    def test_unowned_http_client_not_closed_on_close(self):
        external_client = httpx2.AsyncClient()
        client = MCPClient.from_http("http://127.0.0.1:8000/mcp", http_client=external_client)
        assert client._owns_http_client is False
        client.close()
        assert not external_client.is_closed
        anyio.run(external_client.aclose)

    @pytest.mark.anyio
    async def test_owned_http_client_closed_on_async_close(self, http_server_url: str):
        client = MCPClient.from_http(http_server_url, headers={"X-Test": "1"})
        assert client._owns_http_client is True
        internal_client = client._http_client
        assert not internal_client.is_closed

        async with client:
            assert client.is_connected
            res = await client.call_tool_async("echo", {"text": "async-check"})
            assert res.content[0].text == "async-check"

        assert internal_client.is_closed
