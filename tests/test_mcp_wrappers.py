"""Tests for dynamic MCP tool discovery, schema parsing, and wrapper generation."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from agent_smith.mcp_client import (
    MCPClient,
    ToolRegistry,
    create_tool_signature,
    create_tool_wrapper,
    discover_tools,
    sanitize_identifier,
)
from agent_smith.mcp_client.exceptions import MCPToolCallError

SERVER_A_CODE = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("server-a")

@server.tool()
def add(a: int, b: int) -> int:
    return a + b

@server.tool()
def echo(text: str, prefix: str = "echo: ") -> str:
    return f"{prefix}{text}"

if __name__ == "__main__":
    server.run(transport="stdio")
"""

SERVER_B_CODE = """
from mcp.server.mcpserver import MCPServer

server = MCPServer("server-b")

@server.tool()
def multiply(x: int, y: int) -> int:
    return x * y

@server.tool()
def reverse_str(text: str) -> str:
    return text[::-1]

if __name__ == "__main__":
    server.run(transport="stdio")
"""


@pytest.fixture
def server_a_script(tmp_path: Path) -> Path:
    script = tmp_path / "server_a.py"
    script.write_text(SERVER_A_CODE, encoding="utf-8")
    return script


@pytest.fixture
def server_b_script(tmp_path: Path) -> Path:
    script = tmp_path / "server_b.py"
    script.write_text(SERVER_B_CODE, encoding="utf-8")
    return script


class TestIdentifierSanitization:
    def test_sanitize_identifier(self):
        assert sanitize_identifier("my_tool") == "my_tool"
        assert sanitize_identifier("search-code") == "search_code"
        assert sanitize_identifier("123_calculate") == "tool_123_calculate"
        assert sanitize_identifier("run tests") == "run_tests"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("search.code", "search_code"),
            ("fs/read", "fs_read"),
            ("ns:tool@v1", "ns_tool_v1"),
            ("class", "class_"),
            ("match", "match_"),
            ("", "tool_"),
        ],
    )
    def test_sanitize_invalid_and_keyword_names(self, raw: str, expected: str):
        result = sanitize_identifier(raw)
        assert result == expected
        assert result.isidentifier()


class TestSignatureGeneration:
    def test_signature_ordering_required_before_optional(self):
        tool = Tool(
            name="query_db",
            description="Query database",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 50},
                    "sql": {"type": "string"},
                },
                "required": ["sql"],
            },
        )
        params = list(create_tool_signature(tool).parameters.values())
        assert len(params) == 2
        assert (params[0].name, params[0].default, params[0].annotation) == (
            "sql",
            inspect.Parameter.empty,
            str,
        )
        assert (params[1].name, params[1].default, params[1].annotation) == ("limit", 50, int)

    def test_signature_empty_schema(self):
        tool = Tool(name="ping", inputSchema={"type": "object"})
        assert len(create_tool_signature(tool).parameters) == 0

    def test_type_mappings(self):
        tool = Tool(
            name="types_tool",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "count": {"type": "integer"},
                    "score": {"type": "number"},
                    "active": {"type": "boolean"},
                    "tags": {"type": "array"},
                    "meta": {"type": "object"},
                },
            },
        )
        sig = create_tool_signature(tool)
        assert sig.parameters["text"].annotation is str
        assert sig.parameters["count"].annotation is int
        assert sig.parameters["score"].annotation is float
        assert sig.parameters["active"].annotation is bool
        assert sig.parameters["tags"].annotation is list
        assert sig.parameters["meta"].annotation is dict


class TestToolWrapperExecution:
    def test_call_wrapper_positional_and_keyword(self, server_a_script: Path):
        with MCPClient.from_stdio([sys.executable, str(server_a_script)]) as client:
            add_tool = next(t for t in client.list_tools() if t.name == "add")
            wrapper = create_tool_wrapper(client, add_tool)
            assert wrapper.__name__ == "add"
            assert wrapper(12, 30).content[0].text == "42"
            assert wrapper(a=100, b=50).content[0].text == "150"

    def test_call_wrapper_default_arguments(self, server_a_script: Path):
        with MCPClient.from_stdio([sys.executable, str(server_a_script)]) as client:
            echo_tool = next(t for t in client.list_tools() if t.name == "echo")
            wrapper = create_tool_wrapper(client, echo_tool)
            assert wrapper(text="hello").content[0].text == "echo: hello"
            assert wrapper(text="hello", prefix="msg: ").content[0].text == "msg: hello"

    def test_invalid_arguments_raise_type_error(self, server_a_script: Path):
        with MCPClient.from_stdio([sys.executable, str(server_a_script)]) as client:
            add_tool = next(t for t in client.list_tools() if t.name == "add")
            wrapper = create_tool_wrapper(client, add_tool)
            with pytest.raises(TypeError, match="Invalid call to tool 'add'"):
                wrapper(a=10)
            with pytest.raises(TypeError, match="Invalid call to tool 'add'"):
                wrapper(a=1, b=2, unexpected=3)

    def test_unwrap_text_mode(self, server_a_script: Path):
        with MCPClient.from_stdio([sys.executable, str(server_a_script)]) as client:
            add_tool = next(t for t in client.list_tools() if t.name == "add")
            assert isinstance(
                create_tool_wrapper(client, add_tool, unwrap_text=False)(5, 5), CallToolResult
            )
            assert create_tool_wrapper(client, add_tool, unwrap_text=True)(5, 5) == "10"


class TestToolRegistryAndSandboxNamespace:
    def test_discover_tools_mapping(self, server_a_script: Path):
        with MCPClient.from_stdio([sys.executable, str(server_a_script)]) as client:
            wrappers = discover_tools(client)
            assert callable(wrappers["add"]) and callable(wrappers["echo"])

    def test_registry_sync_and_exec_in_sandbox(self, server_a_script: Path):
        with MCPClient.from_stdio([sys.executable, str(server_a_script)]) as client:
            captured: list[Any] = []
            ns = {
                "final_answer": lambda val: captured.append(val),
                "__builtins__": __builtins__,
            }
            ToolRegistry(namespace=ns, unwrap_text=True).sync_tools(client)

            code = "sum_res = add(20, 22)\nmsg = echo(text=f'Total {sum_res}')\nfinal_answer(msg)"
            exec(code, ns)  # noqa: S102
            assert ns["sum_res"] == "42"
            assert captured == ["echo: Total 42"]

    def test_registry_switches_servers_dynamically_preserving_final_answer(
        self,
        server_a_script: Path,
        server_b_script: Path,
    ):
        client_a = MCPClient.from_stdio([sys.executable, str(server_a_script)])
        client_b = MCPClient.from_stdio([sys.executable, str(server_b_script)])
        final_calls: list[Any] = []
        ns = {"final_answer": lambda ans: final_calls.append(ans), "custom": 999}
        registry = ToolRegistry(namespace=ns)

        with client_a:
            registry.sync_tools(client_a)
            assert "add" in ns and "echo" in ns and "multiply" not in ns
            assert ns["custom"] == 999 and "final_answer" in ns

        with client_b:
            registry.sync_tools(client_b)
            assert "add" not in ns and "echo" not in ns
            assert "multiply" in ns and "reverse_str" in ns
            assert ns["custom"] == 999 and "final_answer" in ns

            exec("res = reverse_str('AgentSmith')", ns)  # noqa: S102
            exec("final_answer(res.content[0].text)", ns)  # noqa: S102
            assert final_calls == ["htimStnegA"]

        registry.clear_tools()
        assert "multiply" not in ns and "reverse_str" not in ns
        assert "final_answer" in ns and ns["custom"] == 999


class FakeClient:
    """Minimal stand-in for MCPClient that records calls and returns canned results."""

    def __init__(self, tools: list[Tool], result: CallToolResult | None = None) -> None:
        self.tools = tools
        self.result = result or CallToolResult(content=[TextContent(type="text", text="ok")])
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def list_tools(self) -> list[Tool]:
        return self.tools

    def call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        return self.result


def _tool(name: str, properties: dict[str, Any] | None = None, required: list[str] | None = None):
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return Tool(name=name, inputSchema=schema)


class TestEdgeCases:
    def test_optional_args_without_default_are_omitted(self):
        client = FakeClient([])
        tool = _tool("q", {"sql": {"type": "string"}, "limit": {"type": "integer"}}, ["sql"])
        create_tool_wrapper(client, tool)("SELECT 1")  # type: ignore[arg-type]
        assert client.calls == [("q", {"sql": "SELECT 1"})]

    def test_param_names_are_mapped_back_to_server_names(self):
        client = FakeClient([])
        tool = _tool("q", {"from": {"type": "string"}, "max-rows": {"type": "integer"}})
        wrapper = create_tool_wrapper(client, tool)  # type: ignore[arg-type]
        assert list(inspect.signature(wrapper).parameters) == ["from_", "max_rows"]
        wrapper(from_="t", max_rows=3)
        assert client.calls == [("q", {"from": "t", "max-rows": 3})]

    def test_colliding_param_names_are_disambiguated(self):
        client = FakeClient([])
        tool = _tool("q", {"a-b": {"type": "string"}, "a_b": {"type": "string"}})
        wrapper = create_tool_wrapper(client, tool)  # type: ignore[arg-type]
        wrapper(a_b="x", a_b_="y")
        assert client.calls == [("q", {"a-b": "x", "a_b": "y"})]

    def test_colliding_tool_names_are_disambiguated(self):
        client = FakeClient([_tool("search.code"), _tool("search_code")])
        wrappers = discover_tools(client)  # type: ignore[arg-type]
        assert set(wrappers) == {"search_code", "search_code_"}
        wrappers["search_code_"]()
        assert client.calls == [("search_code", {})]

    def test_error_result_raises_in_unwrap_mode(self):
        error = CallToolResult(content=[TextContent(type="text", text="boom")], is_error=True)
        client = FakeClient([], result=error)
        wrapper = create_tool_wrapper(client, _tool("t"), unwrap_text=True)  # type: ignore[arg-type]
        with pytest.raises(MCPToolCallError, match="boom"):
            wrapper()
        assert create_tool_wrapper(client, _tool("t"))().is_error  # type: ignore[arg-type]

    def test_registry_does_not_overwrite_final_answer_or_builtins(self):
        def final_answer(x: Any) -> Any:
            return x

        ns: dict[str, Any] = {"final_answer": final_answer}
        client = FakeClient([_tool("final_answer"), _tool("print"), _tool("add")])
        registry = ToolRegistry(namespace=ns, client=client)  # type: ignore[arg-type]
        assert ns["final_answer"] is final_answer
        assert "print" not in ns
        assert registry.registered_tools == {"add"}

        registry.clear_tools()
        assert ns == {"final_answer": final_answer}

    def test_failed_discovery_keeps_existing_tools(self):
        ns: dict[str, Any] = {}
        registry = ToolRegistry(namespace=ns, client=FakeClient([_tool("add")]))  # type: ignore[arg-type]

        class BrokenClient(FakeClient):
            def list_tools(self) -> list[Tool]:
                raise RuntimeError("server down")

        with pytest.raises(RuntimeError):
            registry.sync_tools(BrokenClient([]))  # type: ignore[arg-type]
        assert "add" in ns and registry.registered_tools == {"add"}
