"""Model Context Protocol (MCP) client implementation."""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import anyio
import httpx2
from anyio.from_thread import BlockingPortal, start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client
from mcp.shared.exceptions import MCPError as SDKMCPError
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    PaginatedRequestParams,
    Prompt,
    ReadResourceResult,
    Resource,
    Tool,
)

from agent_smith.mcp_client.exceptions import (
    MCPConnectionError,
    MCPError,
    MCPServerCrashedError,
    MCPTimeoutError,
    MCPToolCallError,
)
from agent_smith.mcp_client.transports import build_stdio_params, validate_http_url

__all__ = ["MCPClient"]

logger = logging.getLogger(__name__)


def _unwrap_exception(exc: BaseException) -> BaseException:
    """Recursively unwrap ExceptionGroup instances to find the root cause exception."""
    current = exc
    while hasattr(current, "exceptions") and current.exceptions:
        current = current.exceptions[0]
    return current


class MCPClient:
    """Client for Model Context Protocol (MCP) servers."""

    def __init__(
        self,
        params: StdioServerParameters | None = None,
        *,
        url: str | None = None,
        http_client: Any | None = None,
        owns_http_client: bool = False,
    ) -> None:
        if params is None and url is None:
            raise ValueError("Either 'params' (for stdio) or 'url' (for HTTP) must be provided.")
        if params is not None and url is not None:
            raise ValueError("Cannot specify both 'params' and 'url'.")

        self._params = params
        self._url = validate_http_url(url) if url is not None else None
        self._http_client = http_client
        self._owns_http_client = owns_http_client

        self._portal_cm: Any | None = None
        self._portal: BlockingPortal | None = None
        self._session: ClientSession | None = None
        self._init_result: Any | None = None
        self._stop_evt: anyio.Event | None = None
        self._async_transport_cm: Any | None = None
        self._async_session_cm: Any | None = None

    @classmethod
    def from_stdio(
        cls,
        command: str | Sequence[str],
        *,
        env: Mapping[str, str | int | float | bool] | None = None,
        cwd: str | Path | None = None,
    ) -> MCPClient:
        """Create an MCPClient configured for standard I/O."""
        params = build_stdio_params(command, env=env, cwd=cwd)
        return cls(params=params)

    @classmethod
    def from_http(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        http_client: Any | None = None,
    ) -> MCPClient:
        """Create an MCPClient configured for streamable HTTP transport."""
        normalized_url = validate_http_url(url)
        client = http_client
        owns_http_client = False
        if client is None and headers is not None:
            client = create_mcp_http_client(headers=dict(headers))
            owns_http_client = True
        return cls(url=normalized_url, http_client=client, owns_http_client=owns_http_client)

    @classmethod
    def from_url(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        http_client: Any | None = None,
    ) -> MCPClient:
        """Alias for from_http."""
        return cls.from_http(url=url, headers=headers, http_client=http_client)

    @property
    def transport(self) -> str:
        return "stdio" if self._params is not None else "http"

    @property
    def url(self) -> str | None:
        return self._url

    @property
    def params(self) -> StdioServerParameters | None:
        return self._params

    @property
    def is_connected(self) -> bool:
        return self._session is not None

    @property
    def server_name(self) -> str | None:
        if self._init_result and hasattr(self._init_result, "server_info"):
            return self._init_result.server_info.name
        return None

    @property
    def server_version(self) -> str | None:
        if self._init_result and hasattr(self._init_result, "server_info"):
            return self._init_result.server_info.version
        return None

    @property
    def instructions(self) -> str | None:
        if self._init_result and hasattr(self._init_result, "instructions"):
            return self._init_result.instructions
        return None

    @property
    def _target_description(self) -> str:
        if self._url is not None:
            return self._url
        if self._params is not None:
            return self._params.command
        return "unknown"

    def connect(self, timeout: float = 30.0) -> MCPClient:
        """Connect to the server and initialize the session."""
        if self.is_connected:
            return self

        portal_cm = start_blocking_portal()
        portal = portal_cm.__enter__()
        ready_evt = anyio.Event()
        stop_evt = anyio.Event()
        holder: dict[str, Any] = {}

        async def _worker() -> None:
            try:
                if self._params is not None:
                    transport_cm = stdio_client(self._params)
                else:
                    transport_cm = streamable_http_client(self._url, http_client=self._http_client)

                async with (
                    transport_cm as (read_stream, write_stream),
                    ClientSession(read_stream, write_stream) as session,
                ):
                    init_result = await session.initialize()
                    holder["session"] = session
                    holder["init_result"] = init_result
                    ready_evt.set()
                    await stop_evt.wait()
            except Exception as exc:  # noqa: BLE001
                holder["error"] = exc
                ready_evt.set()
            finally:
                if (
                    self._owns_http_client
                    and self._http_client is not None
                    and not getattr(self._http_client, "is_closed", False)
                ):
                    try:
                        await self._http_client.aclose()
                    except Exception:  # noqa: BLE001, S110
                        pass

        portal.start_task_soon(_worker)

        async def _wait_ready(evt: anyio.Event, t: float) -> None:
            with anyio.fail_after(t):
                await evt.wait()

        try:
            portal.call(_wait_ready, ready_evt, timeout)
        except TimeoutError as exc:
            try:
                portal.call(stop_evt.set)
            except Exception:  # noqa: BLE001, S110
                pass
            portal_cm.__exit__(None, None, None)
            raise MCPTimeoutError(
                f"Timed out after {timeout:.1f}s connecting to MCP server: {self._target_description}"
            ) from exc

        if "error" in holder:
            err = holder["error"]
            try:
                portal.call(stop_evt.set)
            except Exception:  # noqa: BLE001, S110
                pass
            portal_cm.__exit__(None, None, None)
            raise self._translate_error(err) from err

        self._portal_cm = portal_cm
        self._portal = portal
        self._session = holder["session"]
        self._init_result = holder["init_result"]
        self._stop_evt = stop_evt
        logger.info(
            "Connected to MCP server '%s' (version %s) via %s: %s",
            self.server_name,
            self.server_version,
            self.transport,
            self._target_description,
        )
        return self

    def close(self) -> None:
        """Close connection and shutdown server process."""
        if self._portal is not None:
            try:
                if self._stop_evt is not None:
                    self._portal.call(self._stop_evt.set)
            except Exception:  # noqa: BLE001, S110
                pass
            finally:
                if self._portal_cm is not None:
                    self._portal_cm.__exit__(None, None, None)
                    self._portal_cm = None
                self._portal = None
                self._session = None
                self._init_result = None
                self._stop_evt = None
                logger.info("MCP client connection closed.")

        if (
            self._owns_http_client
            and self._http_client is not None
            and not getattr(self._http_client, "is_closed", False)
        ):
            try:
                anyio.run(self._http_client.aclose)
            except Exception:  # noqa: BLE001, S110
                pass

    def __enter__(self) -> MCPClient:  # noqa: PYI034 (py3.10 has no typing.Self)
        return self.connect()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def list_tools(self) -> list[Tool]:
        """Discover tools exposed by the MCP server."""
        self._ensure_connected()
        return self._list_all(self._session.list_tools, "tools")

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
        """Call a tool on the server and return its result."""
        self._ensure_connected()
        return self._call(self._session.call_tool, name, arguments or {})

    def list_resources(self) -> list[Resource]:
        """List resources provided by the MCP server."""
        self._ensure_connected()
        return self._list_all(self._session.list_resources, "resources")

    def read_resource(self, uri: str) -> ReadResourceResult:
        """Read a resource by its URI from the MCP server."""
        self._ensure_connected()
        return self._call(self._session.read_resource, uri)

    def list_prompts(self) -> list[Prompt]:
        """List prompts provided by the MCP server."""
        self._ensure_connected()
        return self._list_all(self._session.list_prompts, "prompts")

    def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> GetPromptResult:
        """Retrieve a prompt template from the MCP server."""
        self._ensure_connected()
        return self._call(self._session.get_prompt, name, arguments or {})

    def ping(self) -> bool:
        """Ping the server to verify responsiveness."""
        self._ensure_connected()
        self._call(self._session.send_ping)
        return True

    async def connect_async(self) -> MCPClient:
        """Asynchronously connect and initialize the MCP session."""
        if self.is_connected:
            return self

        try:
            if self._params is not None:
                self._async_transport_cm = stdio_client(self._params)
            else:
                self._async_transport_cm = streamable_http_client(
                    self._url, http_client=self._http_client
                )

            read_stream, write_stream = await self._async_transport_cm.__aenter__()
            self._async_session_cm = ClientSession(read_stream, write_stream)
            self._session = await self._async_session_cm.__aenter__()
            self._init_result = await self._session.initialize()
            logger.info(
                "Connected (async) to MCP server '%s' (version %s) via %s: %s",
                self.server_name,
                self.server_version,
                self.transport,
                self._target_description,
            )
            return self
        except Exception as exc:
            await self.close_async()
            raise self._translate_error(exc) from exc

    async def close_async(self) -> None:
        """Asynchronously close the MCP session and terminate transport."""
        if self._async_session_cm is not None:
            try:
                await self._async_session_cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001, S110
                pass
            self._async_session_cm = None
            self._session = None
            self._init_result = None

        if self._async_transport_cm is not None:
            try:
                await self._async_transport_cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001, S110
                pass
            self._async_transport_cm = None

        if (
            self._owns_http_client
            and self._http_client is not None
            and not getattr(self._http_client, "is_closed", False)
        ):
            try:
                await self._http_client.aclose()
            except Exception:  # noqa: BLE001, S110
                pass

    async def __aenter__(self) -> MCPClient:  # noqa: PYI034 (py3.10 has no typing.Self)
        return await self.connect_async()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close_async()

    async def list_tools_async(self) -> list[Tool]:
        self._ensure_connected()
        tools: list[Tool] = []
        cursor: str | None = None
        while True:
            params = PaginatedRequestParams(cursor=cursor) if cursor else None
            result = await self._session.list_tools(params=params)
            tools.extend(result.tools)
            cursor = result.next_cursor
            if not cursor:
                return tools

    async def call_tool_async(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> CallToolResult:
        self._ensure_connected()
        return await self._session.call_tool(name, arguments or {})

    def _ensure_connected(self) -> None:
        if not self.is_connected or self._session is None:
            raise MCPConnectionError("MCPClient is not connected. Call connect() first.")

    def _list_all(self, method: Any, field: str) -> list[Any]:
        """Follow next_cursor so paginating servers return every item, not just page one."""
        items: list[Any] = []
        cursor: str | None = None
        while True:
            params = PaginatedRequestParams(cursor=cursor) if cursor else None
            # portal.call() takes no kwargs, hence the partial.
            result = self._call(functools.partial(method, params=params))
            items.extend(getattr(result, field))
            cursor = result.next_cursor
            if not cursor:
                return items

    def _call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        if self._portal is None:
            raise MCPConnectionError("Synchronous portal is not active.")
        try:
            return self._portal.call(func, *args, **kwargs)
        except Exception as exc:
            raise self._translate_error(exc) from exc

    def _translate_error(self, exc: BaseException) -> MCPError:
        unwrapped = _unwrap_exception(exc)
        if isinstance(unwrapped, MCPError):
            return unwrapped

        target = self._target_description

        if isinstance(unwrapped, FileNotFoundError):
            return MCPConnectionError(f"Cannot spawn MCP server: executable '{target}' not found.")
        if isinstance(unwrapped, PermissionError):
            return MCPConnectionError(f"Permission denied attempting to run executable '{target}'.")

        if isinstance(
            unwrapped,
            (
                httpx2.ConnectError,
                httpx2.NetworkError,
                httpx2.ConnectTimeout,
            ),
        ):
            return MCPConnectionError(
                f"Cannot connect to MCP HTTP server at '{target}': {unwrapped}"
            )

        if isinstance(unwrapped, httpx2.TimeoutException):
            return MCPTimeoutError(
                f"HTTP request timed out communicating with MCP server '{target}': {unwrapped}"
            )

        if isinstance(unwrapped, httpx2.HTTPStatusError):
            return MCPConnectionError(
                f"MCP server at '{target}' returned HTTP error {unwrapped.response.status_code}: {unwrapped}"
            )
        if isinstance(unwrapped, TimeoutError):
            return MCPTimeoutError(f"Timed out during MCP operation on '{target}': {unwrapped}")

        err_str = str(unwrapped).lower()
        if isinstance(unwrapped, SDKMCPError):
            if unwrapped.code == -32000 or "closed" in err_str or "broken pipe" in err_str:
                return MCPServerCrashedError(f"MCP server disconnected unexpectedly: {unwrapped}")
            return MCPToolCallError(f"MCP server error ({unwrapped.code}): {unwrapped.message}")

        if "closed" in err_str or "broken pipe" in err_str or "eof" in err_str:
            return MCPServerCrashedError(f"MCP server terminated unexpectedly: {unwrapped}")

        return MCPError(f"MCP operation failed: {unwrapped}")

    def __repr__(self) -> str:
        status = "connected" if self.is_connected else "disconnected"
        if self._params is not None:
            return (
                f"MCPClient(transport='stdio', command={self._params.command!r}, status={status!r})"
            )
        return f"MCPClient(transport='http', url={self._url!r}, status={status!r})"
