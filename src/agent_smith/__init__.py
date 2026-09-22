"""Agent Smith: autonomous reasoning, code generation, and execution.

Package layout:
    agent_smith.core        shared Pydantic models and the agent loop
    agent_smith.llm         LLM provider abstraction, token rotation, usage tracking
    agent_smith.mcp_client  MCP client (stdio + streamable HTTP transports)
    agent_smith.sandbox     the sandboxed execution boundary and its REPL
"""

__version__ = "0.1.0"
