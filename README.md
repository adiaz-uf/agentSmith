*This project has been created as part of the 42 curriculum by adiaz-uf.*

# Agent Smith

> **Autonomous reasoning, code generation, and execution**

## Description

**Agent Smith** is an autonomous agentic framework designed to solve software engineering tasks and algorithmic coding challenges (MBPP & SWE-bench Verified).

Unlike traditional LLM workflows that rely on static prompts or simple JSON tool calls, Agent Smith implements a dynamic **Thought → Code → Observation** execution loop. The agent reasons about tasks, generates executable Python code, interacts with decoupled MCP (Model Context Protocol) tool servers, executes code safely inside an isolated sandbox, and iteratively refines its solution until verified.

### Key Objectives
- **Agentic Loop**: Autonomous iterative problem solving with code-based tool calling.
- **Configurable Sandbox**: Strict execution boundary preventing privilege escalation, unauthorized imports, network access, or path escapes.
- **Model Context Protocol (MCP)**: Clean tool abstraction over stdio/HTTP transports.
- **Multi-Benchmark**: Evaluated on both **MBPP** (Mostly Basic Python Problems) and **SWE-bench** (real-world repo bug fixing).
- **Multi-Provider LLM Support**: Abstract provider layer supporting free-tier quotas and token rotation.

---

## Instructions

### Prerequisites
- Python 3.10
- [`uv`](https://github.com/astral-sh/uv) package manager
- Docker (for SWE-bench environments)

### Setup
```bash
# Clone the repository
git clone git@github-adiaz:adiaz-uf/agentSmith.git
cd agentSmith

# Install dependencies
uv sync
```

### Execution

#### Interactive Sandbox
```bash
uv run sandbox
```

#### Run on MBPP
```bash
uv run python -m agent_mbpp --task-file <path_to_task.json> \
  --output <path_to_solution.json> \
  --model-name "<model/name>" \
  --provider-url "<provider_url>"
```

#### Run on SWE-bench
```bash
uv run python -m agent_swebench --task-file <path_to_task.json> \
  --output <path_to_solution.json> \
  --model-name "<model/name>" \
  --provider-url "<provider_url>"
```

---

## Resources

- [Model Context Protocol (MCP) Documentation](https://modelcontextprotocol.io/)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues?](https://www.swebench.com/)
- [MBPP (Mostly Basic Python Problems) Dataset](https://github.com/google-research/google-research/tree/master/mbpp)
