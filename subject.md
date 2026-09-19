# Agent Smith
## Autonomous reasoning, code generation, and execution

**Summary:** In this project, you will build an **agentic framework** capable of autonomously solving coding challenges.
Your agent will reason, write code, execute it in a sandboxed environment, and iterate until a solution is found.
This project introduces **Code Agents**, **Model Context Protocol (MCP)**, and controlled code execution.
*Made in collaboration with @ldevelle*  
*Version: 1.1*

---

## Contents
- I. Foreword
- II. AI Instructions
- III. Overview (What is a Code Agent?)
- IV. Common Instructions (General Rules & Technical Constraints)
- V. Mandatory Part
  - V.1 Agentic Framework
  - V.2 The Sandbox
  - V.3 MBPP Agent
  - V.4 SWE-bench Agent
  - V.5 Mandatory Tools
  - V.6 LLM API Providers
  - V.7 Model Benchmark Report
- VI. Evaluation
- VII. Readme Requirements
- VIII. Submission

---

## III. Overview
The agent operates through a structured loop: **Thought → Code → Observation**.
- **Code Agent**: Unlike JSON tool-calling, the LLM generates executable Python code to invoke tools, allowing persistent variables, loops, and complex logic.
- Evaluated on:
  - **MBPP**: algorithmic Python problems.
  - **SWE-bench**: real-world bug fixing in production repositories.

### Architectural Core
1. **Agent/Orchestrator**: Main loop calling LLM, extracting code, feeding it to the sandbox, collecting observations.
2. **Code Extraction**: Parses Python blocks / tool calls from LLM output.
3. **Sandbox**: Execution boundary enforcing safety (allowlisted imports, restricted filesystem paths, timeout, memory limits, no unauthorized network). Exposes MCP client tools as Python functions.
4. **`final_answer()`**: Injected sandbox built-in to signal completion and terminate the loop.
5. **MCP Server(s)**: Separate process via stdio or HTTP providing tools (`mcp_tools_mbpp.py`, `mcp_tools_swebench.py`).

---

## IV. Technical Constraints & Rules
- **Python 3.10** with **`uv`** package manager.
- No third-party agent orchestration frameworks (no LangChain, LangGraph, CrewAI, AutoGen, Smolagents, LlamaIndex). Loop orchestration must be your own code.
- Multi-token rotation and multi-provider LLM support (exclusively using free tiers).
- Output model: standardized `StepMetrics` and `SolutionOutput` (Pydantic).

---

## V. Mandatory Tools (SWE-bench)
- **File System**: `read_file(filepath, start_line, end_line)`, `edit_file(filepath, old_str, new_str)`, `list_files(directory, pattern)`
- **Code Search**: `search_code(pattern, file_pattern)`, `search_function_or_class_definition_in_code(name)`, `find_references(name, filepath, line)`
- **Execution**: `run_tests()`, `get_patch()`, `run_command(command, workdir)`

---

## VI. Evaluation Criteria
- **MBPP**: Pass 4/5 random tasks (≤10 iterations, ≤6k input tokens, ≤1.5k output tokens, ≤120s timeout per task).
- **SWE-bench**: Pass 2/3 random tasks (≤30 iterations, ≤300k input tokens, ≤10k output tokens, ≤900s timeout per task).
- **Sandbox Security**: Must pass all isolation tests (imports, builtins, network, path restrictions, timeouts, memory).
- **Benchmark Report**: `BENCHMARK_REPORT.md` comparing ≥5 models across ≥2 providers on ≥3 SWE-bench tasks.
