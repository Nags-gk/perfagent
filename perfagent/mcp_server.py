"""MCP server exposing the InferSim tools to any MCP client (Claude Code, Codex, IDEs).

    python -m perfagent.mcp_server            # stdio
    claude mcp add perfagent -- python -m perfagent.mcp_server

Each tool returns JSON text. The server keeps one ToolBox per process with a large budget,
so a coding agent can explore freely; the evaluation harness uses its own budgeted ToolBox.
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer

from .tools import ToolBox, ToolError

box = ToolBox(budget=10_000)
server = MCPServer(
    name="perfagent",
    instructions=(
        "LLM inference performance modeling tools (InferSim). Start with list_catalog, prune layouts with "
        "check_fit (free), then use simulate_step / roofline_ops / evaluate_layout. Latencies are modeled."
    ),
)


def _run(name: str, args: dict) -> str:
    try:
        return json.dumps(box.call(name, {k: v for k, v in args.items() if v is not None}))
    except ToolError as e:
        return json.dumps({"error": str(e)})


@server.tool(description="Models, hardware and options. Free.")
def list_catalog() -> str:
    return _run("list_catalog", {})


@server.tool(description="Memory check for a layout: devices, fit, KV-cache capacity in tokens. Free.")
def check_fit(model: str, hw: str, tp: int = 1, pp: int = 1, precision: str = "bf16",
              expert_parallel: bool = False) -> str:
    return _run("check_fit", locals())


@server.tool(description="Time one forward pass: latency, tokens/s, dominant bound and category, slowest ops.")
def simulate_step(model: str, hw: str, phase: str = "decode", batch: int = 1, tokens: int = 2048, tp: int = 1,
                  pp: int = 1, precision: str = "bf16", expert_parallel: bool = False) -> str:
    return _run("simulate_step", locals())


@server.tool(description="Every operator of one step with roofline bound, time and arithmetic intensity.")
def roofline_ops(model: str, hw: str, phase: str = "decode", batch: int = 1, tokens: int = 2048, tp: int = 1,
                 pp: int = 1, precision: str = "bf16", expert_parallel: bool = False) -> str:
    return _run("roofline_ops", locals())


@server.tool(description="Serving metrics for a layout at a decode batch: tokens/s per user and per device, TTFT.")
def evaluate_layout(model: str, hw: str, batch: int, tp: int = 1, pp: int = 1, precision: str = "bf16",
                    prompt: int = 2048, output: int = 256, expert_parallel: bool = False) -> str:
    return _run("evaluate_layout", locals())


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
