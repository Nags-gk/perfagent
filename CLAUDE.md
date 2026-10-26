# PerfAgent: notes for coding agents

- `perfagent/tools.py` and `perfagent/tasks.py` must stay pure Python (plus InferSim): they also run in the browser via Pyodide.
- Never put API keys in code, commits or chat. Providers read `GEMINI_API_KEY` / `GROQ_API_KEY` from the environment.
- Ground truth comes from brute force over InferSim; do not hand-edit expected answers.
- After changes: `ruff check . && pytest -q`, then `python -m perfagent.eval --agents scripted random optuna` and
  `python scripts/build_site.py`.
- This repo ships an MCP server (`.mcp.json`): the simulator tools are available to you while you work here.
