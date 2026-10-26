# PerfAgent

[![ci](https://github.com/Nags-gk/perfagent/actions/workflows/ci.yml/badge.svg)](https://github.com/Nags-gk/perfagent/actions/workflows/ci.yml)

**Live: https://nags-gk.github.io/perfagent/** (results, step-by-step traces, and a bring-your-own-key
agent that runs in the browser)

An LLM agent that automates performance-modeling work. It drives the
[InferSim](https://github.com/Nags-gk/infersim) LLM-inference simulator through tools to find
bottlenecks, size deployments and search the design space, and is scored against
exhaustive-search ground truth, next to Optuna and random search under the same simulation budget.

## Benchmark

25 tasks, 5 kinds:

| Kind | Question | Scoring |
|---|---|---|
| `dse` (8) | Best layout (hardware, precision, TP, PP, batch) maximizing tokens/s per device under a tokens/s-per-user SLA. Spaces of 120-480 layouts; 30 simulations allowed. | fraction of the exhaustive optimum; success if at least 95% and the SLA holds |
| `bottleneck` (6) | Slowest operator in a step, and the step's dominant roofline bound | exact |
| `sizing` (4) | Fewest GPUs that hold the weights plus a KV-cache requirement | exact |
| `crossover` (4) | Smallest decode batch at which an FFN GEMM turns compute-bound | within 2% |
| `compare` (3) | Which of two layouts is faster per user, and by what ratio | within 5% |

Ground truth comes from brute force over the same simulator, so the benchmark measures search
and reasoning, not the simulator's accuracy (InferSim's own calibration is reported in its repo).

### Results

| Agent | Design space: within 95% of optimum | Mean fraction of optimum | Exact-answer tasks |
|---|---|---|---|
| Optuna TPE (5 seeds) | 70% | 93.8% | n/a |
| Random search (5 seeds) | 57% | 89.4% | n/a |
| Rule-based strategy (reads task parameters) | 38% | 50.6% | 17/17 |
| Gemini agent | run the `eval` workflow | | |

## How it works

```
task prompt -> LLM -> tools (check_fit, simulate_step, roofline_ops, evaluate_layout) -> InferSim
                ^                                                                  |
                +-------------------- results / errors ----------------------------+
            -> submit_answer (typed schema) -> verifier (re-simulates) -> grader (exhaustive truth)
```

- **Agent:** a LangGraph state machine (model, tools, verify). Gemini, Groq and local Ollama models all
  run through the OpenAI-compatible API; a Flash model is picked automatically.
- **Tools:** argument validation with errors returned to the model; `check_fit` and `list_catalog` are
  free, simulations are budgeted and cached.
- **Reliability:** answers must match a per-task schema (Pydantic); a verifier re-simulates
  design-space answers and returns a failure once; turn and budget caps; provider errors are
  recorded instead of crashing the run.
- **Observability:** every model call, tool call and check is an OpenTelemetry span, saved with a
  readable event log per run and replayed on the dashboard.
- **MCP:** the tools are also an MCP server, so coding agents can use the simulator directly:
  `claude mcp add perfagent -- python -m perfagent.mcp_server` (also configured in `.mcp.json`).
- **In the browser:** the live page runs the same tools and graders in Pyodide and calls Gemini's
  REST API from the page with your own key.

## Run it

```bash
pip install -e ".[dev]"
pytest -q                                              # 16 tests, including MCP over stdio
python -m perfagent.eval --agents scripted random optuna   # offline agents and baselines
GEMINI_API_KEY=... python -m perfagent.eval --agents gemini --pause 6
python -m perfagent.mcp_server                         # MCP server on stdio
```

In GitHub, the `eval` workflow runs the Gemini agent on all 25 tasks with the `GEMINI_API_KEY`
repository secret and publishes the results to the dashboard.

## Limitations

- The simulator is analytical; answers are right relative to InferSim, not to hardware.
- One run per LLM configuration at temperature 0; LLM results vary between runs and model versions.
- The rule-based strategy reads task parameters directly, so it is a strategy reference, not a
  like-for-like competitor to an agent that must parse the prompt.

MIT licensed.
