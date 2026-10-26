"""The agent: a LangGraph state machine around an LLM with simulation tools.

    model -> tools -> (submitted?) -> verify -> done
      ^        |                         |
      +--------+-------------------------+  (feedback when a check fails)

Reliability features:
* tool arguments are validated by the tools; errors go back to the model as tool results;
* the final answer must match the task's answer schema (Pydantic), or the model is told why;
* a verifier re-simulates design-space answers (charged to the same budget) and returns the
  failure once if the layout does not fit or misses the SLA;
* hard caps on model turns and simulations;
* every model call, tool call and check is an OpenTelemetry span, saved with the trace.
"""

from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import ValidationError, create_model

from .llm import QuotaExhausted
from .tasks import Task
from .tools import ToolBox, ToolError, openai_tools
from .tracing import Tracer

SYSTEM = """You are a performance-modeling engineer. You answer questions about LLM inference \
performance by running an analytical simulator (InferSim) through tools.

How to work:
- Start from what the question gives you; call list_catalog only if you need model or hardware facts.
- check_fit is free: use it to rule out layouts that cannot hold the weights or the KV cache before \
spending simulations.
- simulate_step, roofline_ops and evaluate_layout each cost one simulation from a fixed budget \
(shown in every result as budget_left). Do not repeat a simulation you already ran.
- For design-space questions, reason about trends to prune: larger batches raise throughput per \
device but lower tokens/s per user; more tensor parallelism raises tokens/s per user but costs devices; \
lower precision halves weight traffic. Search near the SLA boundary.
- For crossover questions, binary-search the batch size.
- Numbers come from the tools, never from memory.

When you are done, call submit_answer with `answer` exactly in this JSON shape (same keys and types):
{answer_format}
and a one-sentence `rationale`. Submit once."""

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}


def submit_spec(task: Task) -> dict:
    """submit_answer with a typed answer object (Gemini rejects object schemas without properties)."""
    props = {k: {"type": _JSON_TYPES[type(v)]} for k, v in task.answer_format.items()}
    if task.kind == "dse":
        props["hw"]["enum"] = task.params["hws"]
        props["precision"]["enum"] = task.params["precs"]
    return {
        "name": "submit_answer",
        "description": "Submit the final answer. Ends the task.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "object", "properties": props, "required": list(props)},
            "rationale": {"type": "string"}}, "required": ["answer", "rationale"]},
    }


class State(TypedDict, total=False):
    messages: list[dict]
    turns: int
    answer: dict | None
    rationale: str
    verify_rounds: int
    nudges: int
    status: str
    pending: list[dict]


def answer_model(task: Task):
    fields = {k: (type(v), ...) for k, v in task.answer_format.items()}
    if task.kind == "dse":
        fields["expert_parallel"] = (bool, False)
        fields["pp"] = (int, 1)
    return create_model(f"Answer_{task.kind}", **fields)


def _trim(obj: Any, limit: int = 2500) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + '..."(truncated)"'


class Agent:
    def __init__(self, llm, task: Task, budget: int = 30, max_turns: int = 40, tracer: Tracer | None = None):
        self.llm, self.task = llm, task
        self.box = ToolBox(budget=budget)
        self.max_turns = max_turns
        self.tracer = tracer or Tracer(task.id, getattr(llm, "name", "llm"))
        self.schema = answer_model(task)
        self.tools = openai_tools([submit_spec(task)])
        self.usage = {"in": 0, "out": 0, "llm_calls": 0}
        self.graph = self._build()

    # ---- graph ---------------------------------------------------------------
    def _build(self):
        g = StateGraph(State)
        g.add_node("model", self._model)
        g.add_node("tools", self._tools)
        g.add_node("verify", self._verify)
        g.set_entry_point("model")
        g.add_conditional_edges("model", self._after_model, {"tools": "tools", "model": "model", "end": END})
        g.add_conditional_edges("tools", self._after_tools, {"verify": "verify", "model": "model", "end": END})
        g.add_conditional_edges("verify", lambda s: "end" if s["status"] in ("accepted", "rejected") else "model",
                                {"model": "model", "end": END})
        return g.compile()

    def _model(self, s: State) -> State:
        with self.tracer.span("llm.call", turn=s["turns"]) as sp:
            t0 = time.time()
            r = self.llm.chat(s["messages"], self.tools)
            sp.set(tokens_in=r.usage.get("in", 0), tokens_out=r.usage.get("out", 0), ms=(time.time() - t0) * 1e3,
                   tool_calls=[c["name"] for c in r.tool_calls])
        self.usage["in"] += r.usage.get("in", 0)
        self.usage["out"] += r.usage.get("out", 0)
        self.usage["llm_calls"] += 1
        self.tracer.event("llm", text=r.content or "", tool_calls=r.tool_calls, tokens=r.usage)
        msgs = s["messages"] + [r.raw_message]
        out: State = {"messages": msgs, "turns": s["turns"] + 1, "pending": r.tool_calls}
        if not r.tool_calls:
            if s.get("nudges", 0) < 2 and s["turns"] + 1 < self.max_turns:
                out["messages"] = msgs + [{"role": "user", "content":
                                           "Use the tools to get numbers, then call submit_answer."}]
                out["nudges"] = s.get("nudges", 0) + 1
            else:
                out["status"] = "no_answer"
        return out

    def _after_model(self, s: State) -> str:
        if s.get("status") == "no_answer" or s["turns"] >= self.max_turns and not s.get("pending"):
            return "end"
        return "tools" if s.get("pending") else "model"

    def _tools(self, s: State) -> State:
        msgs = list(s["messages"])
        out: State = {}
        for c in s.get("pending", []):
            name, args = c["name"], c["args"]
            with self.tracer.span(f"tool.{name}", args=args) as sp:
                if name == "submit_answer":
                    content = self._submit(args, out)
                else:
                    before = self.box.used
                    try:
                        res = self.box.call(name, args)
                        content = _trim(res)
                        sp.set(cost=self.box.used - before, cached=bool(res.get("cached")))
                    except ToolError as e:
                        content = json.dumps({"error": str(e)})
                        sp.set(error=str(e))
                    except TypeError as e:  # unexpected/missing arguments
                        content = json.dumps({"error": f"bad arguments: {e}"})
                        sp.set(error=str(e))
            self.tracer.event("tool", name=name, args=args, result=content[:1200])
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": content})
        out["messages"] = msgs
        out["pending"] = []
        if s["turns"] >= self.max_turns and not out.get("answer"):
            out["status"] = "turn_limit"
        return out

    def _submit(self, args: dict, out: State) -> str:
        ans = args.get("answer")
        if isinstance(ans, str):
            try:
                ans = json.loads(ans)
            except json.JSONDecodeError:
                pass
        try:
            parsed = self.schema.model_validate(ans).model_dump()
        except ValidationError as e:
            return json.dumps({"error": "answer does not match the required shape",
                               "required": self.task.answer_format, "details": e.errors()[:3]}, default=str)
        out["answer"] = parsed
        out["rationale"] = str(args.get("rationale", ""))
        return json.dumps({"status": "received"})

    def _after_tools(self, s: State) -> str:
        if s.get("answer"):
            return "verify"
        if s.get("status") == "turn_limit":
            return "end"
        return "model"

    def _verify(self, s: State) -> State:
        """Re-check a design-space answer with the simulator; send one failure back."""
        a = s["answer"]
        if self.task.kind != "dse":
            return {"status": "accepted"}
        p = self.task.params
        with self.tracer.span("verify") as sp:
            try:
                r = self.box.call("evaluate_layout", {"model": p["model"], "hw": a["hw"], "tp": a["tp"], "pp": a["pp"],
                                                      "precision": a["precision"], "batch": a["batch"],
                                                      "expert_parallel": a["expert_parallel"],
                                                      "prompt": 2048, "output": 256})
                problem = None
                if not r.get("fits"):
                    problem = "the submitted layout does not fit"
                elif r["tok_s_per_user"] < p["sla"]:
                    problem = f"the submitted layout gives {r['tok_s_per_user']} tok/s per user, below the SLA"
                elif a["hw"] not in p["hws"] or a["precision"] not in p["precs"]:
                    problem = "the submitted hardware or precision is not allowed"
            except ToolError as e:
                problem = None if "budget" in str(e) else str(e)
            sp.set(problem=problem or "")
        self.tracer.event("verify", problem=problem)
        if problem is None:
            return {"status": "accepted"}
        if s.get("verify_rounds", 0) >= 1:
            return {"status": "rejected"}
        msg = {"role": "user", "content": f"Verification failed: {problem}. Fix it and submit again."}
        return {"messages": s["messages"] + [msg], "answer": None, "verify_rounds": s.get("verify_rounds", 0) + 1,
                "status": "retry"}

    # ---- run -----------------------------------------------------------------
    def run(self) -> dict:
        sys = SYSTEM.replace("{answer_format}", json.dumps(self.task.answer_format))
        init: State = {"messages": [{"role": "system", "content": sys}, {"role": "user", "content": self.task.prompt}],
                       "turns": 0, "answer": None, "verify_rounds": 0, "nudges": 0, "status": "running"}
        t0 = time.time()
        with self.tracer.span("task", task=self.task.id, kind=self.task.kind):
            try:
                final = self.graph.invoke(init, {"recursion_limit": 4 * self.max_turns + 20})
                error = None
            except QuotaExhausted:
                raise  # the caller stops the whole run; this task is not scored
            except Exception as e:  # noqa: BLE001 - provider failure: record it, keep the eval running
                final, error = {"answer": None, "status": "error"}, f"{type(e).__name__}: {e}"
        grade = self.task.grade(final.get("answer"))
        return {
            "task": self.task.id, "kind": self.task.kind, "agent": getattr(self.llm, "name", "llm"),
            "answer": final.get("answer"), "rationale": final.get("rationale", ""), "status": final.get("status"),
            "error": error, **grade, "simulations": self.box.used, "tool_calls": self.box.calls,
            "llm_calls": self.usage["llm_calls"], "tokens_in": self.usage["in"], "tokens_out": self.usage["out"],
            "seconds": round(time.time() - t0, 2), "trace": self.tracer.export(),
        }
