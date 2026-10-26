import asyncio
import json
import sys

import pytest

from perfagent.agent import Agent, answer_model
from perfagent.baselines import optuna_search, random_search
from perfagent.llm import Reply, ScriptedPolicy, _pick_model
from perfagent.tasks import build_suite, dse_space, score_layout
from perfagent.tools import TOOL_SPECS, ToolBox, ToolError, openai_tools

SUITE = {t.id: t for t in build_suite()}


def by_kind(kind):
    return [t for t in SUITE.values() if t.kind == kind]


# ---------------------------------------------------------------- tools
def test_budget_cache_and_errors():
    box = ToolBox(budget=2)
    args = {"model": "llama-3.1-8b", "hw": "h100", "phase": "decode", "batch": 4, "tokens": 1024}
    box.call("simulate_step", args)
    assert box.call("simulate_step", args)["cached"] and box.used == 1
    box.call("check_fit", {"model": "llama-3.1-8b", "hw": "h100"})  # free
    box.call("roofline_ops", args | {"batch": 8})
    with pytest.raises(ToolError, match="budget"):
        box.call("simulate_step", args | {"batch": 16})
    with pytest.raises(ToolError, match="not one of"):
        box.call("check_fit", {"model": "gpt-x", "hw": "h100"})
    with pytest.raises(ToolError, match="power of two"):
        box.call("check_fit", {"model": "llama-3.1-8b", "hw": "h100", "tp": 3})


def test_tool_specs_are_valid_function_schemas():
    names = {s["name"] for s in TOOL_SPECS}
    assert names == {"list_catalog", "check_fit", "simulate_step", "roofline_ops", "evaluate_layout"}
    for t in openai_tools():
        p = t["function"]["parameters"]
        assert p["type"] == "object" and set(p.get("required", [])) <= set(p["properties"])
    json.dumps(openai_tools())


# ---------------------------------------------------------------- tasks
def test_suite_shape_and_truth_grades_perfectly():
    kinds = [t.kind for t in SUITE.values()]
    assert len(kinds) == 25 and set(kinds) == {"dse", "bottleneck", "sizing", "crossover", "compare"}
    for t in SUITE.values():
        if t.kind == "dse":
            ans = t.truth["best"]
        elif t.kind == "compare":
            ans = {"faster": t.truth["faster"], "ratio": t.truth["ratio"]}
        else:
            ans = {k: v for k, v in t.truth.items()}
        g = t.grade(ans)
        assert g["success"] and g["score"] == 1.0, (t.id, g)
        assert not t.grade({})["success"]


def test_dse_truth_is_the_exhaustive_optimum():
    t = by_kind("dse")[0]
    p = t.params
    best = max(score_layout(p["model"], c, p["sla"])[0] for c in dse_space(p["model"], tuple(p["hws"]), tuple(p["precs"])))
    assert best == pytest.approx(t.truth["best_tok_s_device"])
    assert t.grade({**t.truth["best"], "hw": "b200"} if "b200" not in p["hws"] else {})["score"] == 0.0


# ---------------------------------------------------------------- agent loop
@pytest.mark.parametrize("kind", ["bottleneck", "sizing", "crossover", "compare"])
def test_scripted_agent_solves_exact_tasks(kind):
    for t in by_kind(kind):
        r = Agent(ScriptedPolicy(t), t).run()
        assert r["success"], (t.id, r["detail"], r["error"])
        assert r["simulations"] <= 30
        assert r["trace"]["events"] and r["trace"]["otel_spans"]


class FakeLLM:
    """Replays a fixed list of tool calls, one per model turn."""

    name = "fake"

    def __init__(self, calls):
        self.calls = list(calls)
        self.seen = []

    def chat(self, messages, tools):
        self.seen.append(messages[-1])
        if not self.calls:
            return Reply("nothing left", [], {}, {"role": "assistant", "content": "nothing left"})
        name, args = self.calls.pop(0)
        raw = {"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{len(self.calls)}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}
        return Reply(None, [{"id": f"c{len(self.calls)}", "name": name, "args": args}], {"in": 10, "out": 5}, raw)


def test_verifier_rejects_sla_violation_then_accepts_fix():
    t = by_kind("dse")[0]
    good = t.truth["best"]
    bad = dict(good, batch=512, tp=1)  # max batch on one GPU: misses the SLA
    llm = FakeLLM([("submit_answer", {"answer": bad, "rationale": "x"}),
                   ("submit_answer", {"answer": good, "rationale": "y"})])
    r = Agent(llm, t).run()
    assert r["status"] == "accepted" and r["success"]
    assert any("Verification failed" in str(m.get("content", "")) for m in llm.seen)


def test_malformed_answer_gets_schema_error_and_retry():
    t = by_kind("sizing")[0]
    llm = FakeLLM([("submit_answer", {"answer": {"gpus": "eight"}, "rationale": "x"}),
                   ("submit_answer", {"answer": dict(t.truth), "rationale": "y"})])
    r = Agent(llm, t).run()
    assert r["success"]
    assert any("required shape" in str(m.get("content", "")) for m in llm.seen)


def test_tool_errors_are_returned_not_raised():
    t = by_kind("bottleneck")[0]
    llm = FakeLLM([("simulate_step", {"model": "nope", "hw": "h100", "phase": "decode", "batch": 1, "tokens": 1}),
                   ("simulate_step", {"bogus_arg": 1}),
                   ("submit_answer", {"answer": dict(t.truth), "rationale": "z"})])
    r = Agent(llm, t).run()
    assert r["success"] and r["error"] is None
    errs = [e for e in r["trace"]["events"] if e["type"] == "tool" and "error" in e["result"]]
    assert len(errs) == 2


def test_turn_limit_ends_run():
    t = by_kind("bottleneck")[0]
    llm = FakeLLM([("check_fit", {"model": "llama-3.1-8b", "hw": "h100"})] * 50)
    r = Agent(llm, t, max_turns=5).run()
    assert not r["success"] and r["llm_calls"] <= 5


def test_answer_model_types():
    t = by_kind("dse")[0]
    m = answer_model(t)
    assert m.model_validate({"hw": "h100", "precision": "bf16", "tp": 2, "batch": 8}).pp == 1


# ---------------------------------------------------------------- baselines and providers
def test_baselines_respect_budget():
    t = by_kind("dse")[1]
    for fn in (random_search, optuna_search):
        r = fn(t, budget=12, seed=1)
        assert r["simulations"] <= 12 and 0.0 <= r["score"] <= 1.0


def test_model_auto_pick():
    ids = ["models/gemini-2.5-flash-lite", "models/gemini-2.5-flash", "models/gemini-2.5-pro", "models/text-embedding-004"]
    assert _pick_model(ids, "gemini") == "gemini-2.5-flash"
    assert _pick_model(["llama-3.1-8b-instant", "llama-3.3-70b-versatile"], "groq") == "llama-3.3-70b-versatile"


def test_mcp_server_over_stdio():
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable, args=["-m", "perfagent.mcp_server"])
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            names = {t.name for t in (await s.list_tools()).tools}
            res = await s.call_tool("check_fit", {"model": "llama-3.1-70b", "hw": "h100", "tp": 4})
            return names, json.loads(res.content[0].text)

    names, fit = asyncio.run(go())
    assert {"check_fit", "evaluate_layout"} <= names and fit["fits"] is True
