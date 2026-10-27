"""Run agents and baselines on the task suite and write results for the dashboard.

    python -m perfagent.eval --agents scripted random optuna          # offline
    GEMINI_API_KEY=... python -m perfagent.eval --agents gemini        # live LLM

Writes ``site/data/results.json`` (merged with earlier runs of other agents) and one trace
file per (agent, task) under ``site/data/traces/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from .agent import Agent
from .baselines import optuna_search, random_search
from .llm import OpenAICompatLLM, QuotaExhausted, ScriptedPolicy
from .tasks import build_suite

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "site" / "data"


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in name)


def run_agent(kind: str, tasks, budget: int, seeds: int, model: str | None, pause: float, max_turns: int = 25,
              done: set | None = None, on_row=None) -> list[dict]:
    """Run one agent over ``tasks``. ``on_row`` is called after every task so partial runs are saved;
    tasks whose id is in ``done`` are skipped (resume)."""
    rows = []
    emit = on_row or (lambda r: None)
    if kind in ("random", "optuna"):
        fn = random_search if kind == "random" else optuna_search
        for t in tasks:
            if t.kind == "dse":
                for s in range(seeds):
                    rows.append(fn(t, budget, s))
                    emit(rows[-1])
        return rows
    llm = None if kind == "scripted" else OpenAICompatLLM(kind, model)
    dead = 0  # consecutive tasks where the provider never answered
    for i, t in enumerate(tasks):
        if done and t.id in done:
            continue
        agent = Agent(ScriptedPolicy(t) if llm is None else llm, t, budget=budget, max_turns=max_turns)
        try:
            r = agent.run()
        except QuotaExhausted as e:
            left = [x.id for x in tasks[i:] if not (done and x.id in done)]
            print(f"  STOP: {e}\n  {len(left)} task(s) not run; rerun later to resume.", flush=True)
            break
        dead = dead + 1 if (r.get("error") and r["llm_calls"] == 0) else 0
        if dead >= 2:
            print(f"  STOP: the provider failed on {dead} tasks in a row without answering once: {r['error']}",
                  flush=True)
            break
        trace = r.pop("trace")
        name = _safe(f"{r['agent']}__{t.id}")
        (DATA / "traces").mkdir(parents=True, exist_ok=True)
        (DATA / "traces" / f"{name}.json").write_text(json.dumps(trace))
        r["trace_file"] = f"traces/{name}.json"
        rows.append(r)
        emit(r)
        print(f"  {r['agent']:28s} {t.id:48s} {('OK' if r['success'] else '--')} score={r['score']:.2f} "
              f"sims={r['simulations']:2d} llm={r['llm_calls']:2d} {r['detail']} {r.get('error') or ''}", flush=True)
        if llm is not None and pause:
            time.sleep(pause)  # stay under free-tier rate limits
    if llm is not None:
        print(f"  {llm.name}: {llm.requests} requests, {llm.retries} rate-limit retries", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    out = {}
    for agent in sorted({r["agent"] for r in rows}):
        rs = [r for r in rows if r["agent"] == agent]
        by_kind = {}
        for k in sorted({r["kind"] for r in rs}):
            kr = [r for r in rs if r["kind"] == k]
            by_kind[k] = {"n": len(kr), "success_rate": sum(r["success"] for r in kr) / len(kr),
                          "mean_score": statistics.mean(r["score"] for r in kr),
                          "mean_simulations": statistics.mean(r["simulations"] for r in kr)}
        out[agent] = {"n": len(rs), "success_rate": sum(r["success"] for r in rs) / len(rs),
                      "mean_score": statistics.mean(r["score"] for r in rs),
                      "mean_simulations": statistics.mean(r["simulations"] for r in rs),
                      "llm_calls": sum(r["llm_calls"] for r in rs), "tokens_in": sum(r["tokens_in"] for r in rs),
                      "tokens_out": sum(r["tokens_out"] for r in rs), "by_kind": by_kind}
    return out


def _write(path: Path, rows: list[dict], budget: int) -> dict:
    tasks_meta = [{"id": t.id, "kind": t.kind, "prompt": t.prompt, "truth": t.truth} for t in build_suite()]
    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "budget": budget,
           "tasks": tasks_meta, "summary": summarize(rows), "rows": rows}
    path.write_text(json.dumps(out, default=str))
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", default=["scripted", "random", "optuna"],
                    help="scripted, random, optuna, gemini, groq, ollama")
    ap.add_argument("--model", help="provider model id (default: auto-pick)")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--max-turns", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=5, help="seeds for the stochastic baselines")
    ap.add_argument("--only", help="substring filter on task ids")
    ap.add_argument("--pause", type=float, default=0.0, help="seconds between LLM tasks")
    ap.add_argument("--fresh", action="store_true", help="rerun tasks that already have results for this agent")
    a = ap.parse_args(argv)
    tasks = [t for t in build_suite() if not a.only or a.only in t.id]
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "results.json"
    rows = json.loads(path.read_text())["rows"] if path.exists() else []
    for kind in a.agents:
        print(f"== {kind}", flush=True)
        llm_kind = kind not in ("scripted", "random", "optuna")
        if not llm_kind or a.fresh:  # deterministic agents and --fresh: replace this agent's rows for these tasks
            prefix = {"scripted": "scripted-expert", "random": "random-search", "optuna": "optuna-tpe"}.get(kind, kind + ":")
            ids = {t.id for t in tasks}
            rows = [r for r in rows if not (r["agent"].startswith(prefix) and r["task"] in ids)]
            done = set()
        else:  # resume: skip tasks this provider already completed without a provider error
            done = {r["task"] for r in rows if r["agent"].startswith(kind + ":") and not r.get("error")}
            rows = [r for r in rows if not (r["agent"].startswith(kind + ":") and r.get("error"))]

        def save(r, rows=rows):
            rows.append(r)
            _write(path, rows, a.budget)

        run_agent(kind, tasks, a.budget, a.seeds, a.model, a.pause, a.max_turns, done, save)
    out = _write(path, rows, a.budget)
    for agent, s in out["summary"].items():
        print(f"{agent:30s} tasks {s['n']:3d}  success {s['success_rate']:.0%}  mean score {s['mean_score']:.3f}  "
              f"sims {s['mean_simulations']:.1f}")


if __name__ == "__main__":
    main()
