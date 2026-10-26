"""Non-LLM search baselines for the design-space tasks, under the same simulation budget.

* random search over the task's discrete space;
* Optuna's TPE sampler (Bayesian optimization) with categorical parameters.

Both evaluate through the same budgeted ToolBox the agent uses, so a cached repeat is free
and every distinct layout costs one simulation.
"""

from __future__ import annotations

import random
import time

from .tasks import DSE_BATCH, DSE_PP, DSE_TP, OUTPUT, PROMPT, Task, dse_space
from .tools import ToolBox, ToolError

MOE = ("mixtral-8x7b", "deepseek-v3")


def _eval(box: ToolBox, model: str, cfg: dict) -> dict | None:
    try:
        return box.call("evaluate_layout", {"model": model, **cfg, "prompt": PROMPT, "output": OUTPUT})
    except ToolError:
        return None


def _result(task: Task, name: str, box: ToolBox, best_cfg: dict | None, t0: float, seed: int) -> dict:
    p = task.params
    ans = best_cfg or {"hw": p["hws"][0], "precision": p["precs"][0], "tp": 8, "pp": 1, "expert_parallel": False,
                       "batch": 1}
    g = task.grade(ans)
    return {"task": task.id, "kind": task.kind, "agent": name, "seed": seed, "answer": ans, **g,
            "simulations": box.used, "tool_calls": box.calls, "llm_calls": 0, "tokens_in": 0, "tokens_out": 0,
            "seconds": round(time.time() - t0, 2)}


def _better(r: dict | None, sla: float, best: float) -> bool:
    return bool(r and r.get("fits") and r["tok_s_per_user"] >= sla and r["tok_s_per_device"] > best)


def random_search(task: Task, budget: int = 30, seed: int = 0) -> dict:
    p, t0 = task.params, time.time()
    box = ToolBox(budget=budget)
    space = dse_space(p["model"], tuple(p["hws"]), tuple(p["precs"]))
    random.Random(seed).shuffle(space)
    best, best_cfg = 0.0, None
    for cfg in space[:budget]:
        r = _eval(box, p["model"], cfg)
        if _better(r, p["sla"], best):
            best, best_cfg = r["tok_s_per_device"], cfg
    return _result(task, "random-search", box, best_cfg, t0, seed)


def optuna_search(task: Task, budget: int = 30, seed: int = 0) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    p, t0 = task.params, time.time()
    box = ToolBox(budget=budget)
    best = {"v": 0.0, "cfg": None}

    def objective(trial):
        cfg = {"hw": trial.suggest_categorical("hw", p["hws"]),
               "precision": trial.suggest_categorical("precision", p["precs"]),
               "tp": trial.suggest_categorical("tp", list(DSE_TP)),
               "pp": trial.suggest_categorical("pp", list(DSE_PP)),
               "batch": trial.suggest_categorical("batch", list(DSE_BATCH)),
               "expert_parallel": trial.suggest_categorical("ep", [False, True]) if p["model"] in MOE else False}
        r = _eval(box, p["model"], cfg)
        if r is None:
            raise optuna.TrialPruned()
        if not r.get("fits"):
            return -2.0
        if r["tok_s_per_user"] < p["sla"]:
            return -1.0 + r["tok_s_per_user"] / p["sla"]  # shaped: closer to the SLA is better
        if r["tok_s_per_device"] > best["v"]:
            best["v"], best["cfg"] = r["tok_s_per_device"], cfg
        return r["tok_s_per_device"]

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8))
    for _ in range(budget * 8):  # repeats are cached and free; stop when the budget is spent
        if box.used >= budget:
            break
        study.optimize(objective, n_trials=1)
    return _result(task, "optuna-tpe", box, best["cfg"], t0, seed)
