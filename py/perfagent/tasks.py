"""Benchmark tasks with ground truth from exhaustive search over the simulator.

Five kinds, each a real question a performance-modeling team answers:

* ``dse``: best layout under an interactivity SLA (design space exploration). The search space
  has hundreds of points; agents get a budget of 30 simulations. Scored as achieved throughput
  per device / optimum, zero if the answer violates the SLA or does not fit.
* ``bottleneck``: the slowest operator in a step and the step's dominant roofline bound.
* ``sizing``: fewest GPUs that hold the weights plus a KV-cache requirement.
* ``crossover``: smallest decode batch at which an FFN GEMM turns compute-bound (roofline ridge).
* ``compare``: which of two layouts is faster per user, and by what ratio.

Ground truth is computed with the same simulator the agent queries, by brute force, so the
benchmark measures search and reasoning, not model accuracy.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cache

from infersim.dse import evaluate
from infersim.ops import StepShape
from infersim.system import GPUDeployment

DSE_HW = ("h100", "b200")
DSE_PREC = ("bf16", "fp8")
DSE_TP = (1, 2, 4, 8)
DSE_PP = (1, 2, 4)
DSE_BATCH = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
PROMPT, OUTPUT = 2048, 256


@dataclass
class Task:
    id: str
    kind: str
    prompt: str
    answer_format: dict
    truth: dict
    params: dict = field(default_factory=dict)

    def grade(self, answer: dict | None) -> dict:
        return GRADERS[self.kind](self, answer or {})


# ---------------------------------------------------------------- dse
def dse_space(model: str, hws=DSE_HW, precs=DSE_PREC) -> list[dict]:
    moe = model in ("mixtral-8x7b", "deepseek-v3")
    pts = []
    for hw, prec, tp, pp in itertools.product(hws, precs, DSE_TP, DSE_PP):
        for ep in ((False, True) if moe and tp > 1 else (False,)):
            for b in DSE_BATCH:
                pts.append({"hw": hw, "precision": prec, "tp": tp, "pp": pp, "expert_parallel": ep, "batch": b})
    return pts


@cache
def _eval_point(model, hw, precision, tp, pp, expert_parallel, batch):
    d = GPUDeployment(model, hw, tp=tp, pp=pp, precision=precision, expert_parallel=expert_parallel)
    return evaluate(d, batch, PROMPT, OUTPUT)


def score_layout(model: str, cfg: dict, sla: float) -> tuple[float, str]:
    """Throughput per device of ``cfg`` if it is valid and meets the SLA, else (0, reason)."""
    try:
        p = _eval_point(model, cfg["hw"], cfg.get("precision", "bf16"), int(cfg.get("tp", 1)), int(cfg.get("pp", 1)),
                        bool(cfg.get("expert_parallel", False)), int(cfg["batch"]))
    except (KeyError, TypeError, ValueError) as e:  # malformed answer
        return 0.0, f"invalid layout: {e}"
    if p is None:
        return 0.0, "does not fit"
    if p.tok_s_user < sla:
        return 0.0, f"violates SLA ({p.tok_s_user:.1f} < {sla} tok/s/user)"
    return p.tok_s_device, "ok"


def dse_truth(model: str, sla: float, hws=DSE_HW, precs=DSE_PREC) -> dict:
    best, best_cfg, n = 0.0, None, 0
    for cfg in dse_space(model, hws, precs):
        n += 1
        s, _ = score_layout(model, cfg, sla)
        if s > best:
            best, best_cfg = s, cfg
    return {"best_tok_s_device": best, "best": best_cfg, "space_size": n}


def _grade_dse(t: Task, a: dict) -> dict:
    hws, precs = t.params["hws"], t.params["precs"]
    if str(a.get("hw", "")).lower() not in hws or str(a.get("precision", "bf16")).lower() not in precs:
        return {"score": 0.0, "success": False, "detail": "outside the allowed hardware/precision"}
    if int(a.get("tp", 1) or 1) not in DSE_TP or int(a.get("pp", 1) or 1) not in DSE_PP or \
            int(a.get("batch", 0) or 0) not in DSE_BATCH:
        return {"score": 0.0, "success": False, "detail": "tp/pp/batch outside the search space"}
    s, why = score_layout(t.params["model"], a, t.params["sla"])
    opt = t.truth["best_tok_s_device"]
    ratio = s / opt if opt else 0.0
    return {"score": round(ratio, 4), "success": ratio >= 0.95, "detail": why if s == 0 else f"{ratio:.1%} of optimum"}


# ---------------------------------------------------------------- bottleneck
def _grade_bottleneck(t: Task, a: dict) -> dict:
    op_ok = str(a.get("op", "")).strip().lower() == t.truth["op"]
    b_ok = str(a.get("bound", "")).strip().lower() == t.truth["bound"]
    return {"score": (op_ok + b_ok) / 2, "success": op_ok and b_ok,
            "detail": f"op {'ok' if op_ok else 'wrong'}, bound {'ok' if b_ok else 'wrong'}"}


# ---------------------------------------------------------------- sizing
def sizing_truth(model: str, hw: str, precision: str, users: int, ctx: int) -> dict:
    best = None
    for tp, pp in itertools.product(DSE_TP, DSE_PP):
        d = GPUDeployment(model, hw, tp=tp, pp=pp, precision=precision)
        if d.fits() and d.kv_capacity_tokens() >= users * ctx:
            key = (tp * pp, pp)  # fewest devices, prefer less pipelining on ties
            if best is None or key < best[0]:
                best = (key, tp, pp)
    if best is None:
        return {"devices": None}
    return {"devices": best[0][0], "tp": best[1], "pp": best[2]}


def _grade_sizing(t: Task, a: dict) -> dict:
    want = t.truth["devices"]
    try:
        got = int(a.get("devices"))
    except (TypeError, ValueError):
        got = None
    return {"score": float(got == want), "success": got == want, "detail": f"answered {got}, truth {want}"}


# ---------------------------------------------------------------- crossover
def crossover_truth(model: str, hw: str, precision: str, op: str = "gate_up", ctx: int = 1024) -> dict:
    d = GPUDeployment(model, hw, tp=1, precision=precision)
    for b in range(1, 4097):
        t = next(x for x in d.step(StepShape.decode(b, ctx)).timings if x.op.name == op)
        if t.bound == "compute":
            return {"batch": b}
    return {"batch": None}


def _grade_crossover(t: Task, a: dict) -> dict:
    want = t.truth["batch"]
    try:
        got = int(a.get("batch"))
    except (TypeError, ValueError):
        got = None
    ok = got is not None and abs(got - want) <= max(1, 0.02 * want)
    return {"score": float(ok), "success": ok, "detail": f"answered {got}, truth {want}"}


# ---------------------------------------------------------------- compare
def _tok_s_user(layout: dict, model: str, batch: int, ctx: int) -> float:
    d = GPUDeployment(model, layout["hw"], tp=layout["tp"], precision=layout["precision"])
    return 1.0 / d.step(StepShape.decode(batch, ctx)).latency_s


def _grade_compare(t: Task, a: dict) -> dict:
    f_ok = str(a.get("faster", "")).strip().lower() == t.truth["faster"]
    try:
        ratio = float(a.get("ratio"))
    except (TypeError, ValueError):
        ratio = 0.0
    r_ok = abs(ratio - t.truth["ratio"]) <= 0.05 * t.truth["ratio"]
    return {"score": (f_ok + r_ok) / 2, "success": f_ok and r_ok,
            "detail": f"faster {'ok' if f_ok else 'wrong'}, ratio {ratio:.3g} vs {t.truth['ratio']:.3g}"}


GRADERS = {"dse": _grade_dse, "bottleneck": _grade_bottleneck, "sizing": _grade_sizing,
           "crossover": _grade_crossover, "compare": _grade_compare}


# ---------------------------------------------------------------- suite
def build_suite() -> list[Task]:
    tasks: list[Task] = []
    for model, sla, hws, precs in [
            ("llama-3.1-8b", 100, ("h100",), ("bf16",)), ("llama-3.1-8b", 300, DSE_HW, DSE_PREC),
            ("llama-3.1-70b", 50, ("h100",), ("bf16", "fp8")), ("llama-3.1-70b", 120, DSE_HW, ("bf16",)),
            ("mixtral-8x7b", 80, ("h100",), ("bf16",)), ("llama-3.1-405b", 30, ("h100",), ("fp8",)),
            ("deepseek-v3", 40, ("b200",), ("fp8",)), ("qwen2.5-1.5b", 400, DSE_HW, DSE_PREC)]:
        moe = " expert_parallel may be true for this MoE model." if model in ("mixtral-8x7b", "deepseek-v3") else ""
        space = (f"hardware {' or '.join(hws)}; precision {' or '.join(precs)}; tp in 1,2,4,8; pp in 1,2,4; "
                 f"decode batch in {','.join(map(str, DSE_BATCH))}.{moe}")
        tasks.append(Task(
            id=f"dse-{model}-{sla}-{'+'.join(hws)}-{'+'.join(precs)}", kind="dse",
            params={"model": model, "sla": sla, "hws": list(hws), "precs": list(precs)},
            prompt=(f"Find the serving layout for {model} that maximizes output tokens/s per device while giving "
                    f"every user at least {sla} tokens/s, for {PROMPT}-token prompts and {OUTPUT}-token answers. "
                    f"Allowed: {space} Use evaluate_layout with prompt={PROMPT}, output={OUTPUT}."),
            answer_format={"hw": hws[0], "precision": precs[0], "tp": 2, "pp": 1, "expert_parallel": False, "batch": 64},
            truth=dse_truth(model, sla, hws, precs)))
    for model, hw, tp, prec, phase, batch, tokens in [
            ("llama-3.1-8b", "h100", 1, "bf16", "decode", 64, 4096), ("llama-3.1-70b", "h100", 8, "bf16", "decode", 1, 2048),
            ("mixtral-8x7b", "h100", 4, "bf16", "decode", 16, 1024), ("llama-3.1-8b", "h100", 1, "bf16", "prefill", 1, 8192),
            ("deepseek-v3", "b200", 8, "fp8", "decode", 32, 4096), ("qwen2.5-1.5b", "t4", 1, "bf16", "decode", 1, 512)]:
        d = GPUDeployment(model, hw, tp=tp, precision=prec)
        shape = StepShape.prefill_only(batch, tokens) if phase == "prefill" else StepShape.decode(batch, tokens)
        bn = d.step(shape).bottleneck()
        ctx = "prompt" if phase == "prefill" else "context"
        tasks.append(Task(
            id=f"bottleneck-{model}-{hw}-tp{tp}-{phase}-b{batch}", kind="bottleneck",
            params={"model": model, "hw": hw, "tp": tp, "precision": prec, "phase": phase, "batch": batch, "tokens": tokens},
            prompt=(f"For {model} on {tp}x {hw} (TP{tp}, {prec}), one {phase} step with batch {batch} and {ctx} "
                    f"length {tokens}: which operator takes the most time, and is the step dominated by memory, "
                    "compute, comm or overhead?"),
            answer_format={"op": "gate_up", "bound": "memory"},
            truth={"op": bn["top_ops"][0]["name"], "bound": bn["dominant_bound"]}))
    for model, hw, prec, users, ctx in [("llama-3.1-70b", "h100", "bf16", 64, 8192), ("llama-3.1-70b", "h100", "fp8", 256, 4096),
                                        ("llama-3.1-405b", "b200", "fp8", 32, 32768), ("mixtral-8x7b", "a100", "bf16", 128, 4096)]:
        tasks.append(Task(
            id=f"sizing-{model}-{hw}-{prec}-{users}x{ctx}", kind="sizing",
            params={"model": model, "hw": hw, "precision": prec, "users": users, "ctx": ctx},
            prompt=(f"What is the smallest number of {hw} GPUs (TP x PP, TP in 1,2,4,8 and PP in 1,2,4) that can serve "
                    f"{model} in {prec} with KV-cache room for {users} concurrent requests of {ctx} tokens each?"),
            answer_format={"devices": 4, "tp": 4, "pp": 1},
            truth=sizing_truth(model, hw, prec, users, ctx)))
    for model, hw, prec in [("llama-3.1-8b", "h100", "bf16"), ("llama-3.1-8b", "b200", "fp8"),
                            ("qwen2.5-1.5b", "a100", "bf16"), ("llama-3.1-70b", "h200", "fp8")]:
        tasks.append(Task(
            id=f"crossover-{model}-{hw}-{prec}", kind="crossover", params={"model": model, "hw": hw, "precision": prec},
            prompt=(f"On one {hw} in {prec}, at decode context 1024, what is the smallest batch size at which "
                    f"{model}'s gate_up (FFN up-projection) operator becomes compute-bound? (TP1, PP1; only "
                    + ("weights must fit" if model != "llama-3.1-70b" else "assume it fits as modeled") + ")"),
            answer_format={"batch": 128},
            truth=crossover_truth(model, hw, prec)))
    for model, batch, ctx, a, b in [
            ("llama-3.1-70b", 1, 2048, {"hw": "h100", "tp": 8, "precision": "bf16"}, {"hw": "b200", "tp": 2, "precision": "fp8"}),
            ("llama-3.1-8b", 32, 4096, {"hw": "a100", "tp": 1, "precision": "bf16"}, {"hw": "h100", "tp": 2, "precision": "bf16"}),
            ("mixtral-8x7b", 8, 2048, {"hw": "h100", "tp": 2, "precision": "fp8"}, {"hw": "h200", "tp": 4, "precision": "bf16"})]:
        ua, ub = _tok_s_user(a, model, batch, ctx), _tok_s_user(b, model, batch, ctx)
        tasks.append(Task(
            id=f"compare-{model}-b{batch}", kind="compare", params={"model": model, "batch": batch, "ctx": ctx, "a": a, "b": b},
            prompt=(f"{model}, decode batch {batch}, context {ctx}. Layout A: {a['tp']}x {a['hw']} TP{a['tp']} "
                    f"{a['precision']}. Layout B: {b['tp']}x {b['hw']} TP{b['tp']} {b['precision']}. Which gives more "
                    "tokens/s per user, and by what ratio (faster / slower)?"),
            answer_format={"faster": "a", "ratio": 1.5},
            truth={"faster": "a" if ua >= ub else "b", "ratio": max(ua, ub) / min(ua, ub)}))
    return tasks
