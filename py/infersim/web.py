"""JSON views shared by the dashboard (via Pyodide in the browser) and the benchmark script."""

from __future__ import annotations

import json

from .dse import best_under_sla, candidate_deployments, pareto, sweep
from .hardware import CATALOG
from .hardware import get as get_hw
from .models import PRESET_FILES, all_presets
from .models import preset as get_model
from .ops import StepShape
from .roofline import roofline_point
from .serving import poisson_workload, simulate
from .system import GPUDeployment, LPUDeployment

SLAS = (20, 50, 100, 200, 500)


def make_deployment(model: str, hw: str, tp: int = 1, pp: int = 1, precision: str = "bf16", ep: bool = False):
    h = get_hw(hw)
    if h.kind == "lpu":
        return LPUDeployment(model, h, precision="bf16" if precision == "bf16" else precision)
    return GPUDeployment(model, h, tp=tp, pp=pp, precision=precision, expert_parallel=ep)


def catalog_view() -> dict:
    hw = {k: {"name": v.name, "kind": v.kind, "peak_flops": v.peak_flops, "mem_bw": v.mem_bw,
              "mem_capacity": v.mem_capacity, "link_bw": v.link_bw, "sources": list(v.sources),
              "assumptions": list(v.assumptions)} for k, v in CATALOG.items()}
    models = {k: {"params_b": m.param_count() / 1e9, "active_b": m.param_count(True) / 1e9,
                  "layers": m.layers, "hidden": m.hidden, "heads": m.heads, "kv_heads": m.kv_heads,
                  "moe": m.is_moe, "mla": m.is_mla, "kv_kib_per_token_bf16": m.kv_bytes_per_token(2) / 1024}
              for k, m in all_presets().items()}
    return {"hardware": hw, "models": models}


def step_view(model: str, hw: str, tp: int = 1, pp: int = 1, precision: str = "bf16", phase: str = "decode",
              batch: int = 1, tokens: int = 2048, ep: bool = False) -> dict:
    dep = make_deployment(model, hw, tp, pp, precision, ep)
    h = dep.hw
    out = {"label": dep.label, "devices": dep.devices, "fits": dep.fits(),
           "kv_capacity_tokens": dep.kv_capacity_tokens()}
    if hasattr(dep, "weight_bytes_per_device"):
        out["weight_gb_per_device"] = dep.weight_bytes_per_device() / 1e9
    else:
        out["weight_gb_per_device"] = dep.weight_bytes() / dep.devices / 1e9
    if not dep.fits():
        return out
    shape = StepShape.prefill_only(batch, tokens) if phase == "prefill" else StepShape.decode(batch, tokens)
    r = dep.step(shape)
    bn = r.bottleneck()
    out.update(bn)
    out["concurrency"] = r.concurrency
    if phase == "decode":
        out["tok_s_user"] = 1 / r.latency_s
        out["tok_s_total"] = batch * r.concurrency / r.latency_s
    else:
        out["prefill_tok_s"] = batch * tokens / r.latency_s
    gemm_dtype = dep.prec.gemm_dtype
    pts = []
    for t in r.timings:
        if t.op.flops > 0 and t.op.bytes > 0:
            p = roofline_point(t.op, h)
            p["time_ms"] = t.total_s * 1e3
            pts.append(p)
    comm = [{"name": t.op.name, "time_ms": t.total_s * 1e3} for t in r.timings if t.bound == "comm"]
    out["roofline"] = {"peak": h.peak(gemm_dtype), "peak_attention": h.peak(dep.prec.act), "mem_bw": h.mem_bw,
                       "ridge": h.ridge_point(gemm_dtype), "dtype": gemm_dtype, "points": pts, "comm": comm}
    return out


def dse_view(model: str, prompt: int = 2048, output: int = 256) -> dict:
    pts = sweep(candidate_deployments(model), prompt, output)
    front = pareto(pts)
    fams = {}
    for f in sorted({p.family for p in pts}):
        fams[f] = [p.as_dict() for p in pareto([p for p in pts if p.family == f])]
    best = {str(s): (b.as_dict() if (b := best_under_sla(pts, s)) else None) for s in SLAS}
    return {"model": model, "prompt": prompt, "output": output, "n_points": len(pts),
            "points": [p.as_dict() for p in pts], "pareto": [p.as_dict() for p in front],
            "pareto_by_family": fams, "best_under_sla": best}


def serve_view(model: str, hw: str, tp: int = 1, precision: str = "bf16", rate: float = 5.0, requests: int = 200,
               prompt: int = 1024, output: int = 256, max_batch: int = 256) -> dict:
    dep = make_deployment(model, hw, tp, 1, precision)
    reqs = poisson_workload(rate, requests, prompt, output, jitter=0.2, seed=7)
    res = simulate(dep, reqs, max_batch=max_batch)
    s = res.summary()
    s["label"] = dep.label
    s["ttft_ms"] = [r.ttft * 1e3 for r in res.requests]
    s["tpot_ms"] = [r.tpot * 1e3 for r in res.requests]
    return s


def call(fn: str, kwargs_json: str) -> str:
    """Entry point for the browser: ``call("step_view", '{"model": ...}')`` returns JSON text."""
    funcs = {"step_view": step_view, "dse_view": dse_view, "serve_view": serve_view, "catalog_view": catalog_view}
    return json.dumps(funcs[fn](**json.loads(kwargs_json)))


MODELS = sorted(PRESET_FILES)
__all__ = ["MODELS", "call", "catalog_view", "dse_view", "get_model", "serve_view", "step_view"]
