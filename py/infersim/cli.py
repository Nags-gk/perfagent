"""Command line: ``infersim {list,step,roofline,dse,serve,calibrate}``."""

from __future__ import annotations

import argparse
import json
import sys

from . import calibrate as cal
from .dse import best_under_sla, candidate_deployments, pareto, sweep
from .hardware import CATALOG
from .hardware import get as get_hw
from .models import PRESET_FILES
from .ops import StepShape
from .roofline import roofline_point
from .serving import poisson_workload, simulate
from .system import GPUDeployment, LPUDeployment


def _deployment(a):
    hw = get_hw(a.hw)
    if hw.kind == "lpu":
        return LPUDeployment(a.model, hw, precision=a.precision)
    return GPUDeployment(a.model, hw, tp=a.tp, pp=a.pp, precision=a.precision, expert_parallel=a.ep)


def _shape(a) -> StepShape:
    return StepShape.prefill_only(a.batch, a.tokens) if a.phase == "prefill" else StepShape.decode(a.batch, a.tokens)


def _add_dep_args(p):
    p.add_argument("--model", default="llama-3.1-8b", choices=sorted(PRESET_FILES))
    p.add_argument("--hw", default="h100", choices=sorted(CATALOG))
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--pp", type=int, default=1)
    p.add_argument("--ep", action="store_true", help="expert parallelism for MoE layers")
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp8", "fp4"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="infersim", description="LLM inference performance model")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="models and hardware")
    for name in ("step", "roofline"):
        p = sub.add_parser(name)
        _add_dep_args(p)
        p.add_argument("--phase", default="decode", choices=["prefill", "decode"])
        p.add_argument("--batch", type=int, default=1)
        p.add_argument("--tokens", type=int, default=2048, help="prompt length or decode context")
    p = sub.add_parser("dse", help="sweep layouts; print Pareto frontier")
    p.add_argument("--model", default="llama-3.1-70b", choices=sorted(PRESET_FILES))
    p.add_argument("--prompt", type=int, default=2048)
    p.add_argument("--output", type=int, default=256)
    p.add_argument("--sla", type=float, default=50.0, help="min tokens/s per user")
    p = sub.add_parser("serve", help="simulate serving a Poisson workload")
    _add_dep_args(p)
    p.add_argument("--rate", type=float, default=5.0)
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--prompt", type=int, default=1024)
    p.add_argument("--output", type=int, default=256)
    p = sub.add_parser("calibrate", help="fit efficiencies to measurements (a steps.csv, or a folder from measure.py)")
    p.add_argument("path")
    p.add_argument("--hw", default="t4", choices=sorted(CATALOG))
    p.add_argument("--out", help="write the full report as JSON here")
    for p in sub.choices.values():
        p.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        out = {"models": sorted(PRESET_FILES), "hardware": {k: v.name for k, v in CATALOG.items()}}
    elif a.cmd == "step":
        r = _deployment(a).step(_shape(a))
        out = r.bottleneck()
    elif a.cmd == "roofline":
        dep = _deployment(a)
        r = dep.step(_shape(a))
        out = [roofline_point(t.op, dep.hw) for t in r.timings if t.op.flops]
    elif a.cmd == "dse":
        pts = sweep(candidate_deployments(a.model), a.prompt, a.output)
        front = pareto(pts)
        best = best_under_sla(pts, a.sla)
        out = {"pareto": [p.as_dict() for p in front], "best_under_sla": best.as_dict() if best else None}
    elif a.cmd == "serve":
        reqs = poisson_workload(a.rate, a.requests, a.prompt, a.output, jitter=0.2)
        out = simulate(_deployment(a), reqs).summary()
    else:
        out = cal.report_from_path(a.path, get_hw(a.hw))
        if a.out:
            with open(a.out, "w") as f:
                json.dump(out, f)
        out = {k: v for k, v in out.items() if k not in ("rows", "gemm", "kernels")}
    json.dump(out, sys.stdout, indent=None if a.json else 2, default=str)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
