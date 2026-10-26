"""Fit the model to measured latencies, and report the error honestly.

A measurement is one timed forward pass (a prefill of ``batch`` prompts of ``tokens``
each, or a decode step of ``batch`` sequences at context ``tokens``).

Calibration has two parts, kept separate on purpose:

1. **Structure from profiling, not fitting.** The software stack's kernel multiplier is set
   from ``torch.profiler`` kernel counts (measured kernels / modeled kernels), and the
   attention implementation is chosen from what the GPU supports.
2. **Efficiencies by fitting.** ``eff_compute``, ``eff_memory``, the per-kernel launch cost
   and (for math attention) the number of passes over materialized tensors are fitted by
   minimizing squared log error with Nelder-Mead (pure Python), on a training split.

The report gives error on a held-out split, for three setups: the stack InferSim assumes
by default, the measured stack with default efficiencies, and the measured stack fitted.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass

from .hardware import HardwareSpec
from .models import preset as get_model
from .ops import PRECISIONS, StepShape, gemm, step_ops
from .roofline import time_op
from .stack import STACKS, Stack
from .system import GPUDeployment

# (owner, field, lower bound, upper bound, simplex step)
PARAMS = (
    ("hw", "eff_compute", 0.05, 1.0, 0.2),
    ("hw", "eff_memory", 0.05, 1.0, 0.2),
    ("hw", "launch_overhead", 0.0, 1e-4, 5e-6),
    ("stack", "math_passes", 0.5, 10.0, 1.0),
    ("stack", "score_passes", 0.5, 12.0, 1.0),
)


@dataclass(frozen=True)
class Measurement:
    model: str
    phase: str  # prefill | decode
    batch: int
    tokens: int  # prompt length (prefill) or context length (decode)
    measured_s: float
    precision: str = "bf16"

    def shape(self) -> StepShape:
        if self.phase == "prefill":
            return StepShape.prefill_only(self.batch, self.tokens)
        return StepShape.decode(self.batch, self.tokens)


@dataclass(frozen=True)
class Setup:
    hw: HardwareSpec
    stack: Stack = STACKS["fused"]

    def params(self) -> list[tuple]:
        return [p for p in PARAMS if p[0] == "hw" or self.stack.attention == "math"]

    def values(self) -> list[float]:
        return [getattr(self.hw if o == "hw" else self.stack, f) for o, f, *_ in self.params()]

    def with_values(self, x: list[float]) -> Setup:
        hw_kw, st_kw = {}, {}
        for (o, f, lo, hi, _), v in zip(self.params(), x):
            (hw_kw if o == "hw" else st_kw)[f] = min(max(v, lo), hi)
        return Setup(self.hw.with_overrides(**hw_kw), self.stack.with_overrides(**st_kw))

    def as_dict(self) -> dict:
        return {f: v for (_, f, *_), v in zip(self.params(), self.values())}


def load_csv(path: str) -> list[Measurement]:
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append(Measurement(row["model"], row["phase"], int(row["batch"]), int(row["tokens"]),
                                   float(row["measured_s"]), row.get("precision") or "bf16"))
    return out


def _setup(s: Setup | HardwareSpec) -> Setup:
    return s if isinstance(s, Setup) else Setup(s)


def predict(m: Measurement, setup: Setup | HardwareSpec) -> float:
    s = _setup(setup)
    dep = GPUDeployment(get_model(m.model), s.hw, tp=1, precision=m.precision, stack=s.stack)
    return dep.step(m.shape()).latency_s


def mape(ms: list[Measurement], setup: Setup | HardwareSpec) -> float:
    return 100.0 * sum(abs(predict(m, setup) - m.measured_s) / m.measured_s for m in ms) / len(ms)


def _loss(x: list[float], ms: list[Measurement], base: Setup, gemms: list | None = None) -> float:
    """Mean squared log error over steps, plus the same over isolated GEMMs when given.

    The GEMM term pins the efficiencies to what single kernels actually achieve, so the step
    fit cannot quietly push them to unphysical values to absorb work the model lacks.
    """
    s = base.with_values(x)
    loss = sum(math.log(predict(m, s) / m.measured_s) ** 2 for m in ms) / len(ms)
    if gemms:
        loss += sum(math.log(predict_gemm(g, s.hw) / g.measured_s) ** 2 for g in gemms) / len(gemms)
    return loss


def nelder_mead(f, x0: list[float], scale: list[float], iters: int = 600, tol: float = 1e-10) -> list[float]:
    n = len(x0)
    simplex = [list(x0)] + [[x0[j] + (scale[j] if j == i else 0.0) for j in range(n)] for i in range(n)]
    vals = [f(p) for p in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda i: vals[i])
        simplex = [simplex[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        centroid = [sum(p[j] for p in simplex[:-1]) / n for j in range(n)]
        worst = simplex[-1]
        refl = [centroid[j] + (centroid[j] - worst[j]) for j in range(n)]
        fr = f(refl)
        if fr < vals[0]:
            exp = [centroid[j] + 2 * (centroid[j] - worst[j]) for j in range(n)]
            fe = f(exp)
            simplex[-1], vals[-1] = (exp, fe) if fe < fr else (refl, fr)
        elif fr < vals[-2]:
            simplex[-1], vals[-1] = refl, fr
        else:
            contr = [centroid[j] + 0.5 * (worst[j] - centroid[j]) for j in range(n)]
            fc = f(contr)
            if fc < vals[-1]:
                simplex[-1], vals[-1] = contr, fc
            else:
                best = simplex[0]
                simplex = [best] + [[best[j] + 0.5 * (p[j] - best[j]) for j in range(n)] for p in simplex[1:]]
                vals = [vals[0]] + [f(p) for p in simplex[1:]]
    return simplex[min(range(n + 1), key=lambda i: vals[i])]


@dataclass
class CalibrationResult:
    fitted: dict[str, float]
    mape_before: float
    mape_after: float
    holdout_mape_before: float | None
    holdout_mape_after: float | None
    per_phase_after: dict[str, float]
    setup: Setup

    @property
    def hardware(self) -> HardwareSpec:
        return self.setup.hw


def fit(train: list[Measurement], setup: Setup | HardwareSpec,
        holdout: list[Measurement] | None = None, gemms: list | None = None) -> CalibrationResult:
    base = _setup(setup)
    x0 = base.values()
    scale = [p[4] for p in base.params()]
    tuned = base.with_values(nelder_mead(lambda x: _loss(x, train, base, gemms), x0, scale, iters=1500))
    phases = {}
    for ph in ("prefill", "decode"):
        sub = [m for m in (train + (holdout or [])) if m.phase == ph]
        if sub:
            phases[ph] = mape(sub, tuned)
    return CalibrationResult(
        fitted=tuned.as_dict(),
        mape_before=mape(train, base),
        mape_after=mape(train, tuned),
        holdout_mape_before=mape(holdout, base) if holdout else None,
        holdout_mape_after=mape(holdout, tuned) if holdout else None,
        per_phase_after=phases,
        setup=tuned,
    )


def split(ms: list[Measurement], every: int = 4) -> tuple[list[Measurement], list[Measurement]]:
    """Deterministic holdout: every ``every``-th measurement is held out."""
    train = [m for i, m in enumerate(ms) if i % every]
    hold = [m for i, m in enumerate(ms) if not i % every]
    return train, hold


# ---------------------------------------------------------------------------
# GEMM-level validation, kernel counts and the full report


@dataclass(frozen=True)
class GemmMeasurement:
    m: int
    n: int
    k: int
    measured_s: float
    dtype: str = "fp16"


def load_gemm_csv(path: str, dtype: str = "fp16") -> list[GemmMeasurement]:
    with open(path, newline="") as f:
        return [GemmMeasurement(int(r["m"]), int(r["n"]), int(r["k"]), float(r["measured_s"]), dtype)
                for r in csv.DictReader(f)]


def predict_gemm(g: GemmMeasurement, hw: HardwareSpec) -> float:
    """One isolated GEMM: host launch plus device time (nothing to overlap with)."""
    return time_op(gemm("g", "ffn", g.m, g.n, g.k, PRECISIONS[g.dtype], hw.tile_m, hw.tile_n, 1), hw).time_s


def gemm_mape(gs: list[GemmMeasurement], hw: HardwareSpec) -> float:
    return 100.0 * sum(abs(predict_gemm(g, hw) - g.measured_s) / g.measured_s for g in gs) / len(gs)


def model_kernels(m: Measurement) -> int:
    """Kernel launches the fused op list assumes for one step (all layers)."""
    return int(sum(o.kernels * o.count for o in step_ops(get_model(m.model), m.shape(), 1, PRECISIONS[m.precision])
                   if o.kind not in ("allreduce", "alltoall")))


def kernel_multiplier(steps: list[Measurement], kernels: dict) -> float | None:
    ratios = [kernels[(m.model, m.phase, m.batch, m.tokens)] / model_kernels(m) for m in steps
              if kernels.get((m.model, m.phase, m.batch, m.tokens), 0) > 0]
    return statistics.median(ratios) if ratios else None


def load_kernels_csv(path: str) -> dict:
    with open(path, newline="") as f:
        return {(r["model"], r["phase"], int(r["batch"]), int(r["tokens"])): int(r["cuda_kernels"])
                for r in csv.DictReader(f)}


def _rows(ms: list[Measurement], before: Setup, after: Setup, held: set[int]) -> list[dict]:
    out = []
    for i, m in enumerate(ms):
        pb, pa = predict(m, before), predict(m, after)
        out.append({"model": m.model, "phase": m.phase, "batch": m.batch, "tokens": m.tokens,
                    "measured_ms": m.measured_s * 1e3, "predicted_before_ms": pb * 1e3,
                    "predicted_after_ms": pa * 1e3, "error_after_pct": 100 * (pa - m.measured_s) / m.measured_s,
                    "holdout": i in held})
    return out


def report(steps: list[Measurement], hw: HardwareSpec, gemms: list[GemmMeasurement] | None = None,
           kernels: dict | None = None, env: dict | None = None, every: int = 4,
           stack: Stack | None = None) -> dict:
    """Fit on a training split and report everything the dashboard and README need.

    ``stack`` defaults to the eager PyTorch stack when kernel counts are available (with its
    kernel multiplier taken from them), else to the fused stack.
    """
    km = kernel_multiplier(steps, kernels) if kernels else None
    if stack is None:
        stack = STACKS["eager"].with_overrides(kernel_multiplier=km) if km else STACKS["fused"]
    train, hold = split(steps, every)
    default = Setup(hw, STACKS["fused"])
    base = Setup(hw, stack)
    r = fit(train, base, hold, gemms)
    held = {i for i in range(len(steps)) if not i % every}
    by_model = {}
    for name in sorted({m.model for m in steps}):
        for ph in ("prefill", "decode"):
            sub = [m for m in steps if m.model == name and m.phase == ph]
            if sub:
                by_model[f"{name} {ph}"] = {"default": mape(sub, default), "before": mape(sub, base),
                                            "after": mape(sub, r.setup), "n": len(sub)}
    out = {
        "hardware": hw.name,
        "stack": {"name": stack.name, "attention": stack.attention, "kernel_multiplier": stack.kernel_multiplier,
                  "async_launch": stack.async_launch},
        "env": env or {},
        "fitted": r.fitted,
        "at_bound": [f for (o, f, lo, hi, _) in base.params() if not lo < r.fitted[f] < hi],
        "defaults": base.as_dict(),
        "n_measurements": len(steps),
        "n_holdout": len(hold),
        "mape_holdout_default_stack": mape(hold, default) if hold else None,
        "mape_train_before": r.mape_before,
        "mape_train_after": r.mape_after,
        "mape_holdout_before": r.holdout_mape_before,
        "mape_holdout_after": r.holdout_mape_after,
        "by_model_phase": by_model,
        "rows": _rows(steps, base, r.setup, held),
    }
    if gemms:
        out["gemm"] = {
            "mape_before": gemm_mape(gemms, hw),
            "mape_after": gemm_mape(gemms, r.hardware),
            "peak_flops": hw.peak(gemms[0].dtype),
            "mem_bw": hw.mem_bw,
            "rows": [{"m": g.m, "n": g.n, "k": g.k, "measured_us": g.measured_s * 1e6,
                      "predicted_after_us": predict_gemm(g, r.hardware) * 1e6,
                      "tflops": 2 * g.m * g.n * g.k / g.measured_s / 1e12,
                      "intensity": 2 * g.m * g.n * g.k / ((g.m * g.k + g.k * g.n + g.m * g.n) * 2)} for g in gemms],
        }
    if kernels:
        out["kernels"] = [{"model": m.model, "phase": m.phase, "batch": m.batch, "tokens": m.tokens,
                           "measured": kernels[(m.model, m.phase, m.batch, m.tokens)], "modeled": model_kernels(m)}
                          for m in steps if kernels.get((m.model, m.phase, m.batch, m.tokens), 0) > 0]
    return out


def report_from_path(path: str, hw: HardwareSpec) -> dict:
    """``path`` is a steps CSV, or a folder written by ``calibration/measure.py``."""
    import json
    import os

    if os.path.isdir(path):
        steps = load_csv(os.path.join(path, "steps.csv"))
        g = os.path.join(path, "gemm.csv")
        k = os.path.join(path, "kernels.csv")
        e = os.path.join(path, "env.json")
        env = None
        if os.path.exists(e):
            with open(e) as f:
                env = json.load(f)
        return report(steps, hw, load_gemm_csv(g, steps[0].precision) if os.path.exists(g) else None,
                      load_kernels_csv(k) if os.path.exists(k) else None, env)
    return report(load_csv(path), hw)
