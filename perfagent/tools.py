"""Simulation tools the agent (and any MCP client) can call.

Pure Python on top of InferSim, so the same tools run in CI, behind the MCP server, and in
the browser through Pyodide. A ``ToolBox`` enforces a simulation budget: ``simulate_step``,
``roofline_ops`` and ``evaluate_layout`` each cost one simulation; catalog lookups and
``check_fit`` (memory arithmetic, no timing) are free. Repeated calls hit a cache and are free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from infersim.dse import evaluate
from infersim.hardware import CATALOG
from infersim.models import all_presets
from infersim.ops import StepShape
from infersim.system import GPUDeployment, LPUDeployment

HW_GPU = ("t4", "a100", "h100", "h200", "b200")
PRECISIONS = ("bf16", "fp8", "fp4")
PHASES = ("prefill", "decode")


class ToolError(ValueError):
    """Bad arguments or budget exhausted; the message goes back to the model."""


def _str(v, name, choices=None):
    if not isinstance(v, str):
        raise ToolError(f"{name} must be a string")
    v = v.lower().strip()
    if choices and v not in choices:
        raise ToolError(f"{name}={v!r} is not one of {sorted(choices)}")
    return v


def _int(v, name, lo=1, hi=1 << 20, pow2=False):
    try:
        v = int(v)
    except (TypeError, ValueError) as e:
        raise ToolError(f"{name} must be an integer") from e
    if not lo <= v <= hi:
        raise ToolError(f"{name} must be in [{lo}, {hi}]")
    if pow2 and v & (v - 1):
        raise ToolError(f"{name} must be a power of two")
    return v


def _deployment(model, hw, tp=1, pp=1, precision="bf16", expert_parallel=False):
    models = all_presets()
    model = _str(model, "model", models)
    hw = _str(hw, "hw", CATALOG)
    precision = _str(precision, "precision", PRECISIONS)
    if CATALOG[hw].kind == "lpu":
        return LPUDeployment(models[model], CATALOG[hw], precision=precision)
    tp = _int(tp, "tp", 1, 8, pow2=True)
    pp = _int(pp, "pp", 1, 16, pow2=True)
    return GPUDeployment(models[model], CATALOG[hw], tp=tp, pp=pp, precision=precision,
                         expert_parallel=bool(expert_parallel))


def _r(x, n=4):
    return float(f"{x:.{n}g}")


@dataclass
class ToolBox:
    budget: int = 30  # simulations allowed
    used: int = 0
    calls: int = 0
    cache: dict = field(default_factory=dict)
    log: list = field(default_factory=list)

    # ---- dispatch ------------------------------------------------------
    def call(self, name: str, args: dict) -> dict:
        self.calls += 1
        fn = getattr(self, "t_" + name, None)
        if fn is None:
            raise ToolError(f"unknown tool {name!r}")
        key = name + json.dumps(args, sort_keys=True, default=str)
        if key in self.cache:
            out = dict(self.cache[key], cached=True)
            self.log.append({"tool": name, "args": args, "cost": 0, "cached": True})
            return out
        cost = TOOL_COST.get(name, 0)
        if cost and self.used + cost > self.budget:
            raise ToolError(f"simulation budget exhausted ({self.used}/{self.budget}); submit your best answer now")
        out = fn(**args)
        self.used += cost
        self.cache[key] = out
        self.log.append({"tool": name, "args": args, "cost": cost, "cached": False})
        return dict(out, budget_left=self.budget - self.used)

    # ---- tools -------------------------------------------------------------
    def t_list_catalog(self, section: str = "all") -> dict:
        hw = {k: {"name": v.name, "kind": v.kind, "memory_gb": _r(v.mem_capacity / 1e9),
                  "mem_bw_tb_s": _r(v.mem_bw / 1e12),
                  "peak_tflops": {d: _r(f / 1e12) for d, f in v.peak_flops.items()}}
              for k, v in CATALOG.items()}
        models = {k: {"params_b": _r(m.param_count() / 1e9), "active_params_b": _r(m.param_count(True) / 1e9),
                      "layers": m.layers, "moe": m.is_moe, "mla": m.is_mla,
                      "kv_kib_per_token_bf16": _r(m.kv_bytes_per_token(2) / 1024)}
                  for k, m in all_presets().items()}
        section = _str(section, "section", ("all", "models", "hardware"))
        out = {"models": models, "hardware": hw, "precisions": list(PRECISIONS),
               "parallelism": "tp in 1,2,4,8; pp in 1,2,4 (powers of two); expert_parallel for MoE"}
        if section != "all":
            out = {section: out[section]}
        return out

    def t_check_fit(self, model, hw, tp=1, pp=1, precision="bf16", expert_parallel=False) -> dict:
        d = _deployment(model, hw, tp, pp, precision, expert_parallel)
        w = d.weight_bytes_per_device() if hasattr(d, "weight_bytes_per_device") else d.weight_bytes() / d.devices
        return {"layout": d.label, "devices": d.devices, "fits": bool(d.fits()),
                "weight_gb_per_device": _r(w / 1e9), "kv_capacity_tokens": d.kv_capacity_tokens()}

    def t_simulate_step(self, model, hw, phase="decode", batch=1, tokens=2048, tp=1, pp=1, precision="bf16",
                        expert_parallel=False) -> dict:
        d = _deployment(model, hw, tp, pp, precision, expert_parallel)
        if not d.fits():
            return {"layout": d.label, "fits": False, "error": "weights do not fit in memory"}
        phase = _str(phase, "phase", PHASES)
        batch, tokens = _int(batch, "batch", 1, 4096), _int(tokens, "tokens", 1, 131072)
        shape = StepShape.prefill_only(batch, tokens) if phase == "prefill" else StepShape.decode(batch, tokens)
        r = d.step(shape)
        b = r.bottleneck()
        out = {"layout": d.label, "fits": True, "latency_ms": _r(b["latency_ms"]),
               "dominant_bound": b["dominant_bound"], "dominant_category": b["dominant_category"],
               "bound_share": {k: _r(v, 3) for k, v in b["bound_share"].items()},
               "category_share": {k: _r(v, 3) for k, v in b["category_share"].items()},
               "top_ops": [{"name": o["name"], "bound": o["bound"], "ms": _r(o["ms"])} for o in b["top_ops"]]}
        if phase == "decode":
            out["tok_s_per_user"] = _r(1 / r.latency_s)
            out["tok_s_total"] = _r(batch * r.concurrency / r.latency_s)
        return out

    def t_roofline_ops(self, model, hw, phase="decode", batch=1, tokens=2048, tp=1, pp=1, precision="bf16",
                       expert_parallel=False) -> dict:
        d = _deployment(model, hw, tp, pp, precision, expert_parallel)
        if not d.fits():
            return {"layout": d.label, "fits": False, "error": "weights do not fit in memory"}
        phase = _str(phase, "phase", PHASES)
        batch, tokens = _int(batch, "batch", 1, 4096), _int(tokens, "tokens", 1, 131072)
        shape = StepShape.prefill_only(batch, tokens) if phase == "prefill" else StepShape.decode(batch, tokens)
        r = d.step(shape)
        ops = [{"name": t.op.name, "bound": t.bound, "ms": _r(t.total_s * 1e3),
                "intensity_flop_per_byte": _r(t.op.intensity) if t.op.bytes else None}
               for t in r.timings]
        return {"layout": d.label, "ridge_flop_per_byte": _r(d.hw.ridge_point(d.prec.gemm_dtype)), "ops": ops}

    def t_evaluate_layout(self, model, hw, tp=1, pp=1, precision="bf16", batch=1, prompt=2048, output=256,
                          expert_parallel=False) -> dict:
        d = _deployment(model, hw, tp, pp, precision, expert_parallel)
        batch = _int(batch, "batch", 1, 4096)
        p = evaluate(d, batch, _int(prompt, "prompt"), _int(output, "output"))
        if p is None:
            return {"layout": d.label, "batch": batch, "fits": False,
                    "error": "does not fit (weights, or KV cache for this batch)"}
        return {"layout": p.label, "batch": batch, "fits": True, "devices": p.devices,
                "tok_s_per_user": _r(p.tok_s_user), "tok_s_per_device": _r(p.tok_s_device),
                "tpot_ms": _r(p.tpot_ms), "ttft_ms": _r(p.ttft_ms), "dominant_bound": p.dominant_bound}


TOOL_COST = {"simulate_step": 1, "roofline_ops": 1, "evaluate_layout": 1}

_LAYOUT_PROPS = {
    "model": {"type": "string", "description": "model preset, e.g. llama-3.1-70b"},
    "hw": {"type": "string", "description": "hardware key, e.g. h100, b200, groq3"},
    "tp": {"type": "integer", "description": "tensor parallel size: 1, 2, 4 or 8"},
    "pp": {"type": "integer", "description": "pipeline parallel size: 1, 2 or 4"},
    "precision": {"type": "string", "enum": list(PRECISIONS)},
    "expert_parallel": {"type": "boolean", "description": "expert parallelism for MoE models"},
}
_STEP_PROPS = {
    **_LAYOUT_PROPS,
    "phase": {"type": "string", "enum": list(PHASES)},
    "batch": {"type": "integer", "description": "sequences in the step"},
    "tokens": {"type": "integer", "description": "prompt length (prefill) or context length (decode)"},
}

TOOL_SPECS = [
    {"name": "list_catalog", "description": "Models, hardware and options. Free.",
     "parameters": {"type": "object", "properties": {
         "section": {"type": "string", "enum": ["all", "models", "hardware"]}}}},
    {"name": "check_fit",
     "description": "Memory check for a layout: devices, whether weights fit, KV-cache capacity in tokens. "
                    "Free (no simulation).",
     "parameters": {"type": "object", "properties": _LAYOUT_PROPS, "required": ["model", "hw"]}},
    {"name": "simulate_step",
     "description": "Time one forward pass and break it down: latency, tokens/s, dominant bound and category, "
                    "slowest operators. Costs 1 simulation.",
     "parameters": {"type": "object", "properties": _STEP_PROPS, "required": ["model", "hw", "phase", "batch", "tokens"]}},
    {"name": "roofline_ops",
     "description": "Every operator of one step with its roofline bound, time and arithmetic intensity, plus the "
                    "hardware ridge point. Costs 1 simulation.",
     "parameters": {"type": "object", "properties": _STEP_PROPS, "required": ["model", "hw", "phase", "batch", "tokens"]}},
    {"name": "evaluate_layout",
     "description": "Steady-state serving metrics for a layout at a decode batch size: tokens/s per user "
                    "(interactivity), tokens/s per device (throughput), TTFT. Costs 1 simulation.",
     "parameters": {"type": "object", "properties": {
         **_LAYOUT_PROPS, "batch": {"type": "integer", "description": "decode batch per pipeline stage"},
         "prompt": {"type": "integer"}, "output": {"type": "integer"}}, "required": ["model", "hw", "batch"]}},
]


def openai_tools(extra: list[dict] | None = None) -> list[dict]:
    return [{"type": "function", "function": s} for s in TOOL_SPECS + (extra or [])]
