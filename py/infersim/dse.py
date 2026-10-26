"""Design space exploration: sweep layouts and batch sizes, keep the Pareto frontier.

Each design point is scored the way inference teams compare systems: interactivity
(output tokens per second seen by one user, ``1 / TPOT``) against throughput per
device. Steady state is computed analytically: with ``users`` concurrent requests,
generating ``output`` tokens for all of them takes ``output`` decode steps plus one
prefill per request (prefills interleave with decode).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .models import ModelConfig
from .models import preset as get_model
from .ops import StepShape
from .system import AttentionFFNSplit, Disaggregated, GPUDeployment, LPUDeployment


@dataclass(frozen=True)
class DesignPoint:
    label: str
    family: str  # gpu | lpu | disaggregated | afd
    devices: int
    batch: int
    users: int
    tpot_ms: float
    ttft_ms: float
    tok_s_user: float
    tok_s_device: float
    dominant_bound: str
    dominant_category: str

    def as_dict(self) -> dict:
        return asdict(self)


def _family(dep) -> str:
    return {GPUDeployment: "gpu", LPUDeployment: "lpu", Disaggregated: "disaggregated",
            AttentionFFNSplit: "afd"}[type(dep)]


def evaluate(dep, batch: int, prompt: int, output: int) -> DesignPoint | None:
    """Score one deployment at one per-stage decode batch size, or None if it does not fit."""
    ctx = prompt + output // 2
    per_user_kv = prompt + output
    if isinstance(dep, Disaggregated):
        dec = dep.decode.step(StepShape.decode(batch, ctx))
        users_max = dep.decode.kv_capacity_tokens() // per_user_kv
        conc = min(dec.concurrency, users_max // batch) if batch else 0
        if conc < 1:
            return None
        users = batch * conc
        pre = dep.prefill.step(StepShape.prefill_only(1, prompt))
        prefill_req_s = pre.concurrency / pre.latency_s
        decode_tok_s = users / dec.latency_s
        total_tok_s = min(decode_tok_s, prefill_req_s * output)
        ttft = pre.latency_s + dep.kv_transfer_s(prompt)
        bn = dec.bottleneck()
        return DesignPoint(dep.label, _family(dep), dep.devices, batch, users, dec.latency_s * 1e3,
                           ttft * 1e3, 1 / dec.latency_s, total_tok_s / dep.devices,
                           bn["dominant_bound"], bn["dominant_category"])
    if not dep.fits():
        return None
    users_max = dep.kv_capacity_tokens() // per_user_kv
    dec = dep.step(StepShape.decode(batch, ctx))
    conc = min(dec.concurrency, users_max // batch) if batch else 0
    if conc < 1:
        return None
    users = batch * conc
    pre = dep.step(StepShape.prefill_only(1, prompt))
    cycle = output * dec.latency_s + batch * pre.latency_s
    total_tok_s = users * output / cycle
    bn = dec.bottleneck()
    return DesignPoint(dep.label, _family(dep), dep.devices, batch, users, dec.latency_s * 1e3,
                       pre.latency_s * 1e3, 1 / dec.latency_s, total_tok_s / dep.devices,
                       bn["dominant_bound"], bn["dominant_category"])


def pareto(points: list[DesignPoint]) -> list[DesignPoint]:
    """Points not dominated on (tok_s_user, tok_s_device), both maximized; sorted by interactivity."""
    front = []
    for p in points:
        dominated = any(
            q.tok_s_user >= p.tok_s_user and q.tok_s_device >= p.tok_s_device
            and (q.tok_s_user > p.tok_s_user or q.tok_s_device > p.tok_s_device)
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: p.tok_s_user)


def best_under_sla(points: list[DesignPoint], min_tok_s_user: float,
                   max_ttft_ms: float | None = None) -> DesignPoint | None:
    ok = [p for p in points if p.tok_s_user >= min_tok_s_user
          and (max_ttft_ms is None or p.ttft_ms <= max_ttft_ms)]
    return max(ok, key=lambda p: p.tok_s_device) if ok else None


BATCHES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def candidate_deployments(model: ModelConfig | str, gpus=("h100", "b200"), lpus=("groq3",),
                          precisions=("bf16", "fp8"), tps=(1, 2, 4, 8), pps=(1, 2, 4),
                          include_heterogeneous: bool = True) -> list:
    m = get_model(model) if isinstance(model, str) else model
    deps: list = []
    for hw in gpus:
        for prec in precisions:
            for tp in tps:
                for pp in pps:
                    for ep in ((False, True) if m.is_moe and tp > 1 else (False,)):
                        d = GPUDeployment(m, hw, tp=tp, pp=pp, precision=prec, expert_parallel=ep)
                        if d.fits():
                            deps.append(d)
    for hw in lpus:
        lp = LPUDeployment(m, hw, precision="fp8", kv_reserve_tokens=0)
        # give the LPU room for KV cache: 25% more chips than the weights need
        deps.append(LPUDeployment(m, hw, precision="fp8", chips=int(lp.devices * 1.25) + 1))
    if include_heterogeneous and gpus and lpus:
        gpu = gpus[-1]
        lpu_dec = deps[-1] if lpus else None
        pre = GPUDeployment(m, gpu, tp=8, precision="fp8")
        if pre.fits() and lpu_dec is not None:
            deps.append(Disaggregated(pre, lpu_dec))
        afd = AttentionFFNSplit(m, gpu, lpus[0], gpu_tp=8, precision="fp8")
        if afd.fits():
            deps.append(afd)
    return deps


def sweep(deps: list, prompt: int, output: int, batches=BATCHES) -> list[DesignPoint]:
    pts = []
    for d in deps:
        for b in batches:
            p = evaluate(d, b, prompt, output)
            if p is not None:
                pts.append(p)
    return pts
