"""Deployments: how a model is laid out on hardware, and what one step costs.

Four layouts are modeled:

* ``GPUDeployment``: tensor, pipeline and (for MoE) expert parallelism on GPUs.
* ``LPUDeployment``: all weights resident in on-chip SRAM across many LPU chips,
  executed as a layer pipeline across chips (Groq style).
* ``Disaggregated``: prefill on one pool, decode on another, KV cache shipped between them.
* ``AttentionFFNSplit``: attention and KV cache on GPUs, FFN weights on LPUs, activations
  ping-ponged every layer (a heterogeneous decode layout; speculative, labeled as such).

Every deployment answers the same questions: how much KV cache fits, and how long a
forward pass over a ``StepShape`` takes (``latency_s``). ``concurrency`` is how many
independent batches can be in flight at once (pipeline stages), so steady-state
throughput is ``batch * concurrency / latency_s``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .hardware import HardwareSpec
from .hardware import get as get_hw
from .models import ModelConfig
from .models import preset as get_model
from .ops import PRECISIONS, Op, Precision, StepShape, step_ops
from .roofline import OpTiming, allreduce_time, p2p_time, time_op
from .stack import STACKS, Stack


@dataclass
class StepResult:
    latency_s: float
    timings: list[OpTiming]
    devices: int
    concurrency: int = 1
    extra_s: dict[str, float] = field(default_factory=dict)  # transfers not tied to one op

    def by(self, key: str) -> dict[str, float]:
        """Seconds grouped by op ``category`` or by roofline ``bound``."""
        out: dict[str, float] = {}
        for t in self.timings:
            k = getattr(t.op, key) if key == "category" else t.bound
            out[k] = out.get(k, 0.0) + t.total_s
        for k, v in self.extra_s.items():
            name = k if key == "category" else ("overhead" if k == "overlap" else "comm")
            out[name] = out.get(name, 0.0) + v
        return out

    def bottleneck(self) -> dict:
        """Where the time goes: dominant category, dominant bound, and the top ops."""
        cat = self.by("category")
        bnd = self.by("bound")
        top = sorted(self.timings, key=lambda t: t.total_s, reverse=True)[:5]
        return {
            "latency_ms": self.latency_s * 1e3,
            "category_share": {k: v / self.latency_s for k, v in cat.items()},
            "bound_share": {k: v / self.latency_s for k, v in bnd.items()},
            "dominant_category": max(cat, key=cat.get),
            "dominant_bound": max(bnd, key=bnd.get),
            "top_ops": [{"name": t.op.name, "bound": t.bound, "ms": t.total_s * 1e3} for t in top],
        }


def _prec(p: str | Precision) -> Precision:
    return PRECISIONS[p] if isinstance(p, str) else p


@dataclass
class GPUDeployment:
    model: ModelConfig
    hw: HardwareSpec
    tp: int = 1
    pp: int = 1
    precision: str | Precision = "bf16"
    expert_parallel: bool = False
    stack: str | Stack = "fused"

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            self.model = get_model(self.model)
        if isinstance(self.hw, str):
            self.hw = get_hw(self.hw)
        if isinstance(self.stack, str):
            self.stack = STACKS[self.stack]
        self.prec = _prec(self.precision)
        if self.model.layers % self.pp:
            # uneven stages are allowed; the slowest stage sets the interval
            pass

    @property
    def devices(self) -> int:
        return self.tp * self.pp

    @property
    def label(self) -> str:
        ep = " EP" if self.expert_parallel and self.model.is_moe else ""
        return f"{self.devices}x {self.hw.name} TP{self.tp} PP{self.pp}{ep} {self.prec.weights}"

    def weight_bytes_per_device(self) -> float:
        return self.model.param_count() * self.prec.wb / self.devices

    def kv_bytes_per_token_per_device(self) -> float:
        m = self.model
        elems = (m.kv_lora_rank + m.qk_rope_dim) if m.is_mla else 2 * max(1.0, m.kv_heads / self.tp) * m.head_dim
        return math.ceil(m.layers / self.pp) * elems * self.prec.kvb

    def kv_capacity_tokens(self) -> int:
        free = self.hw.mem_capacity * self.hw.usable_memory - self.weight_bytes_per_device()
        return max(0, int(free / self.kv_bytes_per_token_per_device()))

    def fits(self) -> bool:
        return self.kv_capacity_tokens() > 0

    def step(self, step: StepShape) -> StepResult:
        ops = step_ops(self.model, step, self.tp, self.prec, self.hw.tile_m, self.hw.tile_n,
                       self.expert_parallel, self.stack)
        timings = [time_op(o, self.hw, self.stack.kernel_multiplier) for o in ops]
        total = sum(t.total_s for t in timings)
        extra = {}
        if self.stack.async_launch:
            # host enqueues kernels while the GPU runs earlier ones: the slower side sets the pace
            host = sum(t.overhead_s * t.op.count for t in timings)
            device = total - host
            extra["overlap"] = max(host, device) - (host + device)
        if self.pp > 1:
            hop = p2p_time(step.tokens * self.model.hidden * self.prec.ab, self.hw.link_bw, self.hw.link_latency)
            extra["comm"] = (self.pp - 1) * hop
        return StepResult(total + sum(extra.values()), timings, self.devices, self.pp, extra)


@dataclass
class LPUDeployment:
    model: ModelConfig
    hw: HardwareSpec
    precision: str | Precision = "fp8"
    chips: int | None = None  # None: smallest count that holds the weights plus ``kv_reserve_tokens``
    kv_reserve_tokens: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            self.model = get_model(self.model)
        if isinstance(self.hw, str):
            self.hw = get_hw(self.hw)
        self.prec = _prec(self.precision)
        if self.chips is None:
            need = self.weight_bytes() + self.kv_reserve_tokens * self.model.kv_bytes_per_token(self.prec.kvb)
            self.chips = max(1, math.ceil(need / self.chip_bytes()))

    @property
    def devices(self) -> int:
        return int(self.chips)

    @property
    def label(self) -> str:
        return f"{self.devices}x {self.hw.name} {self.prec.weights}"

    def chip_bytes(self) -> float:
        return self.hw.mem_capacity * self.hw.usable_memory

    def weight_bytes(self) -> float:
        return self.model.param_count() * self.prec.wb

    def kv_capacity_tokens(self) -> int:
        free = self.devices * self.chip_bytes() - self.weight_bytes()
        return max(0, int(free / self.model.kv_bytes_per_token(self.prec.kvb)))

    def fits(self) -> bool:
        return self.kv_capacity_tokens() >= 0 and self.devices * self.chip_bytes() >= self.weight_bytes()

    def _time_on_chips(self, op: Op) -> OpTiming:
        """An op whose weights span several chips runs split across them, then all-reduces."""
        if op.kind in ("allreduce", "alltoall"):
            return time_op(op, self.hw)
        span = max(1.0, op.weight_bytes / self.chip_bytes()) if op.weight_bytes else 1.0
        t = time_op(op, self.hw)
        comp, mem = t.compute_s / span, t.memory_s / span
        comm = 0.0
        if span > 1 and op.kind == "gemm":
            m = op.meta.get("m", 0.0)
            comm = allreduce_time(m * self.model.hidden * self.prec.ab, math.ceil(span),
                                  self.hw.link_bw, self.hw.link_latency)
        body = max(comp, mem)
        bound = "compute" if comp >= mem else "memory"
        if comm > body:
            bound = "comm"
        return OpTiming(op, comp, mem, comm, t.overhead_s, body + comm + t.overhead_s, bound,
                        ideal=t.ideal / span)

    def step(self, step: StepShape) -> StepResult:
        ops = step_ops(self.model, step, 1, self.prec, self.hw.tile_m, self.hw.tile_n)
        timings = [self._time_on_chips(o) for o in ops]
        hop = p2p_time(step.tokens * self.model.hidden * self.prec.ab, self.hw.link_bw, self.hw.link_latency)
        extra = {"comm": (self.devices - 1) * hop}
        total = sum(t.total_s for t in timings) + extra["comm"]
        # Pipeline stages are layers (chips holding one layer work on it together), so at most
        # min(chips, layers) independent batches can be in flight.
        return StepResult(total, timings, self.devices, min(self.devices, self.model.layers), extra)


@dataclass
class Disaggregated:
    """Prefill pool and decode pool; the prompt's KV cache moves between them."""

    prefill: GPUDeployment | LPUDeployment
    decode: GPUDeployment | LPUDeployment
    kv_link_bw: float = 50e9  # 400 Gb/s NIC
    kv_link_latency: float = 10e-6

    @property
    def model(self) -> ModelConfig:
        return self.decode.model

    @property
    def devices(self) -> int:
        return self.prefill.devices + self.decode.devices

    @property
    def label(self) -> str:
        return f"prefill[{self.prefill.label}] + decode[{self.decode.label}]"

    def kv_transfer_s(self, prompt_tokens: int) -> float:
        nbytes = prompt_tokens * self.model.kv_bytes_per_token(self.decode.prec.kvb)
        return p2p_time(nbytes, self.kv_link_bw, self.kv_link_latency)

    def kv_capacity_tokens(self) -> int:
        return self.decode.kv_capacity_tokens()

    def fits(self) -> bool:
        return self.prefill.fits() and self.decode.fits()


@dataclass
class AttentionFFNSplit:
    """Attention (and KV cache) on GPUs, FFN weights resident on LPU SRAM.

    Activations cross the GPU-LPU link twice per layer. Two micro-batches ping-pong so
    one runs attention while the other runs FFN (concurrency 2).
    """

    model: ModelConfig
    gpu: HardwareSpec
    lpu: HardwareSpec
    gpu_tp: int = 8
    precision: str | Precision = "fp8"
    link_bw: float = 100e9
    link_latency: float = 5e-6
    assumptions: tuple[str, ...] = (
        "GPU-LPU link: 100 GB/s per direction, 5 us per transfer (no published figure)",
        "Two micro-batches overlap perfectly (attention of one hides FFN of the other)",
    )

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            self.model = get_model(self.model)
        if isinstance(self.gpu, str):
            self.gpu = get_hw(self.gpu)
        if isinstance(self.lpu, str):
            self.lpu = get_hw(self.lpu)
        self.prec = _prec(self.precision)
        m = self.model
        ffn_params = (m.dense_layers * m.mlp_params(m.intermediate) + m.moe_layers * m.moe_layer_params(False))
        self.ffn_bytes = ffn_params * self.prec.wb
        self.lpu_chips = max(1, math.ceil(self.ffn_bytes / (self.lpu.mem_capacity * self.lpu.usable_memory)))
        self._lpu = LPUDeployment(m, self.lpu, self.prec, chips=self.lpu_chips)

    @property
    def devices(self) -> int:
        return self.gpu_tp + self.lpu_chips

    @property
    def label(self) -> str:
        return f"{self.gpu_tp}x {self.gpu.name} attention + {self.lpu_chips}x {self.lpu.name} FFN"

    def gpu_weight_bytes_per_device(self) -> float:
        return (self.model.param_count() * self.prec.wb - self.ffn_bytes) / self.gpu_tp

    def kv_capacity_tokens(self) -> int:
        m = self.model
        elems = (m.kv_lora_rank + m.qk_rope_dim) if m.is_mla else 2 * max(1.0, m.kv_heads / self.gpu_tp) * m.head_dim
        free = self.gpu.mem_capacity * self.gpu.usable_memory - self.gpu_weight_bytes_per_device()
        return max(0, int(free / (m.layers * elems * self.prec.kvb)))

    def fits(self) -> bool:
        return self.kv_capacity_tokens() > 0

    def step(self, step: StepShape) -> StepResult:
        gpu_ops = [o for o in step_ops(self.model, step, self.gpu_tp, self.prec, self.gpu.tile_m,
                                       self.gpu.tile_n) if o.category != "ffn" and o.name != "tp_allreduce_ffn"]
        lpu_ops = [o for o in step_ops(self.model, step, 1, self.prec, 1, 1) if o.category == "ffn"]
        timings = [time_op(o, self.gpu) for o in gpu_ops] + [self._lpu._time_on_chips(o) for o in lpu_ops]
        hop = p2p_time(step.tokens * self.model.hidden * self.prec.ab, self.link_bw, self.link_latency)
        extra = {"comm": 2 * self.model.layers * hop}
        total = sum(t.total_s for t in timings) + extra["comm"]
        return StepResult(total, timings, self.devices, 2, extra)
