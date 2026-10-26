"""Roofline timing for single ops, and the alpha-beta model for collectives."""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import HardwareSpec
from .ops import Op


@dataclass(frozen=True)
class OpTiming:
    op: Op
    compute_s: float  # padded FLOPs at achievable peak
    memory_s: float  # bytes at achievable bandwidth
    comm_s: float  # collective time
    overhead_s: float  # fixed launch / sync cost
    time_s: float  # per layer instance
    bound: str  # compute | memory | comm | overhead
    ideal: float = 0.0  # useful FLOPs at 100% peak vs bytes at 100% bandwidth, no overhead

    @property
    def total_s(self) -> float:
        return self.time_s * self.op.count

    @property
    def ideal_total_s(self) -> float:
        return self.ideal * self.op.count


def allreduce_time(nbytes: float, p: int, bw: float, latency: float) -> float:
    """All-reduce time as the faster of two algorithms, as NCCL / TensorRT-LLM pick per size.

    * ring: 2(p-1)/p of the payload crosses each link, over 2(p-1) latency-bound steps
      (bandwidth-optimal, wins for large messages);
    * one-shot: every rank reads all p-1 peers' buffers directly, two synchronizations
      (latency-optimal, wins for the small messages of low-batch decode).
    """
    if p <= 1 or nbytes <= 0:
        return 0.0
    ring = 2 * (p - 1) / p * nbytes / bw + 2 * (p - 1) * latency
    one_shot = (p - 1) * nbytes / bw + 2 * latency
    return min(ring, one_shot)


def alltoall_time(nbytes: float, p: int, bw: float, latency: float) -> float:
    """Each rank sends (p-1)/p of its payload to the other ranks over its own link."""
    if p <= 1 or nbytes <= 0:
        return 0.0
    return (p - 1) / p * nbytes / bw + (p - 1) * latency


def p2p_time(nbytes: float, bw: float, latency: float) -> float:
    return latency + nbytes / bw if nbytes > 0 else 0.0


def time_op(op: Op, hw: HardwareSpec, kernel_multiplier: float = 1.0) -> OpTiming:
    if op.kind in ("allreduce", "alltoall"):
        fn = allreduce_time if op.kind == "allreduce" else alltoall_time
        c = fn(op.comm_bytes, op.group, hw.link_bw, hw.link_latency)
        return OpTiming(op, 0.0, 0.0, c, 0.0, c, "comm", ideal=c)
    peak = hw.peak(op.dtype)
    compute = op.flops_padded / (peak * hw.eff_compute) if op.flops_padded else 0.0
    memory = op.bytes / (hw.mem_bw * hw.eff_memory) if op.bytes else 0.0
    overhead = hw.launch_overhead * op.kernels * kernel_multiplier
    body = max(compute, memory)
    bound = "compute" if compute >= memory else "memory"
    if overhead > body:
        bound = "overhead"
    ideal = max(op.flops / peak, op.bytes / hw.mem_bw)
    return OpTiming(op, compute, memory, 0.0, overhead, body + overhead, bound, ideal=ideal)


def roofline_point(op: Op, hw: HardwareSpec) -> dict:
    """Achieved FLOP/s and arithmetic intensity, for plotting on a roofline chart."""
    t = time_op(op, hw)
    return {
        "name": op.name,
        "category": op.category,
        "intensity": op.intensity,
        "achieved_flops": op.flops / t.time_s if t.time_s else 0.0,
        "attainable_flops": min(hw.peak(op.dtype), op.intensity * hw.mem_bw),
        "bound": t.bound,
        "time_s": t.total_s,
    }
