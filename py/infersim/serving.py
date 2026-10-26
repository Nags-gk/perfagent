"""Discrete-event serving simulator with continuous batching and chunked prefill.

Each iteration the scheduler admits waiting requests while KV cache has room for
their full prompt + output (no preemption), builds one batch from every running
decode plus as many prefill tokens as the token budget allows, and advances the
clock by the deployment's step latency. Deployments that pipeline independent
batches (``concurrency > 1``) are simulated as that many engines splitting the
requests round-robin.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .ops import StepShape
from .system import Disaggregated


@dataclass(frozen=True)
class Request:
    rid: int
    arrival: float
    prompt: int
    output: int


@dataclass
class _Seq:
    req: Request
    prefilled: int = 0
    generated: int = 0
    first_token: float = -1.0
    cached: bool = False  # prompt KV already present (disaggregated decode pool)


@dataclass
class RequestResult:
    rid: int
    arrival: float
    first_token: float
    finish: float
    prompt: int
    output: int

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def tpot(self) -> float:
        return (self.finish - self.first_token) / (self.output - 1) if self.output > 1 else 0.0

    @property
    def e2e(self) -> float:
        return self.finish - self.arrival


@dataclass
class SimResult:
    requests: list[RequestResult]
    devices: int
    steps: int
    batch_tokens: list[int] = field(default_factory=list)

    def _pct(self, xs: list[float], q: float) -> float:
        xs = sorted(xs)
        if not xs:
            return 0.0
        i = min(len(xs) - 1, max(0, round(q / 100 * (len(xs) - 1))))
        return xs[i]

    def summary(self) -> dict:
        r = self.requests
        start = min(x.arrival for x in r)
        end = max(x.finish for x in r)
        out_tokens = sum(x.output for x in r)
        ttft = [x.ttft for x in r]
        tpot = [x.tpot for x in r if x.output > 1]
        return {
            "requests": len(r),
            "makespan_s": end - start,
            "throughput_tok_s": out_tokens / (end - start),
            "throughput_tok_s_per_device": out_tokens / (end - start) / self.devices,
            "ttft_p50_ms": self._pct(ttft, 50) * 1e3,
            "ttft_p99_ms": self._pct(ttft, 99) * 1e3,
            "tpot_p50_ms": self._pct(tpot, 50) * 1e3,
            "tpot_p99_ms": self._pct(tpot, 99) * 1e3,
            "tok_s_per_user_p50": 1.0 / self._pct(tpot, 50) if tpot and self._pct(tpot, 50) else 0.0,
            "e2e_p50_s": self._pct([x.e2e for x in r], 50),
            "mean_batch_tokens": sum(self.batch_tokens) / max(1, len(self.batch_tokens)),
            "steps": self.steps,
        }


def poisson_workload(rate: float, n: int, prompt: int, output: int, jitter: float = 0.0,
                     seed: int = 0) -> list[Request]:
    """``n`` requests with exponential inter-arrival gaps (mean ``1/rate``) and lengths
    drawn uniformly within +-``jitter`` of the given means."""
    rng = random.Random(seed)
    t = 0.0
    reqs = []
    for i in range(n):
        t += rng.expovariate(rate)
        p = max(1, int(prompt * (1 + rng.uniform(-jitter, jitter))))
        o = max(1, int(output * (1 + rng.uniform(-jitter, jitter))))
        reqs.append(Request(i, t, p, o))
    return reqs


def _run_engine(dep, reqs: list[Request], kv_capacity: int, max_batch: int, max_tokens: int,
                cached: bool) -> tuple[list[RequestResult], int, list[int]]:
    waiting = sorted(reqs, key=lambda r: r.arrival)
    wi = 0
    running: list[_Seq] = []
    kv_used = 0
    t = 0.0
    done: list[RequestResult] = []
    steps = 0
    batch_tokens: list[int] = []
    while wi < len(waiting) or running:
        while (wi < len(waiting) and waiting[wi].arrival <= t and len(running) < max_batch
               and kv_used + waiting[wi].prompt + waiting[wi].output <= kv_capacity):
            r = waiting[wi]
            s = _Seq(r, cached=cached)
            if cached:  # prompt already processed elsewhere; first token arrived with it
                s.prefilled, s.generated, s.first_token = r.prompt, 1, r.arrival
            running.append(s)
            kv_used += r.prompt + r.output
            wi += 1
        if not running:
            if wi < len(waiting) and waiting[wi].prompt + waiting[wi].output > kv_capacity:
                raise ValueError(f"request {waiting[wi].rid} needs more KV cache than the deployment has")
            t = max(t, waiting[wi].arrival)
            continue
        decode = [s for s in running if s.prefilled == s.req.prompt and s.generated < s.req.output]
        budget = max(0, max_tokens - len(decode))
        chunks: list[tuple[_Seq, int]] = []
        for s in running:
            if s.prefilled < s.req.prompt and budget > 0:
                n = min(s.req.prompt - s.prefilled, budget)
                chunks.append((s, n))
                budget -= n
        shape = StepShape(tuple(s.req.prompt + s.generated for s in decode),
                          tuple((n, s.prefilled) for s, n in chunks))
        t += dep.step(shape).latency_s
        steps += 1
        batch_tokens.append(shape.tokens)
        for s in decode:
            s.generated += 1
        for s, n in chunks:
            s.prefilled += n
            if s.prefilled == s.req.prompt:
                s.generated = 1
                s.first_token = t
        still = []
        for s in running:
            if s.generated >= s.req.output and s.prefilled == s.req.prompt:
                done.append(RequestResult(s.req.rid, s.req.arrival, s.first_token, t, s.req.prompt, s.req.output))
                kv_used -= s.req.prompt + s.req.output
            else:
                still.append(s)
        running = still
    return done, steps, batch_tokens


def simulate(dep, reqs: list[Request], max_batch: int = 256, max_tokens: int = 8192) -> SimResult:
    """Serve ``reqs`` on ``dep`` and return per-request timings."""
    if isinstance(dep, Disaggregated):
        return _simulate_disaggregated(dep, reqs, max_batch, max_tokens)
    probe = dep.step(StepShape.decode(1, 1))
    lanes = max(1, min(probe.concurrency, len(reqs)))
    cap = dep.kv_capacity_tokens() // lanes
    results: list[RequestResult] = []
    steps = 0
    batch_tokens: list[int] = []
    for lane in range(lanes):
        d, s, b = _run_engine(dep, reqs[lane::lanes], cap, max_batch, max_tokens, cached=False)
        results += d
        steps += s
        batch_tokens += b
    return SimResult(sorted(results, key=lambda r: r.rid), dep.devices, steps, batch_tokens)


def _simulate_disaggregated(dep: Disaggregated, reqs: list[Request], max_batch: int,
                            max_tokens: int) -> SimResult:
    # Phase 1: prefill pool produces each request's first token.
    first = [Request(r.rid, r.arrival, r.prompt, 1) for r in reqs]
    pre = simulate(dep.prefill, first, max_batch, max_tokens)
    first_at = {x.rid: x.first_token for x in pre.requests}
    # Phase 2: KV cache ships to the decode pool, which generates the rest.
    handoff = [Request(r.rid, first_at[r.rid] + dep.kv_transfer_s(r.prompt), r.prompt, r.output)
               for r in reqs]
    dec = simulate_cached(dep.decode, handoff, max_batch, max_tokens)
    by_id = {x.rid: x for x in dec}
    out = []
    for r in reqs:
        d = by_id[r.rid]
        out.append(RequestResult(r.rid, r.arrival, first_at[r.rid], d.finish, r.prompt, r.output))
    return SimResult(out, dep.devices, pre.steps + len(dec), pre.batch_tokens)


def simulate_cached(dep, reqs: list[Request], max_batch: int, max_tokens: int) -> list[RequestResult]:
    probe = dep.step(StepShape.decode(1, 1))
    lanes = max(1, min(probe.concurrency, len(reqs)))
    cap = dep.kv_capacity_tokens() // lanes
    out: list[RequestResult] = []
    for lane in range(lanes):
        d, _, _ = _run_engine(dep, reqs[lane::lanes], cap, max_batch, max_tokens, cached=True)
        out += d
    return out
