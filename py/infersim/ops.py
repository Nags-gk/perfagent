"""Break one inference step into operators with FLOPs and bytes moved.

A *step* is one forward pass of the model over a batch that can mix decode
tokens (one new token per running sequence) and prefill chunks (several new
prompt tokens of one sequence). Every linear layer is mapped to a GEMM with
dimensions (M, N, K) on one tensor-parallel rank; attention, elementwise work
and collectives become their own ops. Each op carries a ``count`` (how many
layers it appears in) so a whole step is a short list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import ModelConfig
from .stack import Stack

DTYPE_BYTES = {"fp32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0, "int8": 1.0, "fp4": 0.5}


@dataclass(frozen=True)
class Precision:
    weights: str = "bf16"
    kv: str = "bf16"
    act: str = "bf16"

    @property
    def wb(self) -> float:
        return DTYPE_BYTES[self.weights]

    @property
    def kvb(self) -> float:
        return DTYPE_BYTES[self.kv]

    @property
    def ab(self) -> float:
        return DTYPE_BYTES[self.act]

    @property
    def gemm_dtype(self) -> str:
        """Tensor-core rate used for linear layers (W8A8 / W4A4 style)."""
        return self.weights if self.weights in ("fp8", "fp4", "int8") else self.act


PRECISIONS = {
    "bf16": Precision("bf16", "bf16", "bf16"),
    "fp16": Precision("fp16", "fp16", "fp16"),
    "fp32": Precision("fp32", "fp32", "fp32"),
    "fp8": Precision("fp8", "fp8", "bf16"),
    "fp4": Precision("fp4", "fp8", "bf16"),
}


@dataclass(frozen=True)
class StepShape:
    """Work in one forward pass.

    ``decode_ctx``: context length (cached tokens incl. the new one) for each decoding sequence.
    ``prefill``: (new_tokens, already_cached_tokens) for each sequence being prefilled.
    """

    decode_ctx: tuple[int, ...] = ()
    prefill: tuple[tuple[int, int], ...] = ()

    @property
    def tokens(self) -> int:
        return len(self.decode_ctx) + sum(n for n, _ in self.prefill)

    @property
    def seqs(self) -> int:
        return len(self.decode_ctx) + len(self.prefill)

    @staticmethod
    def decode(batch: int, ctx: int) -> StepShape:
        return StepShape(decode_ctx=(ctx,) * batch)

    @staticmethod
    def prefill_only(batch: int, prompt: int) -> StepShape:
        return StepShape(prefill=((prompt, 0),) * batch)


@dataclass(frozen=True)
class Op:
    name: str
    category: str  # attention | ffn | comm | other
    kind: str  # gemm | attention | elementwise | allreduce | alltoall
    flops: float = 0.0  # useful FLOPs on one rank
    flops_padded: float = 0.0  # FLOPs the hardware spends (tile padding)
    bytes: float = 0.0  # bytes moved to/from weight memory on one rank
    weight_bytes: float = 0.0  # part of ``bytes`` that is parameters
    kernels: int = 1
    count: int = 1  # number of layers this op appears in
    dtype: str = "bf16"
    comm_bytes: float = 0.0  # payload per rank for collectives
    group: int = 1  # ranks in the collective
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes if self.bytes else float("inf")


def _pad(x: float, tile: int) -> float:
    if x <= 0:
        return 0.0
    return math.ceil(x / tile) * tile


def gemm(name: str, category: str, m: float, n: float, k: float, prec: Precision,
         tile_m: int, tile_n: int, count: int, weights: bool = True) -> Op:
    flops = 2.0 * m * n * k
    padded = 2.0 * _pad(m, tile_m) * _pad(n, tile_n) * k
    wbytes = k * n * prec.wb if weights else 0.0
    act = (m * k + m * n) * prec.ab
    return Op(name, category, "gemm", flops, padded, wbytes + act, wbytes, 1, count,
              prec.gemm_dtype, meta={"m": m, "n": n, "k": k})


def expected_active_experts(tokens: int, n_experts: int, top_k: int) -> float:
    """Expected number of distinct experts hit when ``tokens`` each pick ``top_k`` of ``n_experts``."""
    if tokens <= 0:
        return 0.0
    return n_experts * (1.0 - (1.0 - top_k / n_experts) ** tokens)


def _attention_ops(model: ModelConfig, step: StepShape, tp: int, prec: Precision,
                   count: int, stack: Stack | None = None) -> list[Op]:
    heads = model.heads / tp
    math = stack is not None and stack.attention == "math" and not model.is_mla
    extra = (stack.math_passes * stack.math_bytes) if math else 0.0
    ops: list[Op] = []
    if model.is_mla:
        kv_elems = model.kv_lora_rank + model.qk_rope_dim  # latent is replicated on every TP rank
    else:
        kv_elems = 2 * max(1.0, model.kv_heads / tp) * model.head_dim
    # decode: one query per sequence reads its whole cache
    if step.decode_ctx:
        ctx_sum = float(sum(step.decode_ctx))
        if model.is_mla:  # weight-absorbed MLA attends over the latent
            per_key = 2 * (model.kv_lora_rank + model.qk_rope_dim) + 2 * model.kv_lora_rank
        else:
            per_key = 2 * model.qk_dim + 2 * model.v_dim
        flops = heads * per_key * ctx_sum
        b = len(step.decode_ctx)
        byts = ctx_sum * kv_elems * prec.kvb + b * kv_elems * prec.kvb  # read cache, append new
        byts += b * heads * (model.qk_dim + model.v_dim) * prec.ab
        if math:  # K/V expanded to every query head, in FP32, read/written ``math_passes`` times
            byts += extra * ctx_sum * heads * (model.qk_dim + model.v_dim)
        ops.append(Op("attn_decode", "attention", "attention", flops, flops, byts, 0.0, 1, count,
                      prec.act, meta={"ctx_sum": ctx_sum, "batch": b}))
    # prefill: causal attention of new tokens over cache + themselves (flash-style, no S^2 traffic)
    if step.prefill:
        keys = 0.0
        new = 0.0
        past = 0.0
        for n, p in step.prefill:
            keys += n * p + n * (n + 1) / 2.0
            new += n
            past += p
        if math:  # the math path computes every score, then masks: no causal savings
            keys = sum(n * (p + n) for n, p in step.prefill)
        flops = heads * 2 * (model.qk_dim + model.v_dim) * keys
        kv_full = (2 * max(1.0, model.kv_heads / tp) * model.head_dim if not model.is_mla
                   else 2 * heads * model.qk_dim)  # MLA prefill expands K/V per head
        byts = (past + new) * kv_full * prec.ab + new * kv_elems * prec.kvb
        byts += new * heads * (model.qk_dim + model.v_dim) * prec.ab
        if math:  # S x S scores and per-head K/V materialized in FP32
            byts += stack.math_bytes * heads * (stack.score_passes * keys
                                               + stack.math_passes * (past + new) * (model.qk_dim + model.v_dim))
        ops.append(Op("attn_prefill", "attention", "attention", flops, flops, byts, 0.0, 1, count,
                      prec.act, meta={"keys": keys, "new_tokens": new}))
    return ops


def _projection_ops(model: ModelConfig, t: int, tp: int, prec: Precision, tm: int, tn: int,
                    count: int) -> list[Op]:
    h = model.hidden
    if model.is_mla:
        hq = model.heads / tp
        ops = []
        if model.q_lora_rank:
            ops.append(gemm("q_a", "attention", t, model.q_lora_rank, h, prec, tm, tn, count))
            ops.append(gemm("q_b", "attention", t, hq * model.qk_dim, model.q_lora_rank, prec, tm, tn, count))
        else:
            ops.append(gemm("q", "attention", t, hq * model.qk_dim, h, prec, tm, tn, count))
        ops.append(gemm("kv_a", "attention", t, model.kv_lora_rank + model.qk_rope_dim, h, prec, tm, tn, count))
        ops.append(gemm("kv_b", "attention", t, hq * (model.qk_nope_dim + model.v_head_dim),
                        model.kv_lora_rank, prec, tm, tn, count))
        ops.append(gemm("o_proj", "attention", t, h, hq * model.v_head_dim, prec, tm, tn, count))
        return ops
    n_qkv = (model.heads / tp + 2 * max(1.0, model.kv_heads / tp)) * model.head_dim
    return [
        gemm("qkv_proj", "attention", t, n_qkv, h, prec, tm, tn, count),
        gemm("o_proj", "attention", t, h, model.heads / tp * model.head_dim, prec, tm, tn, count),
    ]


def _dense_mlp_ops(model: ModelConfig, t: int, width: int, tp: int, prec: Precision, tm: int,
                   tn: int, count: int, prefix: str = "") -> list[Op]:
    w = width / tp
    return [
        gemm(prefix + "gate_up", "ffn", t, 2 * w, model.hidden, prec, tm, tn, count),
        gemm(prefix + "down", "ffn", t, model.hidden, w, prec, tm, tn, count),
    ]


def _moe_ops(model: ModelConfig, t: int, tp: int, prec: Precision, tm: int, tn: int, count: int,
             expert_parallel: bool) -> list[Op]:
    h, ie, e, k = model.hidden, model.moe_intermediate, model.n_experts, model.top_k
    ops = [gemm("router", "ffn", t, e, h, prec, tm, tn, count)]
    e_act = expected_active_experts(t, e, k)
    routed = t * k  # token-expert pairs
    if expert_parallel:
        e_dev, width, pairs_dev = e_act / tp, ie, routed / tp
    else:
        e_dev, width, pairs_dev = e_act, ie / tp, routed
    m_per_expert = pairs_dev / e_dev if e_dev else 0.0
    flops = pairs_dev * 2 * 3 * h * width
    padded = e_dev * 2 * _pad(m_per_expert, tm) * (_pad(2 * width, tn) * h + _pad(h, tn) * width)
    wbytes = e_dev * 3 * h * width * prec.wb
    act = pairs_dev * (2 * h + 3 * width) * prec.ab
    ops.append(Op("experts", "ffn", "gemm", flops, padded, wbytes + act, wbytes, 2, count,
                  prec.gemm_dtype, meta={"active_experts": e_act, "tokens_per_expert": m_per_expert}))
    if model.n_shared_experts:
        ops += _dense_mlp_ops(model, t, ie * model.n_shared_experts, tp, prec, tm, tn, count,
                              prefix="shared_")
    if expert_parallel and tp > 1:
        payload = (t / tp) * k * h * prec.ab
        ops.append(Op("ep_dispatch", "comm", "alltoall", count=count, comm_bytes=payload, group=tp))
        ops.append(Op("ep_combine", "comm", "alltoall", count=count, comm_bytes=payload, group=tp))
    return ops


def _elementwise(model: ModelConfig, t: int, tp: int, prec: Precision, width: int, count: int) -> Op:
    h = model.hidden
    # 2 RMSNorms, 2 residual adds, rotary, SwiGLU activation
    byts = (10 * t * h + 3 * t * width / tp) * prec.ab
    return Op("norm_act_residual", "other", "elementwise", 12.0 * t * h, 12.0 * t * h, byts, 0.0,
              6, count, prec.act)


def step_ops(model: ModelConfig, step: StepShape, tp: int = 1, prec: Precision | None = None,
             tile_m: int = 64, tile_n: int = 128, expert_parallel: bool = False,
             stack: Stack | None = None) -> list[Op]:
    """All ops for one forward pass of ``step`` on one tensor-parallel rank (all layers)."""
    prec = prec or PRECISIONS["bf16"]
    t = step.tokens
    if t == 0:
        return []
    ops: list[Op] = []
    groups = [("dense", model.dense_layers), ("moe", model.moe_layers)]
    for kind, n_layers in groups:
        if n_layers == 0:
            continue
        ops += _projection_ops(model, t, tp, prec, tile_m, tile_n, n_layers)
        ops += _attention_ops(model, step, tp, prec, n_layers, stack)
        if kind == "dense":
            ops += _dense_mlp_ops(model, t, model.intermediate, tp, prec, tile_m, tile_n, n_layers)
            width = model.intermediate
        else:
            ops += _moe_ops(model, t, tp, prec, tile_m, tile_n, n_layers, expert_parallel)
            width = model.moe_intermediate * (model.top_k + model.n_shared_experts)
        ops.append(_elementwise(model, t, tp, prec, width, n_layers))
        if tp > 1:
            ar = t * model.hidden * prec.ab
            ops.append(Op("tp_allreduce_attn", "comm", "allreduce", count=n_layers, comm_bytes=ar, group=tp))
            if not (kind == "moe" and expert_parallel):
                ops.append(Op("tp_allreduce_ffn", "comm", "allreduce", count=n_layers, comm_bytes=ar, group=tp))
    # model head: embedding lookup, final norm and logits for one row per sequence
    ops.append(Op("embed", "other", "elementwise", 0.0, 0.0, t * model.hidden * prec.ab * 2, 0.0, 1, 1, prec.act))
    ops.append(gemm("lm_head", "other", step.seqs, model.vocab / tp, model.hidden, prec, tile_m, tile_n, 1))
    return ops


def total(ops: list[Op], attr: str) -> float:
    return sum(getattr(o, attr) * o.count for o in ops)
