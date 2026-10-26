"""Software stacks: the same GPU runs very differently under different runtimes.

Hardware specs say what the chip can do; the stack says how the inference software
uses it. Three properties matter for the model:

* ``attention``: ``"flash"`` keeps attention scores on chip (FlashAttention, TensorRT-LLM,
  vLLM). ``"math"`` is PyTorch's fallback SDPA path, used on GPUs without FlashAttention
  support (FlashAttention needs Ampere or newer; a T4 is Turing). It materializes the
  S x S score matrix in prefill and expands grouped K/V to every query head in decode,
  in FP32, so attention traffic is many times the KV-cache size. The two materialized
  tensors go through different op chains, so each has its own pass count.
* ``kernel_multiplier``: kernels the stack launches per step relative to the fused minimum
  the op list assumes. Eager PyTorch issues separate kernels for casts, reshapes, rotary
  embedding pieces and residual adds. Set it from ``torch.profiler`` counts, not by fitting.
* ``async_launch``: whether host-side launch work overlaps GPU execution. Eager PyTorch
  enqueues kernels asynchronously, so a step costs about max(host time, GPU time); a
  CUDA-graph stack replays the whole step with negligible host time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Stack:
    name: str = "fused serving engine"
    attention: str = "flash"  # flash | math
    math_passes: float = 3.0  # decode: passes over K/V expanded per query head (expand, read)
    score_passes: float = 5.0  # prefill: passes over the S x S scores (QK^T, scale/mask, softmax, PV)
    math_bytes: float = 4.0  # element size of the materialized tensors (FP32)
    kernel_multiplier: float = 1.0
    async_launch: bool = False

    def with_overrides(self, **kwargs) -> Stack:
        return replace(self, **kwargs)


STACKS = {
    "fused": Stack(),
    "eager": Stack(name="PyTorch eager + transformers (SDPA math path)", attention="math",
                   kernel_multiplier=5.0, async_launch=True),
}
