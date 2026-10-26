"""Transformer architectures, loaded from Hugging Face ``config.json`` files.

Supports dense decoders (Llama, Qwen), mixture-of-experts (Mixtral, DeepSeek) and
multi-head latent attention (DeepSeek MLA). Parameter counts are derived from the
architecture, and tests check them against published model sizes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources


@dataclass(frozen=True)
class ModelConfig:
    name: str
    hidden: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int  # dense MLP width
    vocab: int
    tie_embeddings: bool = False
    qkv_bias: bool = False
    # Mixture of experts
    n_experts: int = 0
    top_k: int = 0
    moe_intermediate: int = 0
    n_shared_experts: int = 0
    first_dense_layers: int = 0
    # Multi-head latent attention (DeepSeek-V2/V3)
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_dim: int = 0
    qk_rope_dim: int = 0
    v_head_dim: int = 0

    # ---- shape helpers -------------------------------------------------
    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def is_mla(self) -> bool:
        return self.kv_lora_rank > 0

    @property
    def moe_layers(self) -> int:
        return self.layers - self.first_dense_layers if self.is_moe else 0

    @property
    def dense_layers(self) -> int:
        return self.layers - self.moe_layers

    @property
    def qk_dim(self) -> int:
        return self.qk_nope_dim + self.qk_rope_dim if self.is_mla else self.head_dim

    @property
    def v_dim(self) -> int:
        return self.v_head_dim if self.is_mla else self.head_dim

    @property
    def gqa_group(self) -> int:
        return self.heads // self.kv_heads if not self.is_mla else self.heads

    def kv_elems_per_token_layer(self) -> int:
        """Elements cached per token per layer: K and V heads, or MLA's latent."""
        if self.is_mla:
            return self.kv_lora_rank + self.qk_rope_dim
        return 2 * self.kv_heads * self.head_dim

    def kv_bytes_per_token(self, kv_bytes: float) -> float:
        return self.layers * self.kv_elems_per_token_layer() * kv_bytes

    # ---- parameter counts ---------------------------------------------
    def attn_params(self) -> int:
        h = self.hidden
        if self.is_mla:
            q = (h * self.q_lora_rank + self.q_lora_rank + self.q_lora_rank * self.heads * self.qk_dim
                 if self.q_lora_rank else h * self.heads * self.qk_dim)
            kv = (h * (self.kv_lora_rank + self.qk_rope_dim) + self.kv_lora_rank
                  + self.kv_lora_rank * self.heads * (self.qk_nope_dim + self.v_head_dim))
            o = self.heads * self.v_head_dim * h
            return q + kv + o
        q = h * self.heads * self.head_dim
        kv = 2 * h * self.kv_heads * self.head_dim
        o = self.heads * self.head_dim * h
        bias = (self.heads + 2 * self.kv_heads) * self.head_dim if self.qkv_bias else 0
        return q + kv + o + bias

    def mlp_params(self, width: int) -> int:
        return 3 * self.hidden * width  # SwiGLU: gate, up, down

    def moe_layer_params(self, active: bool) -> int:
        experts = self.top_k if active else self.n_experts
        return (experts * self.mlp_params(self.moe_intermediate)
                + self.n_shared_experts * self.mlp_params(self.moe_intermediate)
                + self.hidden * self.n_experts)  # router

    def embed_params(self) -> int:
        return self.vocab * self.hidden * (1 if self.tie_embeddings else 2)

    def param_count(self, active: bool = False) -> int:
        per_layer_norms = 2 * self.hidden
        total = self.layers * (self.attn_params() + per_layer_norms)
        total += self.dense_layers * self.mlp_params(self.intermediate)
        total += self.moe_layers * self.moe_layer_params(active)
        total += self.embed_params() + self.hidden
        return total


def from_hf_config(cfg: dict, name: str | None = None) -> ModelConfig:
    """Build a ModelConfig from a Hugging Face ``config.json`` dict."""
    hidden = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg.get("num_key_value_heads", heads)
    head_dim = cfg.get("head_dim") or hidden // heads
    arch = (cfg.get("architectures") or [""])[0]
    n_experts = cfg.get("n_routed_experts") or cfg.get("num_local_experts") or 0
    top_k = cfg.get("num_experts_per_tok", 0) if n_experts else 0
    moe_inter = cfg.get("moe_intermediate_size") or (cfg["intermediate_size"] if n_experts else 0)
    return ModelConfig(
        name=name or cfg.get("_name_or_path", arch),
        hidden=hidden,
        layers=cfg["num_hidden_layers"],
        heads=heads,
        kv_heads=kv_heads,
        head_dim=head_dim,
        intermediate=cfg["intermediate_size"],
        vocab=cfg["vocab_size"],
        tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        qkv_bias=bool(cfg.get("attention_bias", False) or arch.startswith("Qwen2")),
        n_experts=n_experts,
        top_k=top_k,
        moe_intermediate=moe_inter,
        n_shared_experts=cfg.get("n_shared_experts") or 0,
        first_dense_layers=cfg.get("first_k_dense_replace", 0) if n_experts else 0,
        q_lora_rank=cfg.get("q_lora_rank") or 0,
        kv_lora_rank=cfg.get("kv_lora_rank") or 0,
        qk_nope_dim=cfg.get("qk_nope_head_dim") or 0,
        qk_rope_dim=cfg.get("qk_rope_head_dim") or 0,
        v_head_dim=cfg.get("v_head_dim") or 0,
    )


PRESET_FILES = {
    "llama-3.1-8b": "llama-3.1-8b.json",
    "llama-3.1-70b": "llama-3.1-70b.json",
    "llama-3.1-405b": "llama-3.1-405b.json",
    "mixtral-8x7b": "mixtral-8x7b.json",
    "deepseek-v3": "deepseek-v3.json",
    "qwen2.5-0.5b": "qwen2.5-0.5b.json",
    "qwen2.5-1.5b": "qwen2.5-1.5b.json",
}


def preset(name: str) -> ModelConfig:
    key = name.lower()
    if key not in PRESET_FILES:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(PRESET_FILES)}")
    text = resources.files("infersim").joinpath("data", PRESET_FILES[key]).read_text()
    return from_hf_config(json.loads(text), name=key)


def all_presets() -> dict[str, ModelConfig]:
    return {k: preset(k) for k in PRESET_FILES}
