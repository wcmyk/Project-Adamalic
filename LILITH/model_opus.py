"""Opus-inspired decoder that mixes frontier tricks with efficient defaults."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .activations import get_activation
from .config_phase2 import OpusPrototypeConfig
from .positional import ALiBi, RotaryEmbedding


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


class RMSNorm(nn.Module):
    """RMSNorm used by many frontier models (e.g., PaLM, LLaMA)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class MultiQueryAttention(nn.Module):
    """Multi/query-grouped attention with optional RoPE + ALiBi."""

    def __init__(self, config: OpusPrototypeConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.kv_heads = 1 if config.use_multi_query else config.n_heads
        self.kv_heads = max(1, config.kv_heads) if config.use_multi_query else self.kv_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.kv_proj = nn.Linear(config.d_model, self.kv_heads * self.head_dim * 2, bias=False)
        self.out_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.attn_dropout)

        self.rotary = RotaryEmbedding(self.head_dim, max_seq_len=config.max_seq_len, base=config.rope_base) if config.use_rope else None
        self.alibi = ALiBi(config.n_heads, max_seq_len=config.max_seq_len) if config.use_alibi else None

    def _shape_qkv(self, q: torch.Tensor, kv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.n_heads, self.head_dim)
        kv = kv.view(bsz, seq_len, self.kv_heads, 2, self.head_dim)
        k = kv[:, :, :, 0, :]
        v = kv[:, :, :, 1, :]

        if self.kv_heads == 1:
            k = k.expand(-1, -1, self.n_heads, -1)
            v = v.expand(-1, -1, self.n_heads, -1)
        else:
            k = k.repeat_interleave(self.n_heads // self.kv_heads, dim=2)
            v = v.repeat_interleave(self.n_heads // self.kv_heads, dim=2)
        return q, k, v

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        q, k, v = self._shape_qkv(q, kv)

        if self.rotary is not None:
            q, k = self.rotary(q, k)

        attn_scores = torch.einsum("bshd,bThd->bhst", q, k) / math.sqrt(self.head_dim)

        if self.alibi is not None:
            attn_scores = attn_scores + self.alibi(q.size(1)).to(attn_scores.device)

        if mask is not None:
            attn_scores = attn_scores + mask

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.einsum("bhst,bThd->bshd", attn_weights, v)
        attn_output = attn_output.reshape(x.size(0), x.size(1), -1)
        return self.out_proj(attn_output)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, config: OpusPrototypeConfig):
        super().__init__()
        hidden = int(config.d_ff * config.ffn_mult)
        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(config.d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.resid_dropout)
        activation = config.activation.lower()
        if activation == "swiglu":
            self.act = nn.SiLU()
        elif activation == "geglu":
            self.act = nn.GELU()
        elif activation == "reglu":
            self.act = nn.ReLU()
        else:
            self.act = get_activation(config.activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = self.act(self.w1(x)) * self.w2(x)
        return self.dropout(self.w3(gated))


class MoELayer(nn.Module):
    """Small MoE block to boost capacity without huge active params."""

    def __init__(self, config: OpusPrototypeConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = min(config.experts_top_k, self.num_experts)
        self.gate = nn.Linear(config.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLUFeedForward(config) for _ in range(self.num_experts))
        self.dropout = nn.Dropout(config.expert_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_scores = torch.softmax(self.gate(x), dim=-1)
        top_vals, top_idx = gate_scores.topk(self.top_k, dim=-1)
        mask = torch.zeros_like(gate_scores).scatter(-1, top_idx, top_vals)
        weights = mask / (mask.sum(dim=-1, keepdim=True) + 1e-9)

        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        mixed = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        return self.dropout(mixed)


class OpusBlock(nn.Module):
    def __init__(self, config: OpusPrototypeConfig):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn = MultiQueryAttention(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ff = MoELayer(config) if config.use_moe else SwiGLUFeedForward(config)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class OpusDecoder(nn.Module):
    """Claude Opus-inspired decoder with resource-aware defaults."""

    def __init__(self, config: OpusPrototypeConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = None if config.use_rope or config.use_alibi else nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList(OpusBlock(config) for _ in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.head.weight = self.token_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch, seq_len = token_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")

        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        x = self.token_embed(token_ids)
        if self.pos_embed is not None:
            x = x + self.pos_embed(positions)

        mask = _causal_mask(seq_len, token_ids.device)

        for block in self.blocks:
            if self.config.use_gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        generated = prompt.clone()
        device = prompt.device
        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_seq_len :]
            logits = self.forward(window)
            next_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction=reduction,
        )

    def count_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_num_params(self) -> dict:
        return {
            "total": self.count_parameters(trainable_only=False),
            "trainable": self.count_parameters(trainable_only=True),
            "embeddings": sum(p.numel() for p in self.token_embed.parameters()),
            "transformer": sum(p.numel() for p in self.blocks.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }


__all__ = [
    "OpusDecoder",
    "OpusPrototypeConfig",
]
