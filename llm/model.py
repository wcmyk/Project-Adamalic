"""A compact GPT-style decoder built with PyTorch primitives."""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class GPTDecoder(nn.Module):
    """Transformer decoder block for next-token prediction."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (batch, seq)
        batch, seq_len = token_ids.shape
        device = token_ids.device
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        mask = _causal_mask(seq_len, device=device)
        x = self.transformer(x, mask=mask)
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
        """Greedy/temperature sampling generation."""

        generated = prompt.clone()
        device = prompt.device
        for _ in range(max_new_tokens):
            window = generated[:, -self.config.max_seq_len :]
            logits = self.forward(window)
            next_token_logits = logits[:, -1, :]
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    def loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

