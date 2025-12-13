"""A compact GPT-style decoder built with PyTorch primitives."""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from .config import ModelConfig


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class GPTDecoder(nn.Module):
    """Transformer decoder block for next-token prediction."""

    def __init__(self, config: ModelConfig, use_gradient_checkpointing: bool = False):
        super().__init__()
        self.config = config
        self.use_gradient_checkpointing = use_gradient_checkpointing

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

        # Optionally tie embeddings
        # self.head.weight = self.token_embed.weight  # Weight tying

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with small random values."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # token_ids: (batch, seq)
        batch, seq_len = token_ids.shape
        device = token_ids.device
        if seq_len > self.config.max_seq_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_seq_len {self.config.max_seq_len}")
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = self.dropout(x)
        mask = _causal_mask(seq_len, device=device)

        # Use gradient checkpointing if enabled (saves memory during training)
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint.checkpoint(self._forward_transformer, x, mask, use_reentrant=False)
        else:
            x = self.transformer(x, mask=mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits

    def _forward_transformer(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Helper for gradient checkpointing."""
        return self.transformer(x, mask=mask)

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

    def loss(self, logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Compute cross-entropy loss.

        Args:
            logits: Model predictions (batch, seq, vocab_size)
            targets: Target token IDs (batch, seq)
            reduction: Loss reduction method ('mean', 'sum', 'none')

        Returns:
            Loss value
        """
        return nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction=reduction
        )

    def count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters.

        Args:
            trainable_only: Only count trainable parameters

        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_num_params(self) -> dict:
        """Get detailed parameter count breakdown.

        Returns:
            Dictionary with parameter counts
        """
        return {
            "total": self.count_parameters(trainable_only=False),
            "trainable": self.count_parameters(trainable_only=True),
            "embeddings": sum(p.numel() for p in self.token_embed.parameters()) +
                         sum(p.numel() for p in self.pos_embed.parameters()),
            "transformer": sum(p.numel() for p in self.transformer.parameters()),
            "head": sum(p.numel() for p in self.head.parameters()),
        }

