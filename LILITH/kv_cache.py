"""KV-cache implementation for faster autoregressive generation."""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn as nn


class KVCache:
    """Key-Value cache for transformer attention layers."""

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_layers: int,
        n_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize KV cache.

        Args:
            max_batch_size: Maximum batch size
            max_seq_len: Maximum sequence length
            n_layers: Number of transformer layers
            n_heads: Number of attention heads
            head_dim: Dimension per attention head
            device: Device to store cache
            dtype: Data type for cache
        """
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        # Initialize cache tensors
        # Shape: (n_layers, max_batch_size, max_seq_len, n_heads, head_dim)
        self.k_cache = torch.zeros(
            n_layers, max_batch_size, max_seq_len, n_heads, head_dim,
            device=device, dtype=dtype
        )
        self.v_cache = torch.zeros(
            n_layers, max_batch_size, max_seq_len, n_heads, head_dim,
            device=device, dtype=dtype
        )

        # Track current sequence length per batch element
        self.seq_lengths = torch.zeros(max_batch_size, dtype=torch.long, device=device)

    def update(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update cache with new key-value pairs.

        Args:
            layer_idx: Layer index
            k: New keys (batch, seq_len, n_heads, head_dim)
            v: New values (batch, seq_len, n_heads, head_dim)
            batch_idx: Optional batch indices to update

        Returns:
            Complete cached keys and values for this layer
        """
        batch_size, seq_len, _, _ = k.shape

        if batch_idx is None:
            batch_idx = torch.arange(batch_size, device=self.device)

        # Get current sequence positions
        start_pos = self.seq_lengths[batch_idx]

        # Update cache
        for i, b_idx in enumerate(batch_idx):
            pos = start_pos[i].item()
            self.k_cache[layer_idx, b_idx, pos:pos+seq_len] = k[i]
            self.v_cache[layer_idx, b_idx, pos:pos+seq_len] = v[i]

        # Update sequence lengths
        self.seq_lengths[batch_idx] += seq_len

        # Return full cached sequences
        max_len = self.seq_lengths[batch_idx].max().item()
        cached_k = self.k_cache[layer_idx, batch_idx, :max_len]
        cached_v = self.v_cache[layer_idx, batch_idx, :max_len]

        return cached_k, cached_v

    def reset(self, batch_idx: Optional[torch.Tensor] = None):
        """Reset cache for specified batch elements.

        Args:
            batch_idx: Optional batch indices to reset. If None, reset all.
        """
        if batch_idx is None:
            self.k_cache.zero_()
            self.v_cache.zero_()
            self.seq_lengths.zero_()
        else:
            self.k_cache[:, batch_idx].zero_()
            self.v_cache[:, batch_idx].zero_()
            self.seq_lengths[batch_idx].zero_()

    def get_seq_length(self, batch_idx: int = 0) -> int:
        """Get current sequence length for a batch element."""
        return self.seq_lengths[batch_idx].item()


class CachedGPTDecoder(nn.Module):
    """GPTDecoder with KV-cache support for faster generation.

    This is a wrapper that adds caching capabilities to the base GPTDecoder.
    For full implementation, the base model would need to be modified to
    support incremental key-value computation.
    """

    def __init__(self, base_model, max_batch_size: int = 8):
        super().__init__()
        self.model = base_model
        self.config = base_model.config

        # Initialize cache (would need actual head_dim from model)
        # For now, this is a placeholder showing the architecture
        self.cache: Optional[KVCache] = None
        self.max_batch_size = max_batch_size

    def enable_cache(self, device: Optional[torch.device] = None):
        """Enable KV-cache for generation."""
        if device is None:
            device = next(self.model.parameters()).device

        # This is simplified - actual implementation would need proper head_dim
        head_dim = self.config.d_model // self.config.n_heads

        self.cache = KVCache(
            max_batch_size=self.max_batch_size,
            max_seq_len=self.config.max_seq_len,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            head_dim=head_dim,
            device=device,
        )

    def disable_cache(self):
        """Disable KV-cache."""
        self.cache = None

    @torch.no_grad()
    def generate_with_cache(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate using KV-cache for faster inference.

        Note: This is a demonstration. Full implementation would require
        modifying the base GPTDecoder to support incremental attention.
        """
        if self.cache is None:
            # Fallback to standard generation
            return self.model.generate(prompt, max_new_tokens, temperature)

        self.cache.reset()
        generated = prompt.clone()

        for _ in range(max_new_tokens):
            # In full implementation, would pass cache to forward pass
            # to avoid recomputing attention for previous tokens
            window = generated[:, -self.config.max_seq_len:]
            logits = self.model(window)
            next_token_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated


__all__ = ["KVCache", "CachedGPTDecoder"]
