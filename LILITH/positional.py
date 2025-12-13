"""Advanced positional encoding implementations."""
from __future__ import annotations

import math
import torch
import torch.nn as nn
from typing import Tuple


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) from Su et al. (2021).

    Used in GPT-NeoX, LLaMA, PaLM, and other modern LLMs.
    More effective than learned positional embeddings for long sequences.

    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10000,
    ):
        """Initialize RoPE.

        Args:
            dim: Dimension of each attention head
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
        """
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency tensor
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Cache for cos/sin values
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached cos/sin values if needed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            self._cos_cached = emb.cos()[None, :, None, :]
            self._sin_cached = emb.sin()[None, :, None, :]

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to query and key tensors.

        Args:
            q: Query tensor (batch, seq_len, n_heads, head_dim)
            k: Key tensor (batch, seq_len, n_heads, head_dim)

        Returns:
            Rotated query and key tensors
        """
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device)

        # Apply rotation
        q_rot = apply_rotary_pos_emb(q, self._cos_cached[:, :seq_len], self._sin_cached[:, :seq_len])
        k_rot = apply_rotary_pos_emb(k, self._cos_cached[:, :seq_len], self._sin_cached[:, :seq_len])

        return q_rot, k_rot


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding.

    Args:
        x: Input tensor (batch, seq_len, n_heads, head_dim)
        cos: Cosine values
        sin: Sine values

    Returns:
        Rotated tensor
    """
    # Split into first and second half
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]

    # Apply rotation
    # [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
    return torch.cat(
        [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
        dim=-1,
    )


class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi) from Press et al. (2021).

    Alternative to positional encodings that adds biases to attention scores.
    Enables better extrapolation to longer sequences than seen during training.

    Reference: https://arxiv.org/abs/2108.12409
    """

    def __init__(self, n_heads: int, max_seq_len: int = 2048):
        """Initialize ALiBi.

        Args:
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len

        # Compute slopes for each head
        slopes = self._get_slopes(n_heads)
        self.register_buffer("slopes", slopes)

        # Precompute bias matrix
        alibi = self._build_alibi_tensor(max_seq_len, slopes)
        self.register_buffer("alibi", alibi)

    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """Compute slopes for each attention head."""
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        # Handle non-power-of-2 head counts
        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes_a = get_slopes_power_of_2(closest_power_of_2)
            slopes_b = self._get_slopes(2 * closest_power_of_2)[0::2][: n_heads - closest_power_of_2]
            return torch.tensor(slopes_a + slopes_b)

    def _build_alibi_tensor(self, max_seq_len: int, slopes: torch.Tensor) -> torch.Tensor:
        """Build the ALiBi bias tensor."""
        # Create distance matrix
        context_position = torch.arange(max_seq_len)[:, None]
        memory_position = torch.arange(max_seq_len)[None, :]
        relative_position = memory_position - context_position

        # Apply slopes
        alibi = slopes[:, None, None] * relative_position[None, :, :]
        return alibi

    def forward(self, seq_len: int) -> torch.Tensor:
        """Get ALiBi bias for given sequence length.

        Args:
            seq_len: Sequence length

        Returns:
            Bias tensor (n_heads, seq_len, seq_len)
        """
        return self.alibi[:, :seq_len, :seq_len]


__all__ = [
    "RotaryEmbedding",
    "ALiBi",
    "apply_rotary_pos_emb",
]
