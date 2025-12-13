"""Advanced activation functions for neural networks."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """SwiGLU activation from Shazeer (2020).

    Used in PaLM, LLaMA, and other modern LLMs.
    Outperforms GELU and other standard activations.

    SwiGLU(x) = Swish(xW) ⊙ (xV)
    where Swish(x) = x * sigmoid(x)

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        """Initialize SwiGLU.

        Args:
            dim_in: Input dimension
            dim_out: Output dimension
            bias: Whether to use bias
        """
        super().__init__()
        # Need 2x dim_out for the gate mechanism
        self.linear = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU activation.

        Args:
            x: Input tensor (..., dim_in)

        Returns:
            Output tensor (..., dim_out)
        """
        x, gate = self.linear(x).chunk(2, dim=-1)
        return F.silu(gate) * x


class GeGLU(nn.Module):
    """GeGLU activation - GELU variant with gating.

    GeGLU(x) = GELU(xW) ⊙ (xV)

    Used in some vision transformers and language models.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        """Initialize GeGLU.

        Args:
            dim_in: Input dimension
            dim_out: Output dimension
            bias: Whether to use bias
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeGLU activation.

        Args:
            x: Input tensor (..., dim_in)

        Returns:
            Output tensor (..., dim_out)
        """
        x, gate = self.linear(x).chunk(2, dim=-1)
        return F.gelu(gate) * x


class ReGLU(nn.Module):
    """ReGLU activation - ReLU variant with gating.

    ReGLU(x) = ReLU(xW) ⊙ (xV)
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = False):
        """Initialize ReGLU.

        Args:
            dim_in: Input dimension
            dim_out: Output dimension
            bias: Whether to use bias
        """
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out * 2, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply ReGLU activation.

        Args:
            x: Input tensor (..., dim_in)

        Returns:
            Output tensor (..., dim_out)
        """
        x, gate = self.linear(x).chunk(2, dim=-1)
        return F.relu(gate) * x


class MishActivation(nn.Module):
    """Mish activation function.

    Mish(x) = x * tanh(softplus(x))

    Smooth, non-monotonic activation that can improve training dynamics.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Mish activation.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return x * torch.tanh(F.softplus(x))


class SquaredReLU(nn.Module):
    """Squared ReLU activation.

    SquaredReLU(x) = ReLU(x)^2

    Used in some vision transformers. Provides smoother gradients than ReLU.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Squared ReLU.

        Args:
            x: Input tensor

        Returns:
            Activated tensor
        """
        return torch.pow(F.relu(x), 2)


def get_activation(name: str, **kwargs) -> nn.Module:
    """Get activation function by name.

    Args:
        name: Activation name ('relu', 'gelu', 'swiglu', 'geglu', 'reglu', 'mish', 'squared_relu', 'silu')
        **kwargs: Additional arguments for gated activations (dim_in, dim_out)

    Returns:
        Activation module

    Raises:
        ValueError: If activation name is unknown
    """
    activations = {
        'relu': nn.ReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'mish': MishActivation,
        'squared_relu': SquaredReLU,
    }

    gated_activations = {
        'swiglu': SwiGLU,
        'geglu': GeGLU,
        'reglu': ReGLU,
    }

    name = name.lower()

    if name in activations:
        return activations[name]()
    elif name in gated_activations:
        if 'dim_in' not in kwargs or 'dim_out' not in kwargs:
            raise ValueError(f"{name} requires dim_in and dim_out arguments")
        return gated_activations[name](**kwargs)
    else:
        raise ValueError(f"Unknown activation: {name}")


__all__ = [
    "SwiGLU",
    "GeGLU",
    "ReGLU",
    "MishActivation",
    "SquaredReLU",
    "get_activation",
]
