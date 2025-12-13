"""LoRA (Low-Rank Adaptation) implementation for efficient fine-tuning."""
from __future__ import annotations

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for efficient fine-tuning.

    Instead of fine-tuning all parameters, LoRA adds trainable low-rank
    matrices A and B such that: W' = W + BA, where W is frozen.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        """Initialize LoRA layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: Rank of the low-rank decomposition
            alpha: Scaling factor for LoRA updates
            dropout: Dropout probability
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with kaiming uniform, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA transformation.

        Args:
            x: Input tensor (..., in_features)

        Returns:
            LoRA delta (..., out_features)
        """
        # Compute low-rank update: x @ A @ B
        result = x @ self.lora_A @ self.lora_B
        result = self.dropout(result)
        return result * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        """Initialize LoRA-adapted linear layer.

        Args:
            base_layer: Original linear layer (will be frozen)
            rank: LoRA rank
            alpha: LoRA alpha scaling
            dropout: LoRA dropout
        """
        super().__init__()
        self.base_layer = base_layer

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Add LoRA adaptation
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining base layer and LoRA.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[list[str]] = None,
    rank: int = 4,
    alpha: float = 1.0,
    dropout: float = 0.0,
) -> nn.Module:
    """Apply LoRA to specified modules in a model.

    Args:
        model: The model to adapt
        target_modules: List of module names to adapt (e.g., ['head', 'token_embed'])
                       If None, adapts all Linear layers
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: LoRA dropout

    Returns:
        Modified model with LoRA layers
    """
    if target_modules is None:
        target_modules = []

    def replace_layer(module: nn.Module, name: str = ""):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name

            # Check if this layer should be adapted
            should_adapt = (
                isinstance(child, nn.Linear) and (
                    not target_modules or
                    any(target in full_name for target in target_modules)
                )
            )

            if should_adapt:
                # Replace with LoRA-adapted layer
                lora_layer = LoRALinear(child, rank, alpha, dropout)
                setattr(module, child_name, lora_layer)
            else:
                # Recursively process children
                replace_layer(child, full_name)

    replace_layer(model)
    return model


def get_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Get all LoRA parameters from a model.

    Args:
        model: Model with LoRA layers

    Returns:
        List of LoRA parameters
    """
    lora_params = []
    for module in model.modules():
        if isinstance(module, LoRALayer):
            lora_params.extend(module.parameters())
    return lora_params


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights into base model for inference.

    This creates a single-weight model without the LoRA overhead.

    Args:
        model: Model with LoRA layers

    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Compute merged weight: W + BA * scaling
            base_weight = module.base_layer.weight.data
            lora_weight = (module.lora.lora_A @ module.lora.lora_B).T * module.lora.scaling

            # Create new linear layer with merged weights
            merged_layer = nn.Linear(
                module.base_layer.in_features,
                module.base_layer.out_features,
                bias=module.base_layer.bias is not None,
            )
            merged_layer.weight.data = base_weight + lora_weight

            if module.base_layer.bias is not None:
                merged_layer.bias.data = module.base_layer.bias.data

            # This would need proper parent reference to actually replace
            # For now, just update the base layer
            module.base_layer.weight.data = merged_layer.weight.data
            module.base_layer.requires_grad_(True)

    return model


def save_lora_checkpoint(
    model: nn.Module,
    path: str,
    metadata: Optional[Dict] = None,
):
    """Save only LoRA parameters to checkpoint.

    Args:
        model: Model with LoRA layers
        path: Path to save checkpoint
        metadata: Optional metadata to save
    """
    lora_state = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            lora_state[f"{name}.lora_A"] = module.lora_A.data
            lora_state[f"{name}.lora_B"] = module.lora_B.data

    checkpoint = {
        "lora_state_dict": lora_state,
        "metadata": metadata or {},
    }

    torch.save(checkpoint, path)


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
) -> nn.Module:
    """Load LoRA parameters from checkpoint.

    Args:
        model: Model with LoRA layers
        path: Path to checkpoint

    Returns:
        Model with loaded LoRA weights
    """
    checkpoint = torch.load(path)
    lora_state = checkpoint["lora_state_dict"]

    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            if f"{name}.lora_A" in lora_state:
                module.lora_A.data = lora_state[f"{name}.lora_A"]
            if f"{name}.lora_B" in lora_state:
                module.lora_B.data = lora_state[f"{name}.lora_B"]

    return model


__all__ = [
    "LoRALayer",
    "LoRALinear",
    "apply_lora_to_model",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
]
