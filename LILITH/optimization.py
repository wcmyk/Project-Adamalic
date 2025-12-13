"""Advanced optimization utilities for training."""
from __future__ import annotations

from typing import List, Tuple
import torch
import torch.nn as nn
from torch.optim import Optimizer


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 0.1,
    no_decay_bias: bool = True,
    no_decay_norm: bool = True,
) -> List[dict]:
    """Create optimized parameter groups for better training.

    Separates parameters that should/shouldn't have weight decay.
    Following best practices from GPT-3, LLaMA, and other modern LLMs.

    Args:
        model: The model
        weight_decay: Weight decay value for applicable parameters
        no_decay_bias: Don't apply weight decay to bias parameters
        no_decay_norm: Don't apply weight decay to normalization parameters

    Returns:
        List of parameter groups for optimizer
    """
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Check if this parameter should have no decay
        should_not_decay = False

        if no_decay_bias and 'bias' in name:
            should_not_decay = True

        if no_decay_norm and ('norm' in name.lower() or 'ln' in name.lower()):
            should_not_decay = True

        # 1D parameters (biases, norms) typically don't need decay
        if param.ndim == 1:
            should_not_decay = True

        if should_not_decay:
            no_decay.append(param)
        else:
            decay.append(param)

    return [
        {
            'params': decay,
            'weight_decay': weight_decay,
        },
        {
            'params': no_decay,
            'weight_decay': 0.0,
        },
    ]


def get_layer_wise_lr_decay_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float = 0.1,
    layer_decay: float = 0.95,
) -> List[dict]:
    """Create parameter groups with layer-wise learning rate decay.

    Layers closer to output get higher learning rates.
    Technique from ELECTRA and other models.

    Args:
        model: The model
        lr: Base learning rate
        weight_decay: Weight decay
        layer_decay: Decay factor per layer (0.95 = 5% reduction per layer)

    Returns:
        List of parameter groups
    """
    # Get number of layers
    num_layers = None
    for name, _ in model.named_parameters():
        if 'transformer.layers' in name or 'layers.' in name:
            # Extract layer number
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                        if num_layers is None or layer_num > num_layers:
                            num_layers = layer_num
                    except ValueError:
                        pass

    if num_layers is None:
        # Fallback to regular parameter groups
        return get_parameter_groups(model, weight_decay)

    # Create groups for each layer
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine layer number
        layer_num = num_layers  # Default to max (output layers)

        if 'embed' in name:
            layer_num = 0  # Embeddings get lowest LR
        elif 'transformer.layers' in name or 'layers.' in name:
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts):
                    try:
                        layer_num = int(parts[i + 1])
                    except ValueError:
                        pass

        # Calculate LR for this layer
        layer_lr = lr * (layer_decay ** (num_layers - layer_num))

        # Create group key
        group_key = (layer_num, layer_lr)

        if group_key not in groups:
            groups[group_key] = {
                'params': [],
                'lr': layer_lr,
                'weight_decay': weight_decay if param.ndim > 1 else 0.0,
            }

        groups[group_key]['params'].append(param)

    return list(groups.values())


class GradientClipping:
    """Gradient clipping utilities."""

    @staticmethod
    def clip_grad_norm(
        parameters,
        max_norm: float,
        norm_type: float = 2.0,
    ) -> torch.Tensor:
        """Clip gradient norm (wrapper around torch function).

        Args:
            parameters: Model parameters
            max_norm: Maximum norm
            norm_type: Type of norm (default: L2)

        Returns:
            Total norm of gradients
        """
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type)

    @staticmethod
    def clip_grad_value(
        parameters,
        clip_value: float,
    ):
        """Clip gradient values to range [-clip_value, clip_value].

        Args:
            parameters: Model parameters
            clip_value: Clipping value
        """
        torch.nn.utils.clip_grad_value_(parameters, clip_value)


class WarmupScheduler:
    """Learning rate warmup scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        base_lr: float,
    ):
        """Initialize warmup scheduler.

        Args:
            optimizer: Optimizer
            warmup_steps: Number of warmup steps
            base_lr: Base learning rate to reach
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.current_step = 0

    def step(self):
        """Update learning rate."""
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            lr_scale = self.current_step / max(1, self.warmup_steps)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.base_lr * lr_scale


def compute_gradient_stats(model: nn.Module) -> dict:
    """Compute statistics about gradients for monitoring.

    Args:
        model: The model

    Returns:
        Dictionary with gradient statistics
    """
    total_norm = 0.0
    max_grad = 0.0
    min_grad = float('inf')
    num_zero_grads = 0
    total_params = 0

    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_grad = max(max_grad, param.grad.data.abs().max().item())
            min_grad = min(min_grad, param.grad.data.abs().min().item())

            if param.grad.data.abs().max().item() == 0:
                num_zero_grads += 1

            total_params += 1

    total_norm = total_norm ** 0.5

    return {
        'grad_norm': total_norm,
        'max_grad': max_grad,
        'min_grad': min_grad if min_grad != float('inf') else 0.0,
        'num_zero_grads': num_zero_grads,
        'total_params_with_grad': total_params,
    }


__all__ = [
    "get_parameter_groups",
    "get_layer_wise_lr_decay_groups",
    "GradientClipping",
    "WarmupScheduler",
    "compute_gradient_stats",
]
