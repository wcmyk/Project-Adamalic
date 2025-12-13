"""Model quantization for efficient inference."""
from __future__ import annotations

from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.quantization as quant

from .model import GPTDecoder


def quantize_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply dynamic quantization to model.

    Reduces model size by ~4x with minimal quality loss.
    Best for CPU inference where weights dominate memory.

    Args:
        model: Model to quantize
        dtype: Quantization dtype (qint8 or float16)

    Returns:
        Quantized model
    """
    model.eval()

    quantized_model = quant.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize linear layers
        dtype=dtype,
    )

    return quantized_model


def quantize_static(
    model: nn.Module,
    calibration_dataloader,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """Apply static quantization with calibration.

    More aggressive than dynamic, requires calibration data.
    Can achieve 4x size reduction with good quality.

    Args:
        model: Model to quantize
        calibration_dataloader: DataLoader for calibration
        dtype: Quantization dtype

    Returns:
        Quantized model
    """
    model.eval()

    # Set quantization config
    model.qconfig = quant.get_default_qconfig('fbgemm')

    # Prepare model for quantization
    model_prepared = quant.prepare(model)

    # Calibrate with sample data
    with torch.no_grad():
        for batch in calibration_dataloader:
            if isinstance(batch, (tuple, list)):
                x = batch[0]
            else:
                x = batch
            model_prepared(x)
            break  # One batch is usually enough

    # Convert to quantized model
    quantized_model = quant.convert(model_prepared)

    return quantized_model


def quantize_to_int8(model: GPTDecoder) -> GPTDecoder:
    """Quantize GPTDecoder to INT8.

    Convenience function for LILITH models.

    Args:
        model: GPTDecoder model

    Returns:
        Quantized model
    """
    return quantize_dynamic(model, dtype=torch.qint8)


def quantize_to_float16(model: GPTDecoder) -> GPTDecoder:
    """Convert model to FP16.

    Useful for GPU inference - 2x memory reduction.

    Args:
        model: Model to convert

    Returns:
        FP16 model
    """
    return model.half()


def estimate_size_reduction(original_model: nn.Module, quantized_model: nn.Module) -> dict:
    """Estimate size reduction from quantization.

    Args:
        original_model: Original model
        quantized_model: Quantized model

    Returns:
        Dictionary with size information
    """
    def get_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        return (param_size + buffer_size) / 1024 / 1024  # MB

    original_size = get_model_size(original_model)
    quantized_size = get_model_size(quantized_model)
    reduction = (original_size - quantized_size) / original_size * 100

    return {
        "original_size_mb": original_size,
        "quantized_size_mb": quantized_size,
        "reduction_percent": reduction,
        "compression_ratio": original_size / quantized_size,
    }


def save_quantized_model(
    model: nn.Module,
    path: str | Path,
    include_config: bool = True,
):
    """Save quantized model to disk.

    Args:
        model: Quantized model
        path: Path to save
        include_config: Whether to include model config
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "model_state_dict": model.state_dict(),
        "quantized": True,
    }

    if include_config and hasattr(model, 'config'):
        save_dict["config"] = model.config

    torch.save(save_dict, path)


def load_quantized_model(
    path: str | Path,
    model_class: type = GPTDecoder,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load quantized model from disk.

    Args:
        path: Path to model
        model_class: Model class to instantiate
        device: Device to load on

    Returns:
        Loaded quantized model
    """
    checkpoint = torch.load(path, map_location=device)

    if "config" in checkpoint:
        model = model_class(checkpoint["config"])
    else:
        raise ValueError("Config not found in checkpoint. Cannot instantiate model.")

    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    return model


class QuantizedGPTDecoder(nn.Module):
    """Wrapper for quantized GPTDecoder with convenient interface."""

    def __init__(self, original_model: GPTDecoder, quantization_type: str = "dynamic"):
        """Initialize quantized wrapper.

        Args:
            original_model: Original GPTDecoder
            quantization_type: Type of quantization ('dynamic', 'fp16')
        """
        super().__init__()

        if quantization_type == "dynamic":
            self.model = quantize_to_int8(original_model)
        elif quantization_type == "fp16":
            self.model = quantize_to_float16(original_model)
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")

        self.config = original_model.config
        self.quantization_type = quantization_type

    def forward(self, *args, **kwargs):
        """Forward pass."""
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """Generate text."""
        return self.model.generate(*args, **kwargs)


__all__ = [
    "quantize_dynamic",
    "quantize_static",
    "quantize_to_int8",
    "quantize_to_float16",
    "estimate_size_reduction",
    "save_quantized_model",
    "load_quantized_model",
    "QuantizedGPTDecoder",
]
