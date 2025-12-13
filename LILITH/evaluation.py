"""Evaluation utilities for language models."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math


def calculate_perplexity(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> float:
    """Calculate perplexity on a dataset.

    Perplexity is exp(average_loss), a standard metric for language models.
    Lower is better.

    Args:
        model: The language model
        dataloader: DataLoader for evaluation data
        device: Device to run on

    Returns:
        Perplexity score
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = model.loss(logits, y)

            # Accumulate loss weighted by number of tokens
            batch_tokens = y.numel()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    return perplexity


def calculate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
    top_k: int = 1,
) -> Tuple[float, float]:
    """Calculate next-token prediction accuracy.

    Args:
        model: The language model
        dataloader: DataLoader for evaluation data
        device: Device to run on
        top_k: Consider prediction correct if target in top-k predictions

    Returns:
        Tuple of (accuracy, total_tokens)
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            # Get top-k predictions
            _, top_indices = torch.topk(logits, k=top_k, dim=-1)

            # Check if target is in top-k
            y_expanded = y.unsqueeze(-1).expand_as(top_indices)
            matches = (top_indices == y_expanded).any(dim=-1)

            correct += matches.sum().item()
            total += y.numel()

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, total


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: Optional[torch.device] = None,
) -> dict:
    """Comprehensive model evaluation.

    Args:
        model: The language model
        dataloader: DataLoader for evaluation data
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    perplexity = calculate_perplexity(model, dataloader, device)
    top1_acc, total_tokens = calculate_accuracy(model, dataloader, device, top_k=1)
    top5_acc, _ = calculate_accuracy(model, dataloader, device, top_k=5)

    return {
        "perplexity": perplexity,
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "total_tokens": total_tokens,
    }


def sample_quality_metrics(
    model: nn.Module,
    prompts: list[str],
    tokenizer,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> dict:
    """Evaluate generation quality with various metrics.

    Args:
        model: The language model
        prompts: List of prompts to generate from
        tokenizer: Tokenizer for encoding/decoding
        max_new_tokens: Number of tokens to generate
        temperature: Sampling temperature

    Returns:
        Dictionary of quality metrics
    """
    device = next(model.parameters()).device
    model.eval()

    generations = []
    unique_tokens_per_sample = []
    avg_token_probs = []

    with torch.no_grad():
        for prompt in prompts:
            prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
            generated = model.generate(prompt_ids, max_new_tokens, temperature)
            text = tokenizer.decode(generated[0].tolist())

            generations.append(text)

            # Calculate unique token ratio
            tokens = generated[0].tolist()
            unique_ratio = len(set(tokens)) / len(tokens) if len(tokens) > 0 else 0
            unique_tokens_per_sample.append(unique_ratio)

    return {
        "avg_unique_token_ratio": sum(unique_tokens_per_sample) / len(unique_tokens_per_sample),
        "generations": generations,
    }


class EarlyStopping:
    """Early stopping utility to stop training when validation metric stops improving."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for metrics to minimize (loss), 'max' for metrics to maximize (accuracy)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        """Check if should stop training.

        Args:
            score: Current validation score

        Returns:
            True if should stop training
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "min":
            improved = score < (self.best_score - self.min_delta)
        else:  # mode == "max"
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False


__all__ = [
    "calculate_perplexity",
    "calculate_accuracy",
    "evaluate_model",
    "sample_quality_metrics",
    "EarlyStopping",
]
