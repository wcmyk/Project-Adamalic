"""Advanced training loop with mixed precision, gradient accumulation, and more."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler

from .config import ModelConfig, TrainingConfig
from .data import CharacterTokenizer, TextDataset
from .model import GPTDecoder
from .logger import create_logger
from .evaluation import calculate_perplexity, EarlyStopping


def _get_lr_lambda(scheduler_type: str, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1):
    """Get learning rate lambda function based on scheduler type."""

    def warmup_cosine(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    def warmup_linear(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(min_lr_ratio, (max_steps - step) / max(1, max_steps - warmup_steps))

    def constant_with_warmup(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    schedulers = {
        "cosine": warmup_cosine,
        "linear": warmup_linear,
        "constant": constant_with_warmup,
    }

    return schedulers.get(scheduler_type, warmup_cosine)


def create_dataloaders(
    corpus: Iterable[str],
    tokenizer: CharacterTokenizer,
    train_config: TrainingConfig,
    block_size: int = 128,
    validation_split: float = 0.1,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train and validation dataloaders.

    Args:
        corpus: Text corpus
        tokenizer: Tokenizer
        train_config: Training configuration
        block_size: Sequence length
        validation_split: Fraction of data for validation

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Encode corpus
    encoded = [tid for text in corpus for tid in tokenizer.encode(text)]
    dataset = TextDataset(tokens=encoded, block_size=block_size)

    # Split into train and validation
    if validation_split > 0:
        val_size = int(len(dataset) * validation_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_config.batch_size,
            shuffle=train_config.shuffle,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=train_config.batch_size,
            shuffle=False,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=train_config.shuffle,
            num_workers=train_config.num_workers,
            pin_memory=train_config.pin_memory,
        )
        return train_loader, None


def train_advanced(
    corpus: Iterable[str],
    model_config: ModelConfig,
    train_config: TrainingConfig,
    block_size: int = 128,
    checkpoint_path: Optional[str | Path] = None,
) -> GPTDecoder:
    """Advanced training loop with full features.

    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Validation and early stopping
    - Flexible LR schedulers
    - Comprehensive logging
    - Regular checkpointing
    - Gradient checkpointing

    Args:
        corpus: Training corpus
        model_config: Model configuration
        train_config: Training configuration
        block_size: Sequence block size
        checkpoint_path: Optional path to save final checkpoint

    Returns:
        Trained model
    """
    # Create logger
    logger = create_logger(name="LILITH-Train", log_dir=train_config.log_dir)

    # Create tokenizer and dataloaders
    logger.info("Creating tokenizer and dataloaders...")
    tokenizer = CharacterTokenizer(corpus)

    train_loader, val_loader = create_dataloaders(
        corpus=corpus,
        tokenizer=tokenizer,
        train_config=train_config,
        block_size=block_size,
        validation_split=train_config.validation_split,
    )

    # Create model
    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    model = GPTDecoder(
        model_config,
        use_gradient_checkpointing=getattr(model_config, 'use_gradient_checkpointing', False)
    ).to(device)

    logger.log_model_info(model, model_config)

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )

    # Create learning rate scheduler
    min_lr_ratio = train_config.min_lr / train_config.lr
    lr_lambda = _get_lr_lambda(
        train_config.lr_scheduler,
        train_config.warmup_steps,
        train_config.max_steps,
        min_lr_ratio,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision scaler
    scaler = GradScaler() if train_config.use_mixed_precision else None

    # Early stopping
    early_stopping = None
    if train_config.use_early_stopping and val_loader is not None:
        early_stopping = EarlyStopping(
            patience=train_config.early_stopping_patience,
            min_delta=train_config.early_stopping_min_delta,
            mode="min",  # Minimize validation loss
        )

    # Checkpoint directory
    if checkpoint_path:
        checkpoint_dir = Path(train_config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    model.train()

    logger.info("Starting training...")
    logger.info(f"Total training steps: {train_config.max_steps}")
    logger.info(f"Gradient accumulation steps: {train_config.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {train_config.batch_size * train_config.gradient_accumulation_steps}")

    for epoch in range(10_000):  # Large number, will break when max_steps reached
        for batch_idx, (x, y) in enumerate(train_loader):
            if global_step >= train_config.max_steps:
                break

            x = x.to(device)
            y = y.to(device)

            # Forward pass with mixed precision
            if scaler is not None:
                with autocast():
                    logits = model(x)
                    loss = model.loss(logits, y)
                    loss = loss / train_config.gradient_accumulation_steps

                # Backward pass
                scaler.scale(loss).backward()
            else:
                logits = model(x)
                loss = model.loss(logits, y)
                loss = loss / train_config.gradient_accumulation_steps
                loss.backward()

            # Update weights (with gradient accumulation)
            if (batch_idx + 1) % train_config.gradient_accumulation_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                # Logging
                if global_step % train_config.log_interval == 0:
                    lr = scheduler.get_last_lr()[0]
                    logger.log_training_step(
                        step=global_step,
                        loss=loss.item() * train_config.gradient_accumulation_steps,
                        lr=lr,
                    )

                # Validation
                if val_loader is not None and global_step % train_config.eval_interval == 0:
                    model.eval()
                    val_perplexity = calculate_perplexity(model, val_loader, device)
                    val_loss = math.log(val_perplexity)  # Approximate loss from perplexity

                    logger.log_evaluation(
                        {"perplexity": val_perplexity, "loss": val_loss},
                        prefix="validation"
                    )

                    # Early stopping check
                    if early_stopping is not None:
                        if early_stopping(val_loss):
                            logger.info(f"Early stopping triggered at step {global_step}")
                            break

                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        if checkpoint_path:
                            best_path = checkpoint_dir / "best_model.pt"
                            torch.save({
                                "model_state_dict": model.state_dict(),
                                "tokenizer_vocab": tokenizer.itos,
                                "step": global_step,
                                "val_loss": val_loss,
                            }, best_path)
                            logger.info(f"Saved best model to {best_path}")

                    model.train()

                # Regular checkpointing
                if checkpoint_path and global_step % train_config.save_interval == 0:
                    ckpt_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "tokenizer_vocab": tokenizer.itos,
                        "step": global_step,
                        "config": model_config,
                    }, ckpt_path)
                    logger.info(f"Saved checkpoint to {ckpt_path}")

                    # Clean up old checkpoints
                    _cleanup_old_checkpoints(checkpoint_dir, train_config.keep_last_n_checkpoints)

        if global_step >= train_config.max_steps:
            break

        if early_stopping is not None and early_stopping.early_stop:
            break

    # Save final checkpoint
    if checkpoint_path is not None:
        final_path = Path(checkpoint_path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "tokenizer_vocab": tokenizer.itos,
            "step": global_step,
            "config": model_config,
        }, final_path)
        logger.info(f"Saved final model to {final_path}")

    logger.info("Training completed!")
    return model


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int):
    """Keep only the last N checkpoints."""
    checkpoints = sorted(
        checkpoint_dir.glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )

    # Remove older checkpoints
    for ckpt in checkpoints[:-keep_last_n]:
        ckpt.unlink()


def load_checkpoint(
    checkpoint_path: str | Path,
    device: Optional[torch.device] = None,
) -> Tuple[GPTDecoder, CharacterTokenizer, dict]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, metadata)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Reconstruct model
    config = checkpoint["config"]
    model = GPTDecoder(config)
    model.load_state_dict(checkpoint["model_state_dict"])

    if device is not None:
        model = model.to(device)

    # Reconstruct tokenizer (simplified - would need full vocab rebuild)
    tokenizer = CharacterTokenizer([])  # Placeholder
    tokenizer.itos = checkpoint["tokenizer_vocab"]
    tokenizer.stoi = {ch: idx for idx, ch in enumerate(tokenizer.itos)}

    # Metadata
    metadata = {
        "step": checkpoint.get("step", 0),
        "val_loss": checkpoint.get("val_loss", None),
    }

    return model, tokenizer, metadata


__all__ = [
    "train_advanced",
    "create_dataloaders",
    "load_checkpoint",
]
