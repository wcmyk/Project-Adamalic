"""Lightweight training loop for the GPTDecoder."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import ModelConfig, TrainingConfig
from .data import CharacterTokenizer, TextDataset
from .model import GPTDecoder


def _warmup_cosine_lr(step: int, warmup_steps: int, max_steps: int) -> float:
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def train(
    corpus: Iterable[str],
    model_config: ModelConfig,
    train_config: TrainingConfig,
    block_size: int = 128,
    checkpoint_path: str | Path | None = None,
) -> GPTDecoder:
    """Instantiate and train a GPTDecoder on a character corpus."""

    tokenizer = CharacterTokenizer(corpus)
    encoded = [tid for text in corpus for tid in tokenizer.encode(text)]
    dataset = TextDataset(tokens=encoded, block_size=block_size)
    loader = DataLoader(dataset, batch_size=train_config.batch_size, shuffle=True)

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    model = GPTDecoder(model_config).to(device)

    optimizer = AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_lr(step, train_config.warmup_steps, train_config.max_steps),
    )

    global_step = 0
    model.train()
    for epoch in range(10_000):  # loop until max_steps is reached
        for x, y in loader:
            if global_step >= train_config.max_steps:
                break
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = model.loss(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=train_config.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % train_config.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(f"step={global_step} loss={loss.item():.4f} lr={lr:.6f}")
        if global_step >= train_config.max_steps:
            break

    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "tokenizer_vocab": tokenizer.itos}, checkpoint_path)

    return model

