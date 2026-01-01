"""Minimal text utilities for character-level language modeling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset


class CharacterTokenizer:
    """Simple character-level tokenizer with deterministic vocab."""

    def __init__(self, corpus: Iterable[str]):
        vocab = sorted({ch for text in corpus for ch in text})
        self.itos: List[str] = vocab
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids)


@dataclass
class TextDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    """Windowed next-token dataset built from a single text corpus."""

    tokens: List[int]
    block_size: int

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(
            self.tokens[idx:idx + self.block_size], dtype=torch.long
        )
        y = torch.tensor(
            self.tokens[idx + 1:idx + 1 + self.block_size],
            dtype=torch.long
        )
        return x, y
