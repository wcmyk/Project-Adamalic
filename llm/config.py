"""Configuration objects for the GPT-style decoder and training loop."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_steps: int = 2_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_interval: int = 50
    device: str = "cuda"

