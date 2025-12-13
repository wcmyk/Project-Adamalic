"""Advanced configuration with comprehensive training options."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class AdvancedModelConfig:
    """Extended model configuration with advanced features."""

    vocab_size: int
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    max_seq_len: int = 256
    dropout: float = 0.1

    # Advanced features
    use_flash_attention: bool = False
    use_gradient_checkpointing: bool = False
    tie_word_embeddings: bool = True  # Share input/output embeddings
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.1
    residual_dropout: float = 0.1

    # Initialization
    init_std: float = 0.02
    initializer_range: float = 0.02


@dataclass
class AdvancedTrainingConfig:
    """Extended training configuration with advanced features."""

    # Basic training
    batch_size: int = 32
    max_steps: int = 2_000
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0
    log_interval: int = 50
    device: str = "cuda"

    # Advanced optimization
    use_mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    lr_scheduler: Literal["cosine", "linear", "constant"] = "cosine"
    min_lr: float = 1e-6

    # Evaluation and checkpointing
    eval_interval: int = 500
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    keep_last_n_checkpoints: int = 3

    # Early stopping
    use_early_stopping: bool = False
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001

    # Logging
    use_tensorboard: bool = False
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    log_dir: str = "logs"

    # Data
    validation_split: float = 0.1
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = True

    # Distributed training
    use_ddp: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    dist_backend: str = "nccl"


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""

    rank: int = 4
    alpha: float = 1.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["head"])
    merge_weights: bool = False


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    max_new_tokens: int = 64
    temperature: float = 1.0
    strategy: Literal["greedy", "temperature", "top_k", "top_p", "beam", "contrastive"] = "temperature"

    # Strategy-specific parameters
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    beam_width: int = 4
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0

    # Sampling control
    do_sample: bool = True
    num_return_sequences: int = 1

    # Special tokens
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None

    # Cache
    use_cache: bool = True


def get_small_model_config() -> AdvancedModelConfig:
    """Get configuration for small model (~10M params)."""
    return AdvancedModelConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.1,
    )


def get_medium_model_config() -> AdvancedModelConfig:
    """Get configuration for medium model (~50M params)."""
    return AdvancedModelConfig(
        vocab_size=5000,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
    )


def get_large_model_config() -> AdvancedModelConfig:
    """Get configuration for large model (~150M params)."""
    return AdvancedModelConfig(
        vocab_size=10000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1,
        use_gradient_checkpointing=True,
    )


__all__ = [
    "AdvancedModelConfig",
    "AdvancedTrainingConfig",
    "LoRAConfig",
    "GenerationConfig",
    "get_small_model_config",
    "get_medium_model_config",
    "get_large_model_config",
]
