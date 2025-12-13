"""Phase 2 configuration with production-ready training options."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class Phase2ModelConfig:
    """Extended model configuration with Phase 2 features."""

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
class Phase2TrainingConfig:
    """Extended training configuration with Phase 2 features."""

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


def get_small_model_config() -> Phase2ModelConfig:
    """Get configuration for small model (~10M params)."""
    return Phase2ModelConfig(
        vocab_size=1000,
        d_model=256,
        n_layers=4,
        n_heads=4,
        d_ff=1024,
        max_seq_len=256,
        dropout=0.1,
    )


def get_medium_model_config() -> Phase2ModelConfig:
    """Get configuration for medium model (~50M params)."""
    return Phase2ModelConfig(
        vocab_size=5000,
        d_model=512,
        n_layers=8,
        n_heads=8,
        d_ff=2048,
        max_seq_len=512,
        dropout=0.1,
    )


def get_large_model_config() -> Phase2ModelConfig:
    """Get configuration for large model (~150M params)."""
    return Phase2ModelConfig(
        vocab_size=10000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1,
        use_gradient_checkpointing=True,
    )


def get_1b_model_config() -> Phase2ModelConfig:
    """Get configuration for 1 billion parameter model.

    Architecture optimized for code generation:
    - 24 transformer layers
    - 1792 hidden dimensions (112 per head)
    - 16 attention heads
    - 7168 FFN dimensions (4x hidden)
    - 50k vocabulary (optimal for code + text with BPE)
    - 2048 token context (handles longer code files)

    Estimated parameters: ~1.02B
    - Embeddings: 50k * 1792 = 89.6M
    - Transformer layers: 24 * 38M = 912M
    - Position embeddings: 2048 * 1792 = 3.7M
    - Output layer: shared with embeddings

    Memory requirements:
    - Model weights (FP32): ~4GB
    - Model weights (FP16): ~2GB
    - Training (batch_size=1, FP16 + grad checkpointing): ~8-10GB VRAM
    - Training (batch_size=4, FP16 + grad checkpointing): ~24-28GB VRAM
    - Recommended: 40GB+ GPU (A100) or multi-GPU setup
    """
    return Phase2ModelConfig(
        vocab_size=50000,
        d_model=1792,
        n_layers=24,
        n_heads=16,
        d_ff=7168,  # 4x d_model
        max_seq_len=2048,
        dropout=0.1,
        use_gradient_checkpointing=True,  # Essential for memory efficiency
    )


def get_7b_model_config() -> Phase2ModelConfig:
    """Get configuration for 7 billion parameter model (LLaMA-scale).

    Architecture optimized for Claude-level performance:
    - 32 transformer layers (deeper reasoning)
    - 4096 hidden dimensions
    - 32 attention heads (128 per head)
    - 16384 FFN dimensions (4x hidden)
    - 50k vocabulary
    - 4096 token context (long-context reasoning)

    Estimated parameters: ~6.7B
    - Embeddings: 50k * 4096 = 204.8M
    - Transformer layers: 32 * 201M = 6.4B
    - Position embeddings: 4096 * 4096 = 16.8M

    Memory requirements:
    - Model weights (FP32): ~27GB
    - Model weights (FP16): ~13.5GB
    - Training (FP16 + grad checkpointing, batch=1): ~30-35GB VRAM
    - Training (FP16 + grad checkpointing, batch=2): ~50-60GB VRAM
    - Recommended: 8x A100 80GB or A100 80GB with DeepSpeed ZeRO-3

    This scale approaches LLaMA-7B, Mistral-7B capabilities.
    """
    return Phase2ModelConfig(
        vocab_size=50000,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        d_ff=16384,  # 4x d_model (SwiGLU uses 4x instead of standard 4x)
        max_seq_len=4096,
        dropout=0.0,  # Large models use less dropout
        use_gradient_checkpointing=True,
    )


def get_13b_model_config() -> Phase2ModelConfig:
    """Get configuration for 13 billion parameter model (approaching Claude-scale).

    Architecture optimized for maximum capability:
    - 40 transformer layers (very deep reasoning)
    - 5120 hidden dimensions
    - 40 attention heads (128 per head)
    - 20480 FFN dimensions (4x hidden)
    - 50k vocabulary
    - 4096 token context

    Estimated parameters: ~13.0B
    - Embeddings: 50k * 5120 = 256M
    - Transformer layers: 40 * 315M = 12.6B
    - Position embeddings: 4096 * 5120 = 21M

    Memory requirements:
    - Model weights (FP32): ~52GB
    - Model weights (FP16): ~26GB
    - Training (FP16 + grad checkpointing): 60-80GB VRAM per GPU
    - Recommended: 8x A100 80GB with DeepSpeed ZeRO-3 or model parallelism

    This scale approaches LLaMA-13B, approaching GPT-3.5 level capabilities.

    Note: Training 13B requires serious infrastructure (~$15-30k cost).
    Consider starting with 1B or 7B first.
    """
    return Phase2ModelConfig(
        vocab_size=50000,
        d_model=5120,
        n_layers=40,
        n_heads=40,
        d_ff=20480,
        max_seq_len=4096,
        dropout=0.0,
        use_gradient_checkpointing=True,
    )


def get_code_training_config(
    num_gpus: int = 1,
    total_batch_size: int = 512,
) -> Phase2TrainingConfig:
    """Get optimized training configuration for code models.

    This configuration is optimized for training on code with large models:
    - Uses mixed precision (FP16/BF16) for 2x speedup
    - Gradient accumulation to achieve large effective batch sizes
    - Cosine learning rate schedule with warmup
    - Early stopping to prevent overfitting

    Args:
        num_gpus: Number of GPUs available
        total_batch_size: Desired effective batch size (will use gradient accumulation)

    Returns:
        Optimized Phase2TrainingConfig
    """
    # Calculate gradient accumulation steps
    # For 1B model, per_gpu_batch_size is typically 1-4 depending on VRAM
    per_gpu_batch_size = 2 if num_gpus >= 4 else 1
    batch_size = per_gpu_batch_size * num_gpus
    gradient_accumulation_steps = max(1, total_batch_size // batch_size)

    return Phase2TrainingConfig(
        batch_size=per_gpu_batch_size,
        num_epochs=3,
        learning_rate=2e-4,  # Slightly lower for large models
        warmup_steps=2000,
        weight_decay=0.1,
        max_grad_norm=1.0,
        log_interval=100,
        save_interval=5000,

        # Performance optimizations
        use_mixed_precision=True,
        gradient_accumulation_steps=gradient_accumulation_steps,
        lr_scheduler="cosine",

        # Validation and early stopping
        use_early_stopping=True,
        validation_split=0.01,  # 1% for validation (code datasets are large)
        early_stopping_patience=3,
    )


__all__ = [
    "Phase2ModelConfig",
    "Phase2TrainingConfig",
    "LoRAConfig",
    "GenerationConfig",
    "get_small_model_config",
    "get_medium_model_config",
    "get_large_model_config",
    "get_1b_model_config",
    "get_7b_model_config",
    "get_13b_model_config",
    "get_code_training_config",
]
