"""Lightweight GPT-style language model components for experimentation."""

from .config import ModelConfig, TrainingConfig
from .data import CharacterTokenizer, TextDataset
from .model import GPTDecoder
from .system import AngelicMultiLLM, LLMProfile, Route
from .train import train

# Advanced features
from .sampling import (
    top_k_sampling,
    top_p_sampling,
    beam_search,
    contrastive_search,
    sample_with_strategy,
)
from .kv_cache import KVCache, CachedGPTDecoder
from .lora import (
    LoRALayer,
    LoRALinear,
    apply_lora_to_model,
    get_lora_parameters,
    merge_lora_weights,
    save_lora_checkpoint,
    load_lora_checkpoint,
)
from .tokenizer_advanced import BPETokenizer
from .evaluation import (
    calculate_perplexity,
    calculate_accuracy,
    evaluate_model,
    sample_quality_metrics,
    EarlyStopping,
)
from .logger import LILITHLogger, create_logger
from .config_advanced import (
    AdvancedModelConfig,
    AdvancedTrainingConfig,
    LoRAConfig,
    GenerationConfig,
    get_small_model_config,
    get_medium_model_config,
    get_large_model_config,
)
from .train_advanced import train_advanced, create_dataloaders, load_checkpoint

__all__ = [
    # Core
    "ModelConfig",
    "TrainingConfig",
    "CharacterTokenizer",
    "TextDataset",
    "GPTDecoder",
    "AngelicMultiLLM",
    "LLMProfile",
    "Route",
    "train",
    # Sampling
    "top_k_sampling",
    "top_p_sampling",
    "beam_search",
    "contrastive_search",
    "sample_with_strategy",
    # KV Cache
    "KVCache",
    "CachedGPTDecoder",
    # LoRA
    "LoRALayer",
    "LoRALinear",
    "apply_lora_to_model",
    "get_lora_parameters",
    "merge_lora_weights",
    "save_lora_checkpoint",
    "load_lora_checkpoint",
    # Tokenization
    "BPETokenizer",
    # Evaluation
    "calculate_perplexity",
    "calculate_accuracy",
    "evaluate_model",
    "sample_quality_metrics",
    "EarlyStopping",
    # Logging
    "LILITHLogger",
    "create_logger",
    # Advanced Config
    "AdvancedModelConfig",
    "AdvancedTrainingConfig",
    "LoRAConfig",
    "GenerationConfig",
    "get_small_model_config",
    "get_medium_model_config",
    "get_large_model_config",
    # Advanced Training
    "train_advanced",
    "create_dataloaders",
    "load_checkpoint",
]
