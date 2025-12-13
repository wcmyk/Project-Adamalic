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
from .config_phase2 import (
    Phase2ModelConfig,
    Phase2TrainingConfig,
    LoRAConfig,
    GenerationConfig,
    get_small_model_config,
    get_medium_model_config,
    get_large_model_config,
    get_1b_model_config,
    get_code_training_config,
)
from .train_phase2 import train_phase2, create_dataloaders, load_checkpoint

# Production features
from .datasets import (
    StreamingTextDataset,
    WikipediaDataset,
    CodeDataset,
    CombinedDataset,
    LocalTextDataset,
    get_wikipedia_corpus,
    get_code_corpus,
)
from .optimization import (
    get_parameter_groups,
    get_layer_wise_lr_decay_groups,
    GradientClipping,
    compute_gradient_stats,
)
from .positional import RotaryEmbedding, ALiBi, apply_rotary_pos_emb
from .activations import SwiGLU, GeGLU, ReGLU, get_activation
from .quantization import (
    quantize_dynamic,
    quantize_to_int8,
    quantize_to_float16,
    QuantizedGPTDecoder,
)
from .serve import load_model_from_checkpoint

# Phase 3: Agent & Instruction Following
from .instruction_data import (
    InstructionExample,
    InstructionDataset,
    ConversationDataset,
    ToolUseDataset,
    SyntheticInstructionGenerator,
    load_instruction_dataset,
    SYSTEM_PROMPTS,
)
from .tools import (
    Tool,
    ToolParameter,
    ToolCall,
    ToolResult,
    ToolRegistry,
    create_default_tools,
    parse_tool_calls_from_text,
    format_tool_result_for_prompt,
)
from .agent import (
    Message,
    AgentConfig,
    LILITHAgent,
    create_assistant_agent,
)
from .rlhf import (
    PreferencePair,
    PreferenceDataset,
    RewardModel,
    train_reward_model,
    PPOTrainer,
    create_preference_dataset_from_comparisons,
    generate_synthetic_preferences,
)

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
    # Phase 2 Config
    "Phase2ModelConfig",
    "Phase2TrainingConfig",
    "LoRAConfig",
    "GenerationConfig",
    "get_small_model_config",
    "get_medium_model_config",
    "get_large_model_config",
    "get_1b_model_config",
    "get_code_training_config",
    # Phase 2 Training
    "train_phase2",
    "create_dataloaders",
    "load_checkpoint",
    # Production Datasets
    "StreamingTextDataset",
    "WikipediaDataset",
    "CodeDataset",
    "CombinedDataset",
    "LocalTextDataset",
    "get_wikipedia_corpus",
    "get_code_corpus",
    # Optimization
    "get_parameter_groups",
    "get_layer_wise_lr_decay_groups",
    "GradientClipping",
    "compute_gradient_stats",
    # Positional Encodings
    "RotaryEmbedding",
    "ALiBi",
    "apply_rotary_pos_emb",
    # Activations
    "SwiGLU",
    "GeGLU",
    "ReGLU",
    "get_activation",
    # Quantization
    "quantize_dynamic",
    "quantize_to_int8",
    "quantize_to_float16",
    "QuantizedGPTDecoder",
    # Serving
    "load_model_from_checkpoint",
    # Phase 3: Instruction Data
    "InstructionExample",
    "InstructionDataset",
    "ConversationDataset",
    "ToolUseDataset",
    "SyntheticInstructionGenerator",
    "load_instruction_dataset",
    "SYSTEM_PROMPTS",
    # Phase 3: Tools
    "Tool",
    "ToolParameter",
    "ToolCall",
    "ToolResult",
    "ToolRegistry",
    "create_default_tools",
    "parse_tool_calls_from_text",
    "format_tool_result_for_prompt",
    # Phase 3: Agent
    "Message",
    "AgentConfig",
    "LILITHAgent",
    "create_assistant_agent",
    # Phase 3: RLHF
    "PreferencePair",
    "PreferenceDataset",
    "RewardModel",
    "train_reward_model",
    "PPOTrainer",
    "create_preference_dataset_from_comparisons",
    "generate_synthetic_preferences",
]
