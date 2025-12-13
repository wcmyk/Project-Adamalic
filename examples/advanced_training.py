"""Example: Advanced training with all features enabled."""
from LILITH import (
    AdvancedModelConfig,
    AdvancedTrainingConfig,
    train_advanced,
    get_small_model_config,
    BPETokenizer,
)

# Sample corpus (larger for BPE)
corpus = [
    "The quick brown fox jumps over the lazy dog. " * 10,
    "Machine learning is fascinating and powerful technology. " * 10,
    "Python programming language is widely used for AI. " * 10,
    "Deep neural networks learn hierarchical representations. " * 10,
    "Transformers revolutionized natural language processing. " * 10,
]

# Use preset small model configuration
model_config = get_small_model_config()

# Advanced training configuration with all features
train_config = AdvancedTrainingConfig(
    batch_size=8,
    max_steps=1000,
    lr=3e-4,
    weight_decay=0.01,
    warmup_steps=100,
    grad_clip=1.0,
    log_interval=100,
    device="cuda",
    # Advanced features
    use_mixed_precision=True,
    gradient_accumulation_steps=2,
    lr_scheduler="cosine",
    min_lr=1e-6,
    # Evaluation and checkpointing
    eval_interval=200,
    save_interval=500,
    checkpoint_dir="checkpoints/advanced",
    keep_last_n_checkpoints=3,
    # Early stopping
    use_early_stopping=True,
    early_stopping_patience=3,
    early_stopping_min_delta=0.001,
    # Logging
    log_dir="logs",
    # Data
    validation_split=0.1,
    shuffle=True,
)

print("Training advanced LILITH model...")
print("Features enabled:")
print("  - Mixed precision training (AMP)")
print("  - Gradient accumulation")
print("  - Early stopping")
print("  - Validation set")
print("  - Regular checkpointing")
print()

model = train_advanced(
    corpus=corpus,
    model_config=model_config,
    train_config=train_config,
    block_size=128,
    checkpoint_path="checkpoints/advanced/final_model.pt",
)

print("\nTraining complete!")
print(f"Model parameters: {model.count_parameters():,}")
