"""Example: Train a simple LILITH model on sample text."""
from LILITH import (
    ModelConfig,
    TrainingConfig,
    train,
    CharacterTokenizer,
)

# Sample corpus
corpus = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is fascinating and powerful.",
    "Python is a great programming language.",
    "Artificial intelligence will shape our future.",
    "Deep learning models can learn complex patterns.",
]

# Create model configuration
model_config = ModelConfig(
    vocab_size=100,  # Will be updated by tokenizer
    d_model=128,
    n_layers=4,
    n_heads=4,
    d_ff=512,
    max_seq_len=128,
    dropout=0.1,
)

# Create training configuration
train_config = TrainingConfig(
    batch_size=4,
    max_steps=500,
    lr=3e-4,
    weight_decay=0.01,
    warmup_steps=50,
    grad_clip=1.0,
    log_interval=50,
    device="cuda",  # Change to "cpu" if no GPU
)

# Train the model
print("Training simple LILITH model...")
model = train(
    corpus=corpus,
    model_config=model_config,
    train_config=train_config,
    block_size=64,
    checkpoint_path="checkpoints/simple_model.pt",
)

# Generate some text
print("\nGenerating text...")
tokenizer = CharacterTokenizer(corpus)
import torch

prompt_text = "The"
prompt_ids = torch.tensor([tokenizer.encode(prompt_text)])
generated = model.generate(prompt_ids, max_new_tokens=50, temperature=0.8)
generated_text = tokenizer.decode(generated[0].tolist())

print(f"Prompt: {prompt_text}")
print(f"Generated: {generated_text}")
