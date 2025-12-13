"""Example: Fine-tuning with LoRA (Low-Rank Adaptation)."""
import torch
from LILITH import (
    ModelConfig,
    GPTDecoder,
    apply_lora_to_model,
    get_lora_parameters,
    save_lora_checkpoint,
    load_lora_checkpoint,
)

# Create a base model
print("Creating base model...")
config = ModelConfig(
    vocab_size=100,
    d_model=256,
    n_layers=4,
    n_heads=4,
)

base_model = GPTDecoder(config)
initial_params = base_model.count_parameters()
print(f"Base model parameters: {initial_params:,}")

# Apply LoRA to the model
print("\nApplying LoRA adaptation...")
lora_model = apply_lora_to_model(
    base_model,
    target_modules=["head"],  # Only adapt the output head
    rank=4,
    alpha=1.0,
    dropout=0.0,
)

# Count trainable parameters
lora_params = get_lora_parameters(lora_model)
trainable_params = sum(p.numel() for p in lora_params)
print(f"LoRA trainable parameters: {trainable_params:,}")
print(f"Percentage of base: {100 * trainable_params / initial_params:.2f}%")

# Demonstrate that only LoRA parameters have gradients
print("\nChecking parameter requires_grad status:")
for name, param in lora_model.named_parameters():
    if param.requires_grad:
        print(f"  ✓ {name}: trainable ({param.numel():,} params)")

# Save LoRA checkpoint (much smaller than full model)
print("\nSaving LoRA checkpoint...")
save_lora_checkpoint(
    lora_model,
    "checkpoints/lora_adapter.pt",
    metadata={"rank": 4, "alpha": 1.0, "target": "head"}
)
print("  Saved to checkpoints/lora_adapter.pt")

# Load LoRA checkpoint
print("\nLoading LoRA checkpoint...")
lora_model_loaded = apply_lora_to_model(
    GPTDecoder(config),
    target_modules=["head"],
    rank=4,
    alpha=1.0,
)
load_lora_checkpoint(lora_model_loaded, "checkpoints/lora_adapter.pt")
print("  ✓ Loaded successfully")

print("\nLoRA fine-tuning allows efficient adaptation with minimal parameters!")
