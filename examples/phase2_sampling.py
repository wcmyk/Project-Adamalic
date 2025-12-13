"""Example: Advanced text generation with different sampling strategies."""
import torch
from LILITH import (
    ModelConfig,
    GPTDecoder,
    CharacterTokenizer,
    sample_with_strategy,
)

# Create a small model for demo
corpus = ["hello world", "test data", "sample text"]
tokenizer = CharacterTokenizer(corpus)

config = ModelConfig(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    n_layers=2,
    n_heads=2,
    d_ff=256,
    max_seq_len=128,
)

model = GPTDecoder(config)
model.eval()

# Sample prompt
prompt_text = "hello"
prompt_ids = torch.tensor([tokenizer.encode(prompt_text)])

print("Demonstrating different sampling strategies:\n")

# 1. Greedy sampling
print("1. Greedy Sampling (deterministic, picks most likely token):")
generated = sample_with_strategy(
    model, prompt_ids, max_new_tokens=20, strategy="greedy"
)
print(f"   {tokenizer.decode(generated[0].tolist())}\n")

# 2. Temperature sampling
print("2. Temperature Sampling (temperature=0.8):")
generated = sample_with_strategy(
    model, prompt_ids, max_new_tokens=20, strategy="temperature", temperature=0.8
)
print(f"   {tokenizer.decode(generated[0].tolist())}\n")

# 3. Top-k sampling
print("3. Top-K Sampling (k=5):")
generated = sample_with_strategy(
    model, prompt_ids, max_new_tokens=20, strategy="top_k", top_k=5, temperature=1.0
)
print(f"   {tokenizer.decode(generated[0].tolist())}\n")

# 4. Top-p (nucleus) sampling
print("4. Top-P/Nucleus Sampling (p=0.9):")
generated = sample_with_strategy(
    model, prompt_ids, max_new_tokens=20, strategy="top_p", top_p=0.9, temperature=1.0
)
print(f"   {tokenizer.decode(generated[0].tolist())}\n")

# 5. Beam search
print("5. Beam Search (beam_width=4):")
generated = sample_with_strategy(
    model, prompt_ids, max_new_tokens=20, strategy="beam", beam_width=4
)
print(f"   {tokenizer.decode(generated[0].tolist())}\n")

print("Note: Results are random and will vary on each run (except greedy).")
