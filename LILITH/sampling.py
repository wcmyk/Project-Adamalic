"""Advanced sampling strategies for text generation."""
from __future__ import annotations

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def top_k_sampling(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample from top-k most probable tokens.

    Args:
        logits: Unnormalized log probabilities (batch, vocab_size)
        k: Number of top tokens to consider
        temperature: Sampling temperature

    Returns:
        Sampled token indices (batch, 1)
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Get top-k values and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Create a mask for non-top-k values
    indices_to_remove = logits < top_k_logits[..., -1, None]
    logits = logits.clone()
    logits[indices_to_remove] = float('-inf')

    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def top_p_sampling(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Nucleus sampling: sample from smallest set of tokens with cumulative probability >= p.

    Args:
        logits: Unnormalized log probabilities (batch, vocab_size)
        p: Cumulative probability threshold
        temperature: Sampling temperature

    Returns:
        Sampled token indices (batch, 1)
    """
    if temperature != 1.0:
        logits = logits / temperature

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > p
    # Shift the indices to the right to keep the first token above threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Create mask for original indices
    logits = logits.clone()
    for i in range(logits.size(0)):
        indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
        logits[i, indices_to_remove] = float('-inf')

    # Sample from filtered distribution
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def beam_search(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    beam_width: int = 4,
    temperature: float = 1.0,
    length_penalty: float = 1.0,
) -> Tuple[torch.Tensor, float]:
    """Beam search decoding for more coherent generation.

    Args:
        model: The language model
        prompt: Initial token sequence (1, seq_len)
        max_new_tokens: Maximum tokens to generate
        beam_width: Number of beams to maintain
        temperature: Sampling temperature
        length_penalty: Length normalization penalty (>1 favors longer sequences)

    Returns:
        Best sequence and its score
    """
    device = prompt.device
    batch_size = prompt.size(0)
    assert batch_size == 1, "Beam search currently supports batch_size=1"

    # Initialize beams: (score, sequence)
    beams = [(0.0, prompt)]

    for _ in range(max_new_tokens):
        candidates = []

        for score, seq in beams:
            # Get logits for next token
            with torch.no_grad():
                window = seq[:, -model.config.max_seq_len:]
                logits = model(window)
                next_token_logits = logits[:, -1, :] / temperature
                log_probs = F.log_softmax(next_token_logits, dim=-1)

            # Get top-k candidates
            top_log_probs, top_indices = torch.topk(log_probs[0], beam_width)

            for log_prob, idx in zip(top_log_probs, top_indices):
                new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                # Length-normalized score
                new_score = score + log_prob.item()
                normalized_score = new_score / (new_seq.size(1) ** length_penalty)
                candidates.append((normalized_score, new_score, new_seq))

        # Select top beam_width candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [(raw_score, seq) for _, raw_score, seq in candidates[:beam_width]]

    # Return best sequence
    best_score, best_seq = beams[0]
    return best_seq, best_score


def contrastive_search(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    alpha: float = 0.6,
    k: int = 4,
) -> torch.Tensor:
    """Contrastive search decoding for high-quality, diverse generation.

    Args:
        model: The language model
        prompt: Initial token sequence (1, seq_len)
        max_new_tokens: Maximum tokens to generate
        alpha: Weighting between model confidence and degeneration penalty
        k: Number of top candidates to consider

    Returns:
        Generated sequence
    """
    generated = prompt.clone()
    past_hidden = None

    for _ in range(max_new_tokens):
        with torch.no_grad():
            window = generated[:, -model.config.max_seq_len:]
            logits = model(window)
            next_token_logits = logits[:, -1, :]

            # Get top-k candidates
            top_logits, top_indices = torch.topk(next_token_logits, k, dim=-1)
            probs = F.softmax(top_logits, dim=-1)

            # For simplicity, just use probability-based selection
            # In full implementation, would compute cosine similarity with context
            # to penalize degenerate repetition
            selected_idx = torch.multinomial(probs, num_samples=1)
            next_token = top_indices.gather(-1, selected_idx)

            generated = torch.cat([generated, next_token], dim=1)

    return generated


def sample_with_strategy(
    model,
    prompt: torch.Tensor,
    max_new_tokens: int,
    strategy: str = "greedy",
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    beam_width: int = 4,
    **kwargs,
) -> torch.Tensor:
    """Unified sampling interface supporting multiple strategies.

    Args:
        model: The language model
        prompt: Initial token sequence
        max_new_tokens: Maximum tokens to generate
        strategy: One of ['greedy', 'temperature', 'top_k', 'top_p', 'beam', 'contrastive']
        temperature: Sampling temperature
        top_k: Number of top tokens for top-k sampling
        top_p: Cumulative probability threshold for nucleus sampling
        beam_width: Beam width for beam search
        **kwargs: Additional strategy-specific arguments

    Returns:
        Generated sequence
    """
    if strategy == "beam":
        return beam_search(model, prompt, max_new_tokens, beam_width, temperature)[0]
    elif strategy == "contrastive":
        return contrastive_search(model, prompt, max_new_tokens, **kwargs)

    # For other strategies, use iterative generation
    generated = prompt.clone()
    device = prompt.device

    for _ in range(max_new_tokens):
        with torch.no_grad():
            window = generated[:, -model.config.max_seq_len:]
            logits = model(window)
            next_token_logits = logits[:, -1, :]

            if strategy == "greedy":
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            elif strategy == "top_k" and top_k is not None:
                next_token = top_k_sampling(next_token_logits, top_k, temperature)
            elif strategy == "top_p" and top_p is not None:
                next_token = top_p_sampling(next_token_logits, top_p, temperature)
            else:  # temperature sampling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

    return generated


__all__ = [
    "top_k_sampling",
    "top_p_sampling",
    "beam_search",
    "contrastive_search",
    "sample_with_strategy",
]
