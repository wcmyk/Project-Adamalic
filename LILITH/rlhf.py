"""Reinforcement Learning from Human Feedback (RLHF) for LILITH.

Implements preference learning to align models with human values.
"""
from __future__ import annotations

from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .model import GPTDecoder


@dataclass
class PreferencePair:
    """A pair of responses with human preference."""
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Less preferred response


class PreferenceDataset(Dataset):
    """Dataset of preference pairs for RLHF."""

    def __init__(
        self,
        pairs: List[PreferencePair],
        tokenizer,
        max_length: int = 512,
    ):
        """Initialize preference dataset.

        Args:
            pairs: List of preference pairs
            tokenizer: Tokenizer
            max_length: Maximum sequence length
        """
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[idx]

        # Encode prompt + chosen
        chosen_text = f"{pair.prompt}\n\n{pair.chosen}"
        chosen_ids = self.tokenizer.encode(chosen_text)[:self.max_length]

        # Encode prompt + rejected
        rejected_text = f"{pair.prompt}\n\n{pair.rejected}"
        rejected_ids = self.tokenizer.encode(rejected_text)[:self.max_length]

        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
        }


class RewardModel(nn.Module):
    """Reward model for scoring responses.

    Built on top of LILITH to predict human preferences.
    """

    def __init__(self, base_model: GPTDecoder):
        """Initialize reward model.

        Args:
            base_model: Pre-trained LILITH model
        """
        super().__init__()
        self.base_model = base_model

        # Freeze base model (optional)
        # for param in self.base_model.parameters():
        #     param.requires_grad = False

        # Reward head
        self.reward_head = nn.Linear(base_model.config.d_model, 1)

        # Initialize reward head
        nn.init.zeros_(self.reward_head.weight)
        nn.init.zeros_(self.reward_head.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass to get reward score.

        Args:
            input_ids: Token IDs (batch, seq_len)

        Returns:
            Reward scores (batch,)
        """
        # Get hidden states from base model
        logits = self.base_model(input_ids)

        # Use last token's representation
        last_hidden = logits[:, -1, :]

        # Project to scalar reward
        rewards = self.reward_head(last_hidden).squeeze(-1)

        return rewards


def train_reward_model(
    base_model: GPTDecoder,
    preference_pairs: List[PreferencePair],
    tokenizer,
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
    device: str = "cuda",
) -> RewardModel:
    """Train a reward model on preference data.

    Args:
        base_model: Base LILITH model
        preference_pairs: Training preference pairs
        tokenizer: Tokenizer
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to train on

    Returns:
        Trained reward model
    """
    # Create reward model
    reward_model = RewardModel(base_model).to(device)
    reward_model.train()

    # Create dataset and dataloader
    dataset = PreferenceDataset(preference_pairs, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(reward_model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            chosen_ids = batch["chosen_ids"].to(device)
            rejected_ids = batch["rejected_ids"].to(device)

            # Get rewards
            chosen_rewards = reward_model(chosen_ids)
            rejected_rewards = reward_model(rejected_ids)

            # Loss: maximize margin between chosen and rejected
            # Using ranking loss (similar to Bradley-Terry model)
            loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            correct += (chosen_rewards > rejected_rewards).sum().item()
            total += chosen_ids.size(0)

        accuracy = correct / total if total > 0 else 0
        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.4f}")

    reward_model.eval()
    return reward_model


class PPOTrainer:
    """Proximal Policy Optimization trainer for RLHF.

    This is a simplified version. Full implementation would be more complex.
    """

    def __init__(
        self,
        policy_model: GPTDecoder,
        reward_model: RewardModel,
        ref_model: Optional[GPTDecoder] = None,
        kl_coef: float = 0.1,
    ):
        """Initialize PPO trainer.

        Args:
            policy_model: Model being optimized
            reward_model: Reward model for scoring
            ref_model: Reference model for KL penalty (optional)
            kl_coef: KL divergence coefficient
        """
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model or policy_model
        self.kl_coef = kl_coef

    def compute_rewards(
        self,
        prompts: List[str],
        responses: List[str],
        tokenizer,
    ) -> torch.Tensor:
        """Compute rewards for responses.

        Args:
            prompts: Input prompts
            responses: Generated responses
            tokenizer: Tokenizer

        Returns:
            Reward scores
        """
        device = next(self.reward_model.parameters()).device
        rewards = []

        for prompt, response in zip(prompts, responses):
            # Encode prompt + response
            text = f"{prompt}\n\n{response}"
            ids = torch.tensor([tokenizer.encode(text)], device=device)

            # Get reward
            with torch.no_grad():
                reward = self.reward_model(ids)
                rewards.append(reward.item())

        return torch.tensor(rewards, device=device)

    def train_step(
        self,
        prompts: List[str],
        tokenizer,
        max_length: int = 256,
    ) -> Dict[str, float]:
        """Single PPO training step.

        Args:
            prompts: Input prompts
            tokenizer: Tokenizer
            max_length: Max generation length

        Returns:
            Training metrics
        """
        # This is a simplified placeholder
        # Full PPO implementation would include:
        # 1. Generate responses
        # 2. Compute rewards
        # 3. Compute advantages
        # 4. Update policy with clipped objective
        # 5. Add KL penalty w.r.t. reference model

        # For now, just a stub
        return {
            "reward": 0.0,
            "kl": 0.0,
            "loss": 0.0,
        }


def create_preference_dataset_from_comparisons(
    prompts: List[str],
    response_pairs: List[Tuple[str, str]],
    preferences: List[int],  # 0 for first, 1 for second
) -> List[PreferencePair]:
    """Create preference dataset from human comparisons.

    Args:
        prompts: Input prompts
        response_pairs: Pairs of responses to compare
        preferences: Which response was preferred (0 or 1)

    Returns:
        List of preference pairs
    """
    pairs = []

    for prompt, (resp1, resp2), pref in zip(prompts, response_pairs, preferences):
        if pref == 0:
            chosen, rejected = resp1, resp2
        else:
            chosen, rejected = resp2, resp1

        pairs.append(PreferencePair(
            prompt=prompt,
            chosen=chosen,
            rejected=rejected,
        ))

    return pairs


# Synthetic preference generation for bootstrapping
def generate_synthetic_preferences(
    prompts: List[str],
    good_responses: List[str],
    bad_responses: List[str],
) -> List[PreferencePair]:
    """Generate synthetic preference pairs.

    Args:
        prompts: Input prompts
        good_responses: High-quality responses
        bad_responses: Low-quality responses

    Returns:
        Preference pairs
    """
    return [
        PreferencePair(prompt=p, chosen=good, rejected=bad)
        for p, good, bad in zip(prompts, good_responses, bad_responses)
    ]


__all__ = [
    "PreferencePair",
    "PreferenceDataset",
    "RewardModel",
    "train_reward_model",
    "PPOTrainer",
    "create_preference_dataset_from_comparisons",
    "generate_synthetic_preferences",
]
