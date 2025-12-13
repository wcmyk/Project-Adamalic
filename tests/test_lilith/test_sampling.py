"""Tests for LILITH sampling strategies."""
import pytest
import torch
from LILITH.sampling import (
    top_k_sampling,
    top_p_sampling,
    sample_with_strategy,
)


class TestSamplingStrategies:
    """Test suite for sampling strategies."""

    @pytest.fixture
    def logits(self):
        """Create test logits."""
        # Shape: (batch=2, vocab_size=100)
        return torch.randn(2, 100)

    def test_top_k_sampling(self, logits):
        """Test top-k sampling."""
        k = 10
        sampled = top_k_sampling(logits, k, temperature=1.0)

        assert sampled.shape == (2, 1)
        assert torch.all(sampled >= 0)
        assert torch.all(sampled < 100)

    def test_top_k_with_temperature(self, logits):
        """Test top-k sampling with different temperatures."""
        k = 10

        sampled_low = top_k_sampling(logits, k, temperature=0.5)
        sampled_high = top_k_sampling(logits, k, temperature=2.0)

        assert sampled_low.shape == (2, 1)
        assert sampled_high.shape == (2, 1)

    def test_top_p_sampling(self, logits):
        """Test nucleus (top-p) sampling."""
        p = 0.9
        sampled = top_p_sampling(logits, p, temperature=1.0)

        assert sampled.shape == (2, 1)
        assert torch.all(sampled >= 0)
        assert torch.all(sampled < 100)

    def test_top_p_with_temperature(self, logits):
        """Test top-p sampling with temperature."""
        p = 0.9

        sampled_low = top_p_sampling(logits, p, temperature=0.5)
        sampled_high = top_p_sampling(logits, p, temperature=2.0)

        assert sampled_low.shape == (2, 1)
        assert sampled_high.shape == (2, 1)

    def test_sample_with_strategy_greedy(self):
        """Test greedy sampling strategy."""
        from LILITH.config import ModelConfig
        from LILITH.model import GPTDecoder

        config = ModelConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=2)
        model = GPTDecoder(config)
        model.eval()

        prompt = torch.randint(0, 50, (1, 5))
        generated = sample_with_strategy(
            model, prompt, max_new_tokens=10, strategy="greedy"
        )

        assert generated.shape == (1, 15)

    def test_sample_with_strategy_temperature(self):
        """Test temperature sampling strategy."""
        from LILITH.config import ModelConfig
        from LILITH.model import GPTDecoder

        config = ModelConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=2)
        model = GPTDecoder(config)
        model.eval()

        prompt = torch.randint(0, 50, (1, 5))
        generated = sample_with_strategy(
            model, prompt, max_new_tokens=10, strategy="temperature", temperature=0.8
        )

        assert generated.shape == (1, 15)

    def test_sample_with_strategy_top_k(self):
        """Test top-k sampling strategy."""
        from LILITH.config import ModelConfig
        from LILITH.model import GPTDecoder

        config = ModelConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=2)
        model = GPTDecoder(config)
        model.eval()

        prompt = torch.randint(0, 50, (1, 5))
        generated = sample_with_strategy(
            model, prompt, max_new_tokens=10, strategy="top_k", top_k=5
        )

        assert generated.shape == (1, 15)

    def test_sample_with_strategy_top_p(self):
        """Test top-p sampling strategy."""
        from LILITH.config import ModelConfig
        from LILITH.model import GPTDecoder

        config = ModelConfig(vocab_size=50, d_model=32, n_layers=1, n_heads=2)
        model = GPTDecoder(config)
        model.eval()

        prompt = torch.randint(0, 50, (1, 5))
        generated = sample_with_strategy(
            model, prompt, max_new_tokens=10, strategy="top_p", top_p=0.9
        )

        assert generated.shape == (1, 15)
