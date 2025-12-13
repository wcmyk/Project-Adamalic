"""Tests for LILITH model components."""
import pytest
import torch
from LILITH.config import ModelConfig
from LILITH.model import GPTDecoder


class TestGPTDecoder:
    """Test suite for GPTDecoder."""

    @pytest.fixture
    def config(self):
        """Create test model configuration."""
        return ModelConfig(
            vocab_size=100,
            d_model=64,
            n_layers=2,
            n_heads=4,
            d_ff=256,
            max_seq_len=128,
            dropout=0.1,
        )

    @pytest.fixture
    def model(self, config):
        """Create test model."""
        return GPTDecoder(config)

    def test_model_initialization(self, config):
        """Test model initializes correctly."""
        model = GPTDecoder(config)
        assert model.config == config
        assert model.token_embed.num_embeddings == config.vocab_size
        assert model.pos_embed.num_embeddings == config.max_seq_len

    def test_forward_pass(self, model, config):
        """Test forward pass produces correct output shape."""
        batch_size = 4
        seq_len = 32
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(token_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_forward_with_gradient_checkpointing(self, config):
        """Test forward pass with gradient checkpointing."""
        model = GPTDecoder(config, use_gradient_checkpointing=True)
        batch_size = 4
        seq_len = 32
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        model.train()
        logits = model(token_ids)

        assert logits.shape == (batch_size, seq_len, config.vocab_size)

    def test_max_seq_len_validation(self, model, config):
        """Test that sequences longer than max_seq_len raise error."""
        batch_size = 2
        seq_len = config.max_seq_len + 10
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        with pytest.raises(ValueError, match="exceeds max_seq_len"):
            model(token_ids)

    def test_generation(self, model, config):
        """Test text generation."""
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        max_new_tokens = 20

        generated = model.generate(prompt, max_new_tokens, temperature=1.0)

        assert generated.shape == (1, 10 + max_new_tokens)

    def test_generation_with_temperature(self, model, config):
        """Test generation with different temperatures."""
        prompt = torch.randint(0, config.vocab_size, (1, 10))

        gen_low_temp = model.generate(prompt, 10, temperature=0.5)
        gen_high_temp = model.generate(prompt, 10, temperature=2.0)

        assert gen_low_temp.shape[1] == 20
        assert gen_high_temp.shape[1] == 20

    def test_loss_calculation(self, model, config):
        """Test loss calculation."""
        batch_size = 4
        seq_len = 32
        token_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        logits = model(token_ids)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

        loss = model.loss(logits, targets)

        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0

    def test_parameter_counting(self, model):
        """Test parameter counting methods."""
        total_params = model.count_parameters(trainable_only=False)
        trainable_params = model.count_parameters(trainable_only=True)

        assert total_params > 0
        assert trainable_params == total_params

        param_breakdown = model.get_num_params()
        assert "total" in param_breakdown
        assert "trainable" in param_breakdown
        assert "embeddings" in param_breakdown
        assert param_breakdown["total"] == total_params

    def test_model_modes(self, model):
        """Test model can switch between train and eval modes."""
        model.train()
        assert model.training

        model.eval()
        assert not model.training

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_compatibility(self, model):
        """Test model can be moved to CUDA."""
        model_cuda = model.cuda()
        token_ids = torch.randint(0, 100, (2, 16)).cuda()

        logits = model_cuda(token_ids)
        assert logits.device.type == "cuda"
