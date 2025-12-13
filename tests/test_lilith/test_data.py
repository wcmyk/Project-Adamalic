"""Tests for LILITH data components."""
import pytest
import torch
from LILITH.data import CharacterTokenizer, TextDataset


class TestCharacterTokenizer:
    """Test suite for CharacterTokenizer."""

    @pytest.fixture
    def corpus(self):
        """Sample corpus for testing."""
        return ["hello world", "test data"]

    @pytest.fixture
    def tokenizer(self, corpus):
        """Create tokenizer."""
        return CharacterTokenizer(corpus)

    def test_tokenizer_initialization(self, tokenizer):
        """Test tokenizer initializes with correct vocabulary."""
        assert tokenizer.vocab_size > 0
        assert len(tokenizer.itos) == tokenizer.vocab_size
        assert len(tokenizer.stoi) == tokenizer.vocab_size

    def test_vocabulary_deterministic(self, corpus):
        """Test vocabulary is deterministically ordered."""
        tok1 = CharacterTokenizer(corpus)
        tok2 = CharacterTokenizer(corpus)

        assert tok1.itos == tok2.itos
        assert tok1.stoi == tok2.stoi

    def test_encode(self, tokenizer):
        """Test encoding text to IDs."""
        text = "hello"
        encoded = tokenizer.encode(text)

        assert isinstance(encoded, list)
        assert len(encoded) == len(text)
        assert all(isinstance(x, int) for x in encoded)

    def test_decode(self, tokenizer):
        """Test decoding IDs to text."""
        text = "hello"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)

        assert decoded == text

    def test_encode_decode_roundtrip(self, tokenizer, corpus):
        """Test encode-decode is lossless for known characters."""
        for text in corpus:
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            assert decoded == text

    def test_unknown_character_raises_error(self, tokenizer):
        """Test encoding unknown character raises KeyError."""
        with pytest.raises(KeyError):
            tokenizer.encode("§")  # Character not in corpus


class TestTextDataset:
    """Test suite for TextDataset."""

    @pytest.fixture
    def dataset(self):
        """Create test dataset."""
        tokens = list(range(100))  # 0-99
        block_size = 10
        return TextDataset(tokens=tokens, block_size=block_size)

    def test_dataset_length(self, dataset):
        """Test dataset reports correct length."""
        # Length should be len(tokens) - block_size
        assert len(dataset) == 90

    def test_dataset_getitem(self, dataset):
        """Test getting items from dataset."""
        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (10,)
        assert y.shape == (10,)

    def test_dataset_targets_are_shifted(self, dataset):
        """Test that targets are shifted by 1."""
        x, y = dataset[0]

        # y should be x shifted by 1
        assert torch.all(y == x + 1)

    def test_dataset_iteration(self, dataset):
        """Test iterating over dataset."""
        count = 0
        for x, y in dataset:
            count += 1
            assert x.shape == (10,)
            assert y.shape == (10,)

        assert count == len(dataset)

    def test_empty_dataset(self):
        """Test dataset with insufficient tokens."""
        tokens = list(range(5))
        dataset = TextDataset(tokens=tokens, block_size=10)

        assert len(dataset) == 0

    def test_exact_size_dataset(self):
        """Test dataset with exactly block_size tokens."""
        tokens = list(range(10))
        dataset = TextDataset(tokens=tokens, block_size=10)

        assert len(dataset) == 0  # Need block_size + 1 for x and y
