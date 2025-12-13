"""Advanced tokenization with BPE (Byte-Pair Encoding) support."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple
import re


class BPETokenizer:
    """Byte-Pair Encoding tokenizer for subword tokenization.

    BPE is more efficient than character-level tokenization and better
    than word-level for handling rare words and morphology.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        min_frequency: int = 2,
        special_tokens: List[str] = None,
    ):
        """Initialize BPE tokenizer.

        Args:
            vocab_size: Target vocabulary size
            min_frequency: Minimum frequency for a pair to be merged
            special_tokens: Special tokens to add (e.g., ['<pad>', '<unk>', '<s>', '</s>'])
        """
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        # Special tokens
        self.special_tokens = special_tokens or ["<pad>", "<unk>", "<s>", "</s>"]
        self.special_token_ids = {tok: i for i, tok in enumerate(self.special_tokens)}

        # BPE merges and vocabulary
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}

    def train(self, corpus: Iterable[str]) -> None:
        """Train BPE on a corpus.

        Args:
            corpus: Training texts
        """
        # Initialize vocabulary with special tokens
        self.vocab = self.special_token_ids.copy()
        current_id = len(self.special_tokens)

        # Pre-tokenize into words
        word_freqs = Counter()
        for text in corpus:
            words = self._pre_tokenize(text)
            word_freqs.update(words)

        # Initialize word representations (space-separated characters)
        word_splits = {
            word: [c for c in word] for word in word_freqs.keys()
        }

        # Add individual characters to vocabulary
        for word in word_freqs:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = current_id
                    current_id += 1

        # Iteratively merge most frequent pairs
        while len(self.vocab) < self.vocab_size:
            # Count pair frequencies
            pair_freqs = defaultdict(int)
            for word, freq in word_freqs.items():
                split = word_splits[word]
                if len(split) < 2:
                    continue

                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    pair_freqs[pair] += freq

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair = max(pair_freqs.items(), key=lambda x: x[1])
            if best_pair[1] < self.min_frequency:
                break

            pair, freq = best_pair

            # Merge this pair in all words
            merged = pair[0] + pair[1]
            self.merges.append(pair)

            if merged not in self.vocab:
                self.vocab[merged] = current_id
                current_id += 1

            # Update word splits
            for word in word_freqs:
                split = word_splits[word]
                i = 0
                new_split = []

                while i < len(split):
                    if i < len(split) - 1 and (split[i], split[i + 1]) == pair:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1

                word_splits[word] = new_split

        # Create inverse vocabulary
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def _pre_tokenize(self, text: str) -> List[str]:
        """Pre-tokenize text into words.

        Args:
            text: Input text

        Returns:
            List of words
        """
        # Simple whitespace + punctuation tokenization
        pattern = r'\w+|[^\w\s]'
        return re.findall(pattern, text.lower())

    def _tokenize_word(self, word: str) -> List[str]:
        """Tokenize a single word using learned merges.

        Args:
            word: Input word

        Returns:
            List of subword tokens
        """
        # Start with character-level split
        tokens = list(word)

        # Apply merges in order
        for pair in self.merges:
            i = 0
            new_tokens = []

            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add <s> and </s> tokens

        Returns:
            List of token IDs
        """
        # Pre-tokenize into words
        words = self._pre_tokenize(text)

        # Tokenize each word
        token_ids = []
        if add_special_tokens:
            token_ids.append(self.special_token_ids["<s>"])

        unk_id = self.special_token_ids["<unk>"]

        for word in words:
            tokens = self._tokenize_word(word)
            for token in tokens:
                token_id = self.vocab.get(token, unk_id)
                token_ids.append(token_id)

        if add_special_tokens:
            token_ids.append(self.special_token_ids["</s>"])

        return token_ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.

        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens

        Returns:
            Decoded text
        """
        tokens = []
        special_ids = set(self.special_token_ids.values())

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            token = self.inverse_vocab.get(token_id, "<unk>")
            tokens.append(token)

        # Simple joining (could be improved with better detokenization)
        return "".join(tokens)

    @property
    def vocab_size_actual(self) -> int:
        """Get actual vocabulary size."""
        return len(self.vocab)

    def save(self, path: str) -> None:
        """Save tokenizer to file.

        Args:
            path: Path to save
        """
        import json

        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
            "config": {
                "vocab_size": self.vocab_size,
                "min_frequency": self.min_frequency,
            }
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Load tokenizer from file.

        Args:
            path: Path to load from

        Returns:
            Loaded tokenizer
        """
        import json

        with open(path, 'r') as f:
            data = json.load(f)

        tokenizer = cls(
            vocab_size=data["config"]["vocab_size"],
            min_frequency=data["config"]["min_frequency"],
            special_tokens=data["special_tokens"],
        )

        tokenizer.vocab = {k: int(v) for k, v in data["vocab"].items()}
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        tokenizer.special_token_ids = {
            tok: i for i, tok in enumerate(tokenizer.special_tokens)
        }

        return tokenizer


__all__ = ["BPETokenizer"]
