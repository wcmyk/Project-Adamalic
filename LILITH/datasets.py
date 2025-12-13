"""Dataset loaders for training LLMs on real data."""
from __future__ import annotations

import os
from typing import Iterator, List, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, Dataset

# Optional imports - will work without them but with limited functionality
try:
    from datasets import load_dataset as hf_load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    hf_load_dataset = None


class StreamingTextDataset(IterableDataset):
    """Stream large text datasets without loading everything into memory.

    Works with HuggingFace datasets for efficient memory usage.
    Perfect for training on massive corpora like OpenWebText, C4, or The Pile.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        text_field: str = "text",
        streaming: bool = True,
        buffer_size: int = 10000,
    ):
        """Initialize streaming dataset.

        Args:
            dataset_name: HuggingFace dataset name (e.g., 'wikipedia', 'openwebtext')
            split: Dataset split ('train', 'validation', 'test')
            text_field: Name of the text field in the dataset
            streaming: Whether to stream the dataset
            buffer_size: Buffer size for shuffling

        Raises:
            ImportError: If datasets library is not installed
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library required for StreamingTextDataset. "
                "Install with: pip install datasets"
            )

        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.text_field = text_field
        self.streaming = streaming
        self.buffer_size = buffer_size

        # Load dataset
        self.dataset = hf_load_dataset(
            dataset_name,
            split=split,
            streaming=streaming,
        )

        if streaming:
            self.dataset = self.dataset.shuffle(buffer_size=buffer_size)

    def __iter__(self) -> Iterator[str]:
        """Iterate over text samples."""
        for item in self.dataset:
            if self.text_field in item:
                yield item[self.text_field]
            else:
                # Try to find text field automatically
                for key in ['text', 'content', 'document', 'article']:
                    if key in item:
                        yield item[key]
                        break


class WikipediaDataset(IterableDataset):
    """Wikipedia dataset loader with convenient defaults.

    Loads English Wikipedia (or other languages) for training.
    """

    def __init__(
        self,
        language: str = "en",
        date: str = "20220301",
        split: str = "train",
        streaming: bool = True,
        subset_percentage: Optional[float] = None,
    ):
        """Initialize Wikipedia dataset.

        Args:
            language: Wikipedia language code ('en', 'de', 'fr', etc.)
            date: Wikipedia dump date (format: YYYYMMDD)
            split: Dataset split
            streaming: Whether to stream
            subset_percentage: If set, only use this percentage of data (e.g., 0.01 for 1%)

        Raises:
            ImportError: If datasets library is not installed
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library required for WikipediaDataset. "
                "Install with: pip install datasets"
            )

        super().__init__()
        self.language = language
        self.date = date
        self.streaming = streaming
        self.subset_percentage = subset_percentage

        # Construct dataset name
        dataset_name = f"{date}.{language}"

        # Load dataset
        if subset_percentage is not None and not streaming:
            # Load subset for non-streaming
            split_str = f"{split}[:{int(subset_percentage * 100)}%]"
        else:
            split_str = split

        self.dataset = hf_load_dataset(
            "wikipedia",
            dataset_name,
            split=split_str,
            streaming=streaming,
        )

    def __iter__(self) -> Iterator[str]:
        """Iterate over Wikipedia articles."""
        count = 0
        total_limit = None

        if self.streaming and self.subset_percentage is not None:
            # For streaming, we need to count manually
            # This is approximate since we don't know total size
            total_limit = int(1000000 * self.subset_percentage)  # Assume ~1M articles

        for item in self.dataset:
            if total_limit and count >= total_limit:
                break

            text = item.get('text', '')
            if text:
                yield text
                count += 1


class CodeDataset(IterableDataset):
    """Dataset for code from GitHub and other sources."""

    def __init__(
        self,
        language: Optional[str] = None,
        split: str = "train",
        streaming: bool = True,
    ):
        """Initialize code dataset.

        Args:
            language: Programming language filter (e.g., 'python', 'javascript')
            split: Dataset split
            streaming: Whether to stream

        Raises:
            ImportError: If datasets library is not installed
        """
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library required for CodeDataset. "
                "Install with: pip install datasets"
            )

        super().__init__()

        # Use The Stack dataset
        self.dataset = hf_load_dataset(
            "bigcode/the-stack-dedup",
            data_dir=f"data/{language}" if language else None,
            split=split,
            streaming=streaming,
        )

    def __iter__(self) -> Iterator[str]:
        """Iterate over code samples."""
        for item in self.dataset:
            content = item.get('content', '')
            if content:
                yield content


class CombinedDataset(IterableDataset):
    """Combine multiple datasets with configurable mixing ratios."""

    def __init__(
        self,
        datasets: List[IterableDataset],
        weights: Optional[List[float]] = None,
        seed: int = 42,
    ):
        """Initialize combined dataset.

        Args:
            datasets: List of datasets to combine
            weights: Sampling weights for each dataset (default: equal)
            seed: Random seed for sampling
        """
        super().__init__()
        self.datasets = datasets

        if weights is None:
            weights = [1.0] * len(datasets)

        # Normalize weights
        total = sum(weights)
        self.weights = [w / total for w in weights]
        self.seed = seed

    def __iter__(self) -> Iterator[str]:
        """Iterate over combined datasets."""
        import random

        rng = random.Random(self.seed)
        iterators = [iter(ds) for ds in self.datasets]

        while True:
            # Sample a dataset based on weights
            idx = rng.choices(range(len(self.datasets)), weights=self.weights)[0]

            try:
                item = next(iterators[idx])
                yield item
            except StopIteration:
                # This dataset is exhausted
                break


class LocalTextDataset(Dataset):
    """Load text from local files."""

    def __init__(
        self,
        file_paths: Union[str, Path, List[Union[str, Path]]],
        chunk_size: int = 1024,
        overlap: int = 128,
    ):
        """Initialize local text dataset.

        Args:
            file_paths: Single file path or list of file paths
            chunk_size: Characters per chunk
            overlap: Overlap between chunks
        """
        super().__init__()

        if isinstance(file_paths, (str, Path)):
            file_paths = [file_paths]

        self.file_paths = [Path(p) for p in file_paths]
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Load and chunk all texts
        self.chunks = []
        for path in self.file_paths:
            if path.exists():
                text = path.read_text(encoding='utf-8', errors='ignore')
                self._chunk_text(text)

    def _chunk_text(self, text: str):
        """Split text into overlapping chunks."""
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk:
                self.chunks.append(chunk)
            start += (self.chunk_size - self.overlap)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> str:
        return self.chunks[idx]


def get_wikipedia_corpus(
    language: str = "en",
    subset_percentage: float = 1.0,
    streaming: bool = True,
) -> WikipediaDataset:
    """Convenience function to get Wikipedia corpus.

    Args:
        language: Language code
        subset_percentage: Percentage of data to use (0.01 = 1%)
        streaming: Whether to stream

    Returns:
        Wikipedia dataset
    """
    return WikipediaDataset(
        language=language,
        streaming=streaming,
        subset_percentage=subset_percentage if subset_percentage < 1.0 else None,
    )


def get_code_corpus(
    language: str = "python",
    streaming: bool = True,
) -> CodeDataset:
    """Convenience function to get code corpus.

    Args:
        language: Programming language
        streaming: Whether to stream

    Returns:
        Code dataset
    """
    return CodeDataset(language=language, streaming=streaming)


__all__ = [
    "StreamingTextDataset",
    "WikipediaDataset",
    "CodeDataset",
    "CombinedDataset",
    "LocalTextDataset",
    "get_wikipedia_corpus",
    "get_code_corpus",
]
