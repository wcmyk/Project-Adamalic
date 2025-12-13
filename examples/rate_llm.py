"""Rate a LILITH checkpoint with quick quantitative metrics.

This script loads a saved checkpoint, evaluates it on a small corpus,
and prints out perplexity/accuracy numbers alongside a few sampled
generations so you can gauge qualitative quality.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from torch.utils.data import DataLoader

from LILITH.data import CharacterTokenizer, TextDataset
from LILITH.evaluation import evaluate_model, sample_quality_metrics
from LILITH.train_phase2 import load_checkpoint


DEFAULT_CORPUS: List[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning systems thrive on high-quality data.",
    "Python remains a go-to language for AI research.",
    "Transformers power most modern large language models.",
    "Evaluation should balance quantitative scores and human judgment.",
]

DEFAULT_PROMPTS: List[str] = [
    "Once upon a time in a distant lab,",
    "def greet(name):",
    "In this study, we demonstrate",
]


def _read_corpus(path: Path | None) -> List[str]:
    """Load evaluation corpus from file or fall back to defaults."""

    if path is None:
        return DEFAULT_CORPUS

    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")

    text = path.read_text(encoding="utf-8")
    # Use non-empty lines; if the file is a single paragraph keep it whole
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines if lines else [text.strip()]


def _filter_for_vocab(texts: Iterable[str], tokenizer: CharacterTokenizer) -> List[str]:
    """Remove characters that are not covered by the tokenizer vocabulary."""

    filtered: List[str] = []
    for text in texts:
        sanitized = "".join(ch for ch in text if ch in tokenizer.stoi)
        if sanitized:
            filtered.append(sanitized)
    return filtered


def _build_dataloader(
    corpus: List[str], tokenizer: CharacterTokenizer, block_size: int, batch_size: int
) -> DataLoader:
    """Construct a DataLoader for perplexity/accuracy evaluation."""

    tokens: List[int] = []
    for text in corpus:
        tokens.extend(tokenizer.encode(text))

    if len(tokens) <= block_size:
        raise ValueError(
            "Evaluation corpus is too small for the configured block size. "
            f"Need more than {block_size} tokens but only have {len(tokens)}."
        )

    dataset = TextDataset(tokens=tokens, block_size=block_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _prepare_prompts(raw_prompts: List[str], tokenizer: CharacterTokenizer) -> List[str]:
    """Ensure prompts only contain known characters."""

    prompts = _filter_for_vocab(raw_prompts, tokenizer)
    if not prompts:
        prompts = ["hello"]
    return prompts


def rate_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model, tokenizer, metadata = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    raw_corpus = _read_corpus(args.eval_file)
    corpus = _filter_for_vocab(raw_corpus, tokenizer)

    if not corpus:
        raise ValueError(
            "No evaluation text left after filtering out unsupported characters. "
            "Consider using text that fits the tokenizer vocabulary."
        )

    dataloader = _build_dataloader(corpus, tokenizer, args.block_size, args.batch_size)
    metrics = evaluate_model(model, dataloader, device=device)

    prompts = _prepare_prompts(args.prompts or DEFAULT_PROMPTS, tokenizer)
    quality = sample_quality_metrics(
        model,
        prompts,
        tokenizer,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    print("\n=== LILITH Checkpoint Rating ===")
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Device: {device}")
    print(f"Tokens evaluated: {metrics['total_tokens']}")
    if metadata.get("step") is not None:
        print(f"Training step: {metadata['step']}")
    if metadata.get("val_loss") is not None:
        print(f"Validation loss: {metadata['val_loss']:.4f}")
    print("\nQuantitative metrics:")
    print(f"  Perplexity:     {metrics['perplexity']:.4f}")
    print(f"  Top-1 accuracy: {metrics['top1_accuracy']*100:.2f}%")
    print(f"  Top-5 accuracy: {metrics['top5_accuracy']*100:.2f}%")

    print("\nSample generations:")
    for prompt, generation in zip(prompts, quality["generations"], strict=False):
        print(f"- Prompt: {prompt}")
        print(f"  Generated: {generation}\n")

    print("Diversity:")
    print(f"  Avg unique token ratio: {quality['avg_unique_token_ratio']:.3f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rate a LILITH checkpoint.")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint to evaluate")
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=None,
        help="Optional text file to use for evaluation (defaults to built-in corpus)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run evaluation on")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--block-size", type=int, default=128, help="Context window for evaluation")
    parser.add_argument(
        "--prompts",
        nargs="*",
        help="Optional prompts for qualitative sampling (defaults to mixed prompts)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to generate for qualitative samples",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature for qualitative samples",
    )
    return parser.parse_args()


if __name__ == "__main__":
    rate_checkpoint(parse_args())
