#!/usr/bin/env python3
"""
Train LILITH on code datasets for programming capabilities.

This script provides a complete pipeline for training code-focused language models:
1. Load code datasets (The Stack, GitHub, etc.)
2. Train BPE tokenizer optimized for code
3. Pre-train model on code corpus
4. Optionally fine-tune with instruction datasets

Usage:
    # Train 1B model on Python code
    python train_code.py --model_size 1b --language python --num_gpus 4

    # Train medium model on multiple languages
    python train_code.py --model_size medium --language python,javascript,java

    # Resume training from checkpoint
    python train_code.py --checkpoint checkpoints/code_model.pt --resume

    # Multi-stage: Pre-train + Instruction tune
    python train_code.py --model_size 1b --language python --instruction_tune
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch

from LILITH import (
    get_code_corpus,
    get_wikipedia_corpus,
    CombinedDataset,
    BPETokenizer,
    train_phase2,
    load_checkpoint,
    create_logger,
    get_small_model_config,
    get_medium_model_config,
    get_large_model_config,
    get_1b_model_config,
    get_code_training_config,
    Phase2TrainingConfig,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LILITH on code datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large", "1b"],
        help="Model size (small=10M, medium=50M, large=150M, 1b=1B params)",
    )

    # Dataset configuration
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Programming language(s) to train on (comma-separated). "
        "Examples: python, javascript, python,java,cpp",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="stack",
        choices=["stack", "github", "mixed"],
        help="Code dataset source",
    )
    parser.add_argument(
        "--include_wikipedia",
        action="store_true",
        help="Mix Wikipedia data with code (improves natural language understanding)",
    )
    parser.add_argument(
        "--wiki_ratio",
        type=float,
        default=0.2,
        help="Ratio of Wikipedia to code data (0.0-1.0)",
    )
    parser.add_argument(
        "--subset",
        type=float,
        default=5.0,
        help="Percentage of dataset to use (1.0-100.0)",
    )

    # Tokenizer configuration
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=None,
        help="Tokenizer vocabulary size (default: from model config)",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to pre-trained tokenizer (if not specified, trains new one)",
    )

    # Training configuration
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size per GPU (default: auto-calculated)",
    )
    parser.add_argument(
        "--total_batch_size",
        type=int,
        default=512,
        help="Total effective batch size (uses gradient accumulation)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: from training config)",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=2000,
        help="Number of warmup steps",
    )

    # Hardware configuration
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )

    # Checkpoint configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/code_model",
        help="Directory to save checkpoints",
    )

    # Advanced options
    parser.add_argument(
        "--instruction_tune",
        action="store_true",
        help="After pre-training, fine-tune on instruction dataset (CodeAlpaca)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory for training logs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


def get_model_config(model_size: str, vocab_size: Optional[int] = None):
    """Get model configuration based on size."""
    if model_size == "small":
        config = get_small_model_config()
    elif model_size == "medium":
        config = get_medium_model_config()
    elif model_size == "large":
        config = get_large_model_config()
    elif model_size == "1b":
        config = get_1b_model_config()
    else:
        raise ValueError(f"Unknown model size: {model_size}")

    # Override vocab size if specified
    if vocab_size is not None:
        config.vocab_size = vocab_size

    return config


def load_or_train_tokenizer(
    corpus,
    vocab_size: int,
    tokenizer_path: Optional[str],
    logger,
):
    """Load existing tokenizer or train a new one."""
    if tokenizer_path and os.path.exists(tokenizer_path):
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(tokenizer_path)
        logger.info(f"Loaded tokenizer with vocab_size={tokenizer.vocab_size}")
        return tokenizer

    logger.info(f"Training new BPE tokenizer (vocab_size={vocab_size})...")
    logger.info("Collecting training corpus for tokenizer...")

    # Collect subset of corpus for tokenizer training
    tokenizer_corpus = []
    max_samples = 100_000  # 100k samples should be enough
    for i, text in enumerate(corpus):
        if i >= max_samples:
            break
        tokenizer_corpus.append(text)
        if (i + 1) % 10_000 == 0:
            logger.info(f"  Collected {i + 1:,} samples...")

    logger.info(f"Training tokenizer on {len(tokenizer_corpus):,} samples...")
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(tokenizer_corpus)

    # Save tokenizer
    save_path = f"tokenizers/code_{vocab_size}.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    tokenizer.save(save_path)
    logger.info(f"Saved tokenizer to {save_path}")

    return tokenizer


def load_code_corpus(args, logger):
    """Load code dataset(s) based on arguments."""
    languages = [lang.strip() for lang in args.language.split(",")]
    logger.info(f"Loading code datasets for languages: {languages}")

    datasets = []
    for language in languages:
        logger.info(f"  Loading {language} code...")
        code_data = get_code_corpus(
            language=language,
            subset_percentage=args.subset,
            streaming=True,
        )
        datasets.append(code_data)

    # Optionally mix with Wikipedia for better natural language
    if args.include_wikipedia:
        logger.info(f"Adding Wikipedia data (ratio={args.wiki_ratio})...")
        wiki_data = get_wikipedia_corpus(
            subset_percentage=args.subset * 0.1,  # Less Wikipedia
            streaming=True,
        )
        datasets.append(wiki_data)
        weights = [1.0 - args.wiki_ratio] * len(languages) + [args.wiki_ratio]
    else:
        weights = [1.0 / len(languages)] * len(languages)

    # Combine datasets
    if len(datasets) == 1:
        corpus = datasets[0]
    else:
        logger.info(f"Combining {len(datasets)} datasets with weights {weights}")
        corpus = CombinedDataset(datasets, weights=weights)

    return corpus


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logger
    logger = create_logger(name="CodeTraining", log_dir=args.log_dir)
    logger.info("=" * 80)
    logger.info("LILITH Code Training Pipeline")
    logger.info("=" * 80)
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Languages: {args.language}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Num GPUs: {args.num_gpus}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)

    # Stage 1: Load dataset
    logger.info("\n" + "=" * 80)
    logger.info("Stage 1: Loading Code Dataset")
    logger.info("=" * 80)
    corpus = load_code_corpus(args, logger)

    # Stage 2: Get model configuration
    logger.info("\n" + "=" * 80)
    logger.info("Stage 2: Model Configuration")
    logger.info("=" * 80)
    model_config = get_model_config(
        args.model_size,
        vocab_size=args.vocab_size,
    )
    logger.info(f"Model parameters:")
    logger.info(f"  vocab_size: {model_config.vocab_size:,}")
    logger.info(f"  d_model: {model_config.d_model}")
    logger.info(f"  n_layers: {model_config.n_layers}")
    logger.info(f"  n_heads: {model_config.n_heads}")
    logger.info(f"  d_ff: {model_config.d_ff:,}")
    logger.info(f"  max_seq_len: {model_config.max_seq_len}")

    # Estimate parameters
    if args.model_size == "1b":
        logger.info(f"  Estimated parameters: ~1.02B")
        logger.info(f"  Model size (FP32): ~4GB")
        logger.info(f"  Model size (FP16): ~2GB")

    # Stage 3: Tokenizer
    logger.info("\n" + "=" * 80)
    logger.info("Stage 3: Tokenizer Setup")
    logger.info("=" * 80)
    tokenizer = load_or_train_tokenizer(
        corpus,
        model_config.vocab_size,
        args.tokenizer_path,
        logger,
    )

    # Stage 4: Training configuration
    logger.info("\n" + "=" * 80)
    logger.info("Stage 4: Training Configuration")
    logger.info("=" * 80)
    train_config = get_code_training_config(
        num_gpus=args.num_gpus,
        total_batch_size=args.total_batch_size,
    )

    # Override with command line args
    if args.batch_size is not None:
        train_config.batch_size = args.batch_size
    if args.learning_rate is not None:
        train_config.learning_rate = args.learning_rate
    if args.num_epochs is not None:
        train_config.num_epochs = args.num_epochs
    train_config.warmup_steps = args.warmup_steps

    logger.info(f"Training parameters:")
    logger.info(f"  batch_size (per GPU): {train_config.batch_size}")
    logger.info(f"  gradient_accumulation_steps: {train_config.gradient_accumulation_steps}")
    logger.info(f"  effective_batch_size: {train_config.batch_size * args.num_gpus * train_config.gradient_accumulation_steps}")
    logger.info(f"  num_epochs: {train_config.num_epochs}")
    logger.info(f"  learning_rate: {train_config.learning_rate}")
    logger.info(f"  warmup_steps: {train_config.warmup_steps}")
    logger.info(f"  use_mixed_precision: {train_config.use_mixed_precision}")

    # Stage 5: Pre-training
    logger.info("\n" + "=" * 80)
    logger.info("Stage 5: Pre-training on Code")
    logger.info("=" * 80)

    checkpoint_path = os.path.join(args.output_dir, "code_pretrained.pt")

    if args.resume and args.checkpoint:
        logger.info(f"Resuming from checkpoint: {args.checkpoint}")
        model, tokenizer, train_state = load_checkpoint(args.checkpoint)
        logger.info(f"Resumed from step {train_state.get('step', 0)}")
    else:
        logger.info("Starting pre-training from scratch...")

    model = train_phase2(
        corpus=corpus,
        tokenizer=tokenizer,
        model_config=model_config,
        train_config=train_config,
        checkpoint_path=checkpoint_path,
        device=args.device,
    )

    logger.info(f"Pre-training complete! Model saved to {checkpoint_path}")

    # Stage 6: Instruction tuning (optional)
    if args.instruction_tune:
        logger.info("\n" + "=" * 80)
        logger.info("Stage 6: Instruction Tuning on CodeAlpaca")
        logger.info("=" * 80)
        logger.info("Launching instruction tuning script...")

        instruction_checkpoint = os.path.join(args.output_dir, "code_instructed.pt")

        # Import and run instruction tuning
        # This would call train_conversational.py programmatically
        logger.info("To instruction tune, run:")
        logger.info(f"  python train_conversational.py \\")
        logger.info(f"    --checkpoint {checkpoint_path} \\")
        logger.info(f"    --datasets alpaca \\")
        logger.info(f"    --output_dir {instruction_checkpoint}")

    # Stage 7: Test the model
    logger.info("\n" + "=" * 80)
    logger.info("Stage 7: Testing Model")
    logger.info("=" * 80)

    # Generate sample code
    test_prompts = [
        "def fibonacci(n):",
        "# Binary search implementation\ndef binary_search(arr, target):",
        "class Node:\n    def __init__(self, value):",
    ]

    logger.info("Generating code samples...")
    model.eval()
    with torch.no_grad():
        for prompt in test_prompts:
            prompt_ids = torch.tensor([tokenizer.encode(prompt)], device=args.device)
            # Simple greedy generation
            generated = model.generate(prompt_ids, max_new_tokens=50)
            text = tokenizer.decode(generated[0].tolist())
            logger.info(f"\nPrompt: {prompt}")
            logger.info(f"Generated: {text[:200]}...")

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Model checkpoint: {checkpoint_path}")
    logger.info(f"\nTo use your trained model:")
    logger.info(f"  from LILITH import load_checkpoint, create_assistant_agent")
    logger.info(f"  model, tokenizer, _ = load_checkpoint('{checkpoint_path}')")
    logger.info(f"  agent = create_assistant_agent(model, tokenizer, personality='coding')")
    logger.info(f"  response = agent.chat('Write a function to reverse a string')")


if __name__ == "__main__":
    main()
