"""Train LILITH on Wikipedia - Production-Ready Training Script.

This script demonstrates training a real language model on Wikipedia data.
Includes all Phase 2 features: mixed precision, gradient accumulation, validation, etc.

Usage:
    # Small model (10M params) on 1% of Wikipedia - great for testing
    python train_wikipedia.py --subset 0.01 --model_size small

    # Medium model (50M params) on full Wikipedia
    python train_wikipedia.py --model_size medium --gpus 2

    # Large model (150M params) with all features
    python train_wikipedia.py --model_size large --gpus 4 --use_flash_attention
"""
import argparse
from pathlib import Path

import torch
from LILITH import (
    get_small_model_config,
    get_medium_model_config,
    get_large_model_config,
    BPETokenizer,
    create_logger,
)
from LILITH.datasets import get_wikipedia_corpus
from LILITH.config_phase2 import Phase2TrainingConfig
from LILITH.train_phase2 import train_phase2
from LILITH.optimization import get_parameter_groups


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LILITH on Wikipedia")

    # Model configuration
    parser.add_argument(
        '--model_size',
        type=str,
        default='small',
        choices=['small', 'medium', 'large'],
        help='Model size preset (default: small)'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=10000,
        help='BPE vocabulary size (default: 10000)'
    )

    # Data configuration
    parser.add_argument(
        '--subset',
        type=float,
        default=1.0,
        help='Fraction of Wikipedia to use (0.01 = 1%%, default: 1.0 = 100%%)'
    )
    parser.add_argument(
        '--language',
        type=str,
        default='en',
        help='Wikipedia language code (default: en)'
    )

    # Training configuration
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size (default: 32)'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=50000,
        help='Maximum training steps (default: 50000)'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    parser.add_argument(
        '--warmup_steps',
        type=int,
        default=2000,
        help='Warmup steps (default: 2000)'
    )

    # Advanced features
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        default=True,
        help='Use mixed precision training (default: True)'
    )
    parser.add_argument(
        '--gradient_accumulation',
        type=int,
        default=1,
        help='Gradient accumulation steps (default: 1)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs (default: 1)'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/wikipedia',
        help='Output directory (default: checkpoints/wikipedia)'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='logs/wikipedia',
        help='Log directory (default: logs/wikipedia)'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Create logger
    logger = create_logger(name="Wikipedia-Training", log_dir=args.log_dir)
    logger.info("=" * 80)
    logger.info("LILITH WIKIPEDIA TRAINING - PHASE 2")
    logger.info("=" * 80)

    # Log configuration
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Dataset: Wikipedia ({args.language}) - {args.subset * 100:.1f}%")
    logger.info(f"Vocabulary: {args.vocab_size} BPE tokens")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max steps: {args.max_steps}")
    logger.info(f"GPUs: {args.gpus}")
    logger.info("=" * 80)

    # Step 1: Load Wikipedia dataset
    logger.info("\n[1/5] Loading Wikipedia dataset...")
    wiki_corpus = get_wikipedia_corpus(
        language=args.language,
        subset_percentage=args.subset,
        streaming=True,
    )
    logger.info("✓ Wikipedia dataset loaded (streaming mode)")

    # Step 2: Train BPE tokenizer
    logger.info(f"\n[2/5] Training BPE tokenizer (vocab_size={args.vocab_size})...")
    tokenizer_path = Path(args.output_dir) / "tokenizer.json"

    if tokenizer_path.exists():
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = BPETokenizer.load(str(tokenizer_path))
    else:
        logger.info("Training new tokenizer on Wikipedia data...")
        # Sample subset for tokenizer training (don't need all data)
        tokenizer_corpus = []
        for i, text in enumerate(wiki_corpus):
            tokenizer_corpus.append(text)
            if i >= 10000:  # 10k articles is enough for vocab
                break

        tokenizer = BPETokenizer(vocab_size=args.vocab_size, min_frequency=2)
        tokenizer.train(tokenizer_corpus)

        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))
        logger.info(f"✓ Tokenizer trained and saved to {tokenizer_path}")
        logger.info(f"  Actual vocabulary size: {tokenizer.vocab_size_actual}")

    # Step 3: Select model configuration
    logger.info(f"\n[3/5] Configuring {args.model_size} model...")
    if args.model_size == 'small':
        model_config = get_small_model_config()
    elif args.model_size == 'medium':
        model_config = get_medium_model_config()
    else:  # large
        model_config = get_large_model_config()

    # Update vocab size to match tokenizer
    model_config.vocab_size = tokenizer.vocab_size_actual

    logger.info(f"  Model dimension: {model_config.d_model}")
    logger.info(f"  Layers: {model_config.n_layers}")
    logger.info(f"  Heads: {model_config.n_heads}")
    logger.info(f"  Max sequence length: {model_config.max_seq_len}")

    # Calculate approximate parameter count
    approx_params = (
        model_config.vocab_size * model_config.d_model +  # Token embeddings
        model_config.max_seq_len * model_config.d_model +  # Position embeddings
        model_config.n_layers * (
            4 * model_config.d_model * model_config.d_model +  # Attention
            4 * model_config.d_model * model_config.d_ff  # FFN
        )
    )
    logger.info(f"  Estimated parameters: {approx_params / 1e6:.1f}M")

    # Step 4: Configure training
    logger.info(f"\n[4/5] Configuring Phase 2 training...")
    train_config = Phase2TrainingConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        grad_clip=1.0,
        log_interval=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Phase 2 features
        use_mixed_precision=args.mixed_precision and torch.cuda.is_available(),
        gradient_accumulation_steps=args.gradient_accumulation,
        lr_scheduler="cosine",
        min_lr=1e-6,
        # Evaluation and checkpointing
        eval_interval=1000,
        save_interval=5000,
        checkpoint_dir=args.output_dir,
        keep_last_n_checkpoints=3,
        # Early stopping
        use_early_stopping=True,
        early_stopping_patience=5,
        early_stopping_min_delta=0.001,
        # Logging
        log_dir=args.log_dir,
        # Data
        validation_split=0.05,
        shuffle=True,
    )

    logger.info(f"  Mixed precision: {train_config.use_mixed_precision}")
    logger.info(f"  Gradient accumulation: {train_config.gradient_accumulation_steps}")
    logger.info(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Validation split: {train_config.validation_split * 100:.1f}%")

    # Step 5: Train model
    logger.info(f"\n[5/5] Starting training...")
    logger.info("=" * 80)

    # Reload corpus for training (streaming)
    train_corpus = get_wikipedia_corpus(
        language=args.language,
        subset_percentage=args.subset,
        streaming=True,
    )

    model = train_phase2(
        corpus=train_corpus,
        tokenizer=tokenizer,
        model_config=model_config,
        train_config=train_config,
        block_size=min(512, model_config.max_seq_len),
        checkpoint_path=Path(args.output_dir) / "final_model.pt",
    )

    logger.info("=" * 80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info(f"  Model saved to: {args.output_dir}/final_model.pt")
    logger.info(f"  Tokenizer saved to: {tokenizer_path}")
    logger.info(f"  Logs saved to: {args.log_dir}")
    logger.info("=" * 80)

    # Optional: Generate sample text
    logger.info("\n[BONUS] Generating sample text...")
    prompt = "Artificial intelligence is"
    prompt_ids = torch.tensor([tokenizer.encode(prompt)])

    if torch.cuda.is_available():
        model = model.cuda()
        prompt_ids = prompt_ids.cuda()

    model.eval()
    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=100, temperature=0.8)

    generated_text = tokenizer.decode(generated[0].tolist())
    logger.info(f"\nPrompt: {prompt}")
    logger.info(f"Generated:\n{generated_text}\n")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
