"""Train LILITH to be a conversational AI assistant.

This script implements instruction fine-tuning to make LILITH helpful and conversational.

Usage:
    # Basic instruction tuning on Alpaca
    python train_conversational.py --dataset alpaca --model_size small

    # Multi-dataset training
    python train_conversational.py --datasets alpaca dolly ultrachat --model_size medium

    # Fine-tune existing model
    python train_conversational.py --base_model checkpoints/wikipedia/final_model.pt
"""
import argparse
from pathlib import Path

import torch
from LILITH import (
    get_small_model_config,
    get_medium_model_config,
    create_logger,
)
from LILITH.instruction_data import load_instruction_dataset, SYSTEM_PROMPTS
from LILITH.config_phase2 import Phase2TrainingConfig
from LILITH.train_phase2 import train_phase2, load_checkpoint
from LILITH.datasets import CombinedDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train conversational AI")

    # Model
    parser.add_argument(
        '--model_size',
        type=str,
        default='small',
        choices=['small', 'medium', 'large'],
        help='Model size preset'
    )
    parser.add_argument(
        '--base_model',
        type=str,
        help='Path to pre-trained model to fine-tune'
    )

    # Data
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['alpaca'],
        choices=['alpaca', 'dolly', 'oasst1', 'ultrachat'],
        help='Instruction datasets to use'
    )
    parser.add_argument(
        '--format',
        type=str,
        default='alpaca',
        choices=['alpaca', 'chat'],
        help='Prompt format'
    )

    # Training
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=10000,
        help='Max training steps'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=2e-5,  # Lower LR for fine-tuning
        help='Learning rate'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='checkpoints/conversational',
        help='Output directory'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Create logger
    logger = create_logger(name="Conversational-Training", log_dir="logs/conversational")
    logger.info("=" * 80)
    logger.info("LILITH CONVERSATIONAL AI TRAINING - Phase 3")
    logger.info("=" * 80)

    # Log configuration
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Datasets: {', '.join(args.datasets)}")
    logger.info(f"Format: {args.format}")
    logger.info(f"Base model: {args.base_model or 'training from scratch'}")
    logger.info("=" * 80)

    # Step 1: Load/create model
    if args.base_model:
        logger.info(f"\n[1/4] Loading pre-trained model from {args.base_model}...")
        model, tokenizer, metadata = load_checkpoint(args.base_model)
        model_config = model.config
        logger.info(f"✓ Loaded model with {model.count_parameters():,} parameters")
    else:
        logger.info(f"\n[1/4] Creating {args.model_size} model from scratch...")
        if args.model_size == 'small':
            model_config = get_small_model_config()
        elif args.model_size == 'medium':
            model_config = get_medium_model_config()
        else:
            from LILITH import get_large_model_config
            model_config = get_large_model_config()

        model = None  # Will be created during training
        tokenizer = None

    # Step 2: Load instruction datasets
    logger.info(f"\n[2/4] Loading instruction datasets...")

    datasets = []
    for dataset_name in args.datasets:
        logger.info(f"  Loading {dataset_name}...")
        dataset = load_instruction_dataset(dataset_name, format_type=args.format)
        datasets.append(dataset)

    # Combine if multiple datasets
    if len(datasets) > 1:
        corpus = CombinedDataset(
            datasets=datasets,
            weights=[1.0] * len(datasets),  # Equal weighting
        )
        logger.info(f"✓ Combined {len(datasets)} datasets")
    else:
        corpus = datasets[0]
        logger.info(f"✓ Loaded single dataset")

    # Step 3: Configure training
    logger.info(f"\n[3/4] Configuring instruction tuning...")

    train_config = Phase2TrainingConfig(
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        lr=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=min(1000, args.max_steps // 10),
        grad_clip=1.0,
        log_interval=100,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # Phase 2 features
        use_mixed_precision=True,
        gradient_accumulation_steps=2,
        lr_scheduler="cosine",
        min_lr=1e-6,
        # Evaluation
        eval_interval=500,
        save_interval=2000,
        checkpoint_dir=args.output_dir,
        keep_last_n_checkpoints=3,
        # Early stopping
        use_early_stopping=True,
        early_stopping_patience=3,
        # Logging
        log_dir="logs/conversational",
        # Data
        validation_split=0.05,
    )

    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Max steps: {args.max_steps}")
    logger.info(f"  Mixed precision: {train_config.use_mixed_precision}")

    # Step 4: Train
    logger.info(f"\n[4/4] Starting instruction tuning...")
    logger.info("=" * 80)

    if model:
        # Fine-tuning existing model
        # Would need to implement continue_training function
        logger.info("Fine-tuning from checkpoint...")
        # For now, just log that this would happen
        logger.info("⚠ Fine-tuning from checkpoint not yet implemented")
        logger.info("  Training from scratch instead...")

    model = train_phase2(
        corpus=corpus,
        tokenizer=tokenizer if tokenizer else None,
        model_config=model_config,
        train_config=train_config,
        block_size=512,  # Longer context for conversations
        checkpoint_path=Path(args.output_dir) / "conversational_model.pt",
    )

    logger.info("=" * 80)
    logger.info("✓ INSTRUCTION TUNING COMPLETE!")
    logger.info(f"  Model saved to: {args.output_dir}/conversational_model.pt")
    logger.info("=" * 80)

    # Test the model
    logger.info("\n[BONUS] Testing conversational capabilities...")

    from LILITH.agent import create_assistant_agent

    agent = create_assistant_agent(model, tokenizer, personality="helpful")

    test_prompts = [
        "What is machine learning?",
        "Write a Python function to calculate fibonacci numbers",
        "Explain quantum computing in simple terms",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\nTest {i}: {prompt}")
        response = agent.chat(prompt, verbose=False)
        logger.info(f"Response: {response[:200]}...")

    logger.info("\n" + "=" * 80)
    logger.info("✓ LILITH is now a conversational AI assistant!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
