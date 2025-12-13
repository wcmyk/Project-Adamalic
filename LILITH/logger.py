"""Logging utilities for LILITH training and evaluation."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class LILITHLogger:
    """Custom logger for LILITH with console and file output."""

    def __init__(
        self,
        name: str = "LILITH",
        log_file: Optional[str] = None,
        level: int = logging.INFO,
        console_output: bool = True,
    ):
        """Initialize logger.

        Args:
            name: Logger name
            log_file: Path to log file (optional)
            level: Logging level
            console_output: Whether to output to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False

        # Remove existing handlers
        self.logger.handlers.clear()

        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message."""
        self.logger.critical(message)

    def log_training_step(
        self,
        step: int,
        loss: float,
        lr: float,
        **kwargs,
    ):
        """Log training step with metrics.

        Args:
            step: Training step
            loss: Loss value
            lr: Learning rate
            **kwargs: Additional metrics to log
        """
        metrics_str = f"step={step} loss={loss:.4f} lr={lr:.6f}"

        for key, value in kwargs.items():
            if isinstance(value, float):
                metrics_str += f" {key}={value:.4f}"
            else:
                metrics_str += f" {key}={value}"

        self.logger.info(metrics_str)

    def log_evaluation(self, metrics: dict, prefix: str = "eval"):
        """Log evaluation metrics.

        Args:
            metrics: Dictionary of metrics
            prefix: Prefix for log message
        """
        metrics_str = f"{prefix}:"
        for key, value in metrics.items():
            if isinstance(value, float):
                metrics_str += f" {key}={value:.4f}"
            elif isinstance(value, int):
                metrics_str += f" {key}={value}"

        self.logger.info(metrics_str)

    def log_model_info(self, model, config):
        """Log model architecture and configuration.

        Args:
            model: The model
            config: Model configuration
        """
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info("=" * 60)
        self.logger.info("MODEL INFORMATION")
        self.logger.info("=" * 60)
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Model config: {config}")
        self.logger.info("=" * 60)


def create_logger(
    name: str = "LILITH",
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> LILITHLogger:
    """Create a logger with automatic log file naming.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level

    Returns:
        Configured logger
    """
    log_file = None
    if log_dir:
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(log_dir_path / f"{name}_{timestamp}.log")

    return LILITHLogger(name=name, log_file=log_file, level=level)


__all__ = ["LILITHLogger", "create_logger"]
