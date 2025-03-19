"""Logging utilities for HOLA components."""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logging(name: str, level: int = logging.INFO, log_dir: Path = None) -> logging.Logger:
    """Configure logging for a component with console and file handlers.

    Args:
        name: Name of the logger
        level: Logging level
        log_dir: Directory for log files (uses current directory if None)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers if they already exist
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to parent

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create file handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'{name.lower()}_{timestamp}.log'

    if log_dir:
        log_dir.mkdir(exist_ok=True, parents=True)
        file_path = log_dir / filename
    else:
        file_path = filename

    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger