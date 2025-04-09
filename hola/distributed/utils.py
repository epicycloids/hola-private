"""
Utility functions for the distributed optimization components.
"""

import logging
import sys
import os
from datetime import datetime


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure logging for a component with console and file handlers.

    Args:
        name: Name of the component (e.g., 'Scheduler', 'Worker-1').
        level: Logging level (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding multiple handlers if logger already exists and is configured
    if logger.hasHandlers():
        return logger

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create logs directory if it doesn't exist
    # Use a more robust way to determine the project root if possible,
    # but for now, assume it's relative to this file's location.
    try:
        # Find the project root assuming 'hola' is a direct subdirectory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir)) # Go up two levels (utils.py -> distributed -> hola)
        logs_dir = os.path.join(project_root, 'logs')
        os.makedirs(logs_dir, exist_ok=True)

        # Create file handler in the logs directory
        # Use a consistent naming scheme, perhaps just the component name
        log_file_path = os.path.join(logs_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Log an error if file logging setup fails, but continue with console logging
        logger.error(f"Failed to set up file logging for {name}: {e}", exc_info=True)


    return logger