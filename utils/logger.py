"""
Unified logging facade for training runs.

Wraps Python's logging module with a pre-configured formatter and supports
both console and file output so every training script gets consistent logs.
"""

import logging
import os
import sys
from typing import Optional


def get_logger(
    name: str,
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create or retrieve a logger with console and optional file handlers.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).
        log_dir: If provided, a ``{name}.log`` file is written here.
        level: Logging level. Defaults to ``logging.INFO``.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"{name}.log"),
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
