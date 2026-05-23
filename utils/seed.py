"""
Reproducibility utilities for deterministic experiment runs.

Sets seeds across all random number generators used in the project:
Python stdlib, NumPy, PyTorch (CPU + CUDA), and Gymnasium.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: int = 42) -> None:
    """Set seed across all random number generators for reproducibility.

    Args:
        seed: Integer seed value. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic algorithms (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Return the best available compute device.

    Returns:
        torch.device: CUDA device if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
