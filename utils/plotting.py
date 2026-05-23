"""
Visualization utilities for reward curves, loss curves, and algorithm comparisons.

All plots are saved to the ``results/`` directory and use a consistent dark
theme for a polished, portfolio-ready appearance.
"""

import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Global plot styling ──────────────────────────────────────────────────────

plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "legend.facecolor": "#161b22",
        "legend.edgecolor": "#30363d",
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 11,
    }
)

PALETTE = {
    "DQN": "#58a6ff",
    "DDQN": "#f78166",
    "PPO": "#7ee787",
}


def _smooth(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Apply a centred moving-average with the given window size."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


# ── Single-algorithm plots ───────────────────────────────────────────────────


def plot_rewards(
    rewards: List[float],
    algorithm: str,
    save_dir: str = "results",
    window: int = 50,
) -> str:
    """Plot and save a reward curve for a single algorithm.

    Args:
        rewards: Per-episode reward values.
        algorithm: Algorithm name (``DQN``, ``DDQN``, ``PPO``).
        save_dir: Output directory.
        window: Smoothing window size.

    Returns:
        Absolute path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()

    episodes = np.arange(len(rewards))
    color = PALETTE.get(algorithm.upper(), "#58a6ff")

    # Raw data (faint)
    ax.plot(episodes, rewards, alpha=0.15, color=color, linewidth=0.6)

    # Smoothed curve
    smoothed = _smooth(np.array(rewards), window)
    offset = (len(rewards) - len(smoothed)) // 2
    ax.plot(
        np.arange(offset, offset + len(smoothed)),
        smoothed,
        color=color,
        linewidth=2.0,
        label=f"{algorithm} (smoothed)",
    )

    # Solved threshold
    ax.axhline(y=200, color="#f0883e", linestyle="--", alpha=0.6, label="Solved (200)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title(f"{algorithm} — Training Rewards")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"{algorithm.lower()}_rewards.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return os.path.abspath(path)


def plot_losses(
    losses: List[float],
    algorithm: str,
    save_dir: str = "results",
    window: int = 100,
) -> str:
    """Plot and save a loss curve for a single algorithm.

    Args:
        losses: Per-step loss values.
        algorithm: Algorithm name.
        save_dir: Output directory.
        window: Smoothing window size.

    Returns:
        Absolute path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()

    color = PALETTE.get(algorithm.upper(), "#58a6ff")
    steps = np.arange(len(losses))

    ax.plot(steps, losses, alpha=0.10, color=color, linewidth=0.4)

    smoothed = _smooth(np.array(losses), window)
    offset = (len(losses) - len(smoothed)) // 2
    ax.plot(
        np.arange(offset, offset + len(smoothed)),
        smoothed,
        color=color,
        linewidth=2.0,
        label=f"{algorithm} loss (smoothed)",
    )

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{algorithm} — Training Loss")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, f"{algorithm.lower()}_loss.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return os.path.abspath(path)


# ── Comparison plot ──────────────────────────────────────────────────────────


def plot_comparison(
    reward_dict: Dict[str, List[float]],
    save_dir: str = "results",
    window: int = 50,
) -> str:
    """Overlay smoothed reward curves for multiple algorithms.

    Args:
        reward_dict: Mapping of ``{algorithm_name: [rewards]}``.
        save_dir: Output directory.
        window: Smoothing window size.

    Returns:
        Absolute path to the saved figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for algo, rewards in reward_dict.items():
        color = PALETTE.get(algo.upper(), "#58a6ff")
        smoothed = _smooth(np.array(rewards), window)
        offset = (len(rewards) - len(smoothed)) // 2
        ax.plot(
            np.arange(offset, offset + len(smoothed)),
            smoothed,
            color=color,
            linewidth=2.0,
            label=algo,
        )

    ax.axhline(y=200, color="#f0883e", linestyle="--", alpha=0.6, label="Solved (200)")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Algorithm Comparison — Training Rewards")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    path = os.path.join(save_dir, "comparison.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return os.path.abspath(path)
