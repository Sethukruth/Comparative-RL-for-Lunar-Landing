"""
Evaluation metrics for Lunar Lander experiments.

Provides structured metric collection, aggregation, and JSON serialization
for evaluation runs (average reward, success/crash rates, landing accuracy).
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any

import numpy as np


@dataclass
class EpisodeResult:
    """Outcome of a single evaluation episode."""

    reward: float
    landed: bool
    crashed: bool
    x_position: float
    y_position: float
    steps: int


class EvaluationMetrics:
    """Collect and aggregate evaluation statistics over multiple episodes.

    Attributes:
        results: List of per-episode results.
    """

    def __init__(self) -> None:
        self.results: List[EpisodeResult] = []

    # ── Recording ────────────────────────────────────────────────────────

    def add_episode(
        self,
        reward: float,
        landed: bool,
        crashed: bool,
        x_position: float = 0.0,
        y_position: float = 0.0,
        steps: int = 0,
    ) -> None:
        """Append a single episode result."""
        self.results.append(
            EpisodeResult(
                reward=reward,
                landed=landed,
                crashed=crashed,
                x_position=x_position,
                y_position=y_position,
                steps=steps,
            )
        )

    # ── Aggregation ──────────────────────────────────────────────────────

    @property
    def rewards(self) -> np.ndarray:
        return np.array([r.reward for r in self.results])

    @property
    def average_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.results else 0.0

    @property
    def median_reward(self) -> float:
        return float(np.median(self.rewards)) if self.results else 0.0

    @property
    def max_reward(self) -> float:
        return float(np.max(self.rewards)) if self.results else 0.0

    @property
    def min_reward(self) -> float:
        return float(np.min(self.rewards)) if self.results else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards)) if self.results else 0.0

    @property
    def success_rate(self) -> float:
        """Fraction of episodes where the lander touched down safely."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.landed) / len(self.results)

    @property
    def crash_rate(self) -> float:
        """Fraction of episodes where the lander crashed."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.crashed) / len(self.results)

    @property
    def landing_accuracy(self) -> float:
        """Fraction of successful landings that were on the pad (|x| < 0.2)."""
        landed = [r for r in self.results if r.landed]
        if not landed:
            return 0.0
        on_pad = sum(1 for r in landed if abs(r.x_position) < 0.2)
        return on_pad / len(landed)

    # ── Serialization ────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Return a JSON-serializable summary dictionary."""
        return {
            "num_episodes": len(self.results),
            "average_reward": round(self.average_reward, 2),
            "median_reward": round(self.median_reward, 2),
            "max_reward": round(self.max_reward, 2),
            "min_reward": round(self.min_reward, 2),
            "std_reward": round(self.std_reward, 2),
            "success_rate": round(self.success_rate * 100, 2),
            "crash_rate": round(self.crash_rate * 100, 2),
            "landing_accuracy": round(self.landing_accuracy * 100, 2),
        }

    def save(self, path: str) -> None:
        """Persist summary to a JSON file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def __repr__(self) -> str:
        s = self.summary()
        lines = [
            f"  Episodes       : {s['num_episodes']}",
            f"  Avg Reward     : {s['average_reward']}",
            f"  Median Reward  : {s['median_reward']}",
            f"  Max Reward     : {s['max_reward']}",
            f"  Min Reward     : {s['min_reward']}",
            f"  Std Reward     : {s['std_reward']}",
            f"  Success Rate   : {s['success_rate']}%",
            f"  Crash Rate     : {s['crash_rate']}%",
            f"  Landing Acc.   : {s['landing_accuracy']}%",
        ]
        return "EvaluationMetrics(\n" + "\n".join(lines) + "\n)"
