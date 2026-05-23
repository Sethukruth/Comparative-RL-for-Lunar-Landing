"""
Proportional Prioritized Experience Replay (PER).

Higher-TD-error transitions are sampled more frequently, which can speed
up learning. Uses a simple array-based implementation suitable for the
scale of LunarLander experiments.

Reference:
    Schaul et al., "Prioritized Experience Replay", ICLR 2016.
"""

import random
from typing import List, Tuple

import numpy as np
import torch


class PrioritizedReplayBuffer:
    """Sum-tree–free proportional PER (array implementation).

    Good enough for buffers up to ~200k. For larger buffers, swap in
    a proper segment-tree.

    Args:
        capacity: Maximum number of transitions.
        alpha: Priority exponent (0 = uniform, 1 = full prioritization).
        beta_start: Initial importance-sampling exponent.
        beta_frames: Number of frames over which beta is annealed to 1.
        seed: RNG seed.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        seed: int = 42,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames

        self.buffer: List = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.frame = 0

        self._rng = np.random.RandomState(seed)

    # ── Storage ──────────────────────────────────────────────────────────

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition with maximum existing priority."""
        max_prio = self.priorities[: len(self.buffer)].max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    # ── Sampling ─────────────────────────────────────────────────────────

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        np.ndarray,
        np.ndarray,
    ]:
        """Sample a prioritized mini-batch.

        Returns:
            ``(states, actions, rewards, next_states, dones, indices, weights)``
        """
        self.frame += 1
        n = len(self.buffer)

        prios = self.priorities[:n] ** self.alpha
        probs = prios / prios.sum()

        indices = self._rng.choice(n, batch_size, replace=False, p=probs)

        beta = min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)
        weights = (n * probs[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(device),
            indices,
            torch.FloatTensor(weights).unsqueeze(1).to(device),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after a training step.

        Args:
            indices: Indices returned by :meth:`sample`.
            td_errors: Absolute TD errors for the sampled transitions.
        """
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6

    def __len__(self) -> int:
        return len(self.buffer)

    def __repr__(self) -> str:
        return (
            f"PrioritizedReplayBuffer(size={len(self)}, "
            f"capacity={self.capacity}, α={self.alpha})"
        )
