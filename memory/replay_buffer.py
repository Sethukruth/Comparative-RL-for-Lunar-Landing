"""
Uniform experience-replay buffer for DQN / DDQN.

Stores (s, a, r, s', done) transitions in a fixed-capacity ring buffer
and supports uniform random sampling for mini-batch training.
"""

import random
from collections import deque
from typing import List, Tuple

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size replay memory with uniform sampling.

    Args:
        capacity: Maximum number of transitions to store.
        seed: RNG seed for reproducibility.
    """

    def __init__(self, capacity: int = 100_000, seed: int = 42) -> None:
        self.memory: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a single transition.

        Args:
            state: Current observation.
            action: Action taken.
            reward: Reward received.
            next_state: Next observation.
            done: Whether the episode ended.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch and return tensors on ``device``.

        Args:
            batch_size: Number of transitions to sample.
            device: Target device (cpu / cuda).

        Returns:
            Tuple of ``(states, actions, rewards, next_states, dones)``
            tensors.
        """
        batch = self._rng.sample(list(self.memory), batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(np.array(actions)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1).to(device),
        )

    def __len__(self) -> int:
        return len(self.memory)

    def __repr__(self) -> str:
        return f"ReplayBuffer(size={len(self)}, capacity={self.memory.maxlen})"