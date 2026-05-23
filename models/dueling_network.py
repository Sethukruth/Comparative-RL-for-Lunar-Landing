"""
Dueling DQN network architecture (optional advanced variant).

Splits the value stream into a *state-value* V(s) head and an
*advantage* A(s,a) head, then combines them:

    Q(s, a) = V(s) + A(s, a) − mean(A(s, ·))

This can accelerate learning by letting the network learn which states
are valuable independently of the action taken.
"""

from typing import List

import torch
import torch.nn as nn


class DuelingNetwork(nn.Module):
    """Dueling DQN architecture.

    Args:
        state_size: Dimensionality of the observation space.
        action_size: Number of discrete actions.
        hidden_layers: Sizes of the shared feature layers.
    """

    def __init__(
        self,
        state_size: int = 8,
        action_size: int = 4,
        hidden_layers: List[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256]

        # Shared feature extractor
        shared: list[nn.Module] = []
        prev = state_size
        for h in hidden_layers:
            shared.append(nn.Linear(prev, h))
            shared.append(nn.ReLU())
            prev = h
        self.feature = nn.Sequential(*shared)

        # Value stream  → scalar V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(prev, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        # Advantage stream → A(s, a) for each action
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values using dueling decomposition.

        Args:
            x: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, action_size)``.
        """
        features = self.feature(x)
        value = self.value_stream(features)              # (batch, 1)
        advantage = self.advantage_stream(features)      # (batch, action_size)
        # Mean-centering for identifiability
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values
