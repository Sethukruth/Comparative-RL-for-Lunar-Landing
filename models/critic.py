"""
Critic network for PPO (reference implementation).

Maps an 8-dimensional state to a single scalar state-value V(s).
Uses Tanh activations in the hidden layers to match the Stable-Baselines3
default ``MlpPolicy`` architecture.

Architecture: 8 → 256 (Tanh) → 256 (Tanh) → 1
"""

import torch
import torch.nn as nn


class Critic(nn.Module):
    """Value (critic) head for PPO.

    Args:
        state_size: Observation dimensionality.
    """

    def __init__(self, state_size: int = 8) -> None:
        super().__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the estimated state value V(s).

        Args:
            x: State tensor of shape ``(batch, state_size)``.

        Returns:
            Scalar value of shape ``(batch, 1)``.
        """
        return self.critic(x)
