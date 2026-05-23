"""
Actor network for PPO (reference implementation).

Maps an 8-dimensional state to 4 action logits using Tanh activations
in the hidden layers. This mirrors the MlpPolicy architecture used by
Stable-Baselines3, and is provided for educational reference.

Architecture: 8 → 256 (Tanh) → 256 (Tanh) → 4
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Policy (actor) head for PPO.

    Args:
        state_size: Observation dimensionality.
        action_size: Number of discrete actions.
    """

    def __init__(self, state_size: int = 8, action_size: int = 4) -> None:
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw action logits.

        Args:
            x: State tensor of shape ``(batch, state_size)``.

        Returns:
            Logits of shape ``(batch, action_size)``.
        """
        return self.actor(x)
