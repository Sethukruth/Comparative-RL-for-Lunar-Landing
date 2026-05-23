"""
Deep Q-Network architecture for LunarLander-v3.

A fully-connected feedforward network that maps an 8-dimensional state vector
to Q-values for each of the 4 discrete actions.

Architecture: 8 → 256 → 256 → 128 → 4  (ReLU activations)
"""

from typing import List

import torch
import torch.nn as nn


class DQNetwork(nn.Module):
    """Vanilla DQN value network.

    Args:
        state_size: Dimensionality of the observation space (8 for LunarLander).
        action_size: Number of discrete actions (4 for LunarLander).
        hidden_layers: Sizes of hidden layers. Defaults to ``[256, 256, 128]``.
    """

    def __init__(
        self,
        state_size: int = 8,
        action_size: int = 4,
        hidden_layers: List[int] | None = None,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [256, 256, 128]

        layers: list[nn.Module] = []
        prev_size = state_size

        for h in hidden_layers:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h

        layers.append(nn.Linear(prev_size, action_size))

        self.network = nn.Sequential(*layers)

        # Xavier-uniform initialization
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for every action given a batch of states.

        Args:
            x: Tensor of shape ``(batch, state_size)``.

        Returns:
            Tensor of shape ``(batch, action_size)``.
        """
        return self.network(x)
