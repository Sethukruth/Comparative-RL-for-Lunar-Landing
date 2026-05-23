"""
Double DQN Agent.

The key difference from vanilla DQN: action selection uses the **online**
(policy) network while action *evaluation* uses the **target** network.
This decoupling reduces the maximization bias inherent in standard DQN.

    Q_target = r + γ · Q_target(s', argmax_a Q_online(s', a))

Everything else (ε-greedy, replay buffer, hard target updates) is identical
to the standard DQN agent.
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.dqn_network import DQNetwork
from memory.replay_buffer import ReplayBuffer
from utils.logger import get_logger

log = get_logger("DDQNAgent")


class DDQNAgent:
    """Double Deep Q-Network agent.

    Args:
        state_size: Observation dimensionality.
        action_size: Number of discrete actions.
        hidden_layers: Network hidden-layer sizes.
        lr: Learning rate.
        gamma: Discount factor.
        batch_size: Mini-batch size.
        memory_capacity: Replay buffer capacity.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Multiplicative decay per episode.
        target_update_freq: Episodes between target-net syncs.
        min_samples: Minimum buffer size before training begins.
        device: Torch device.
        tb_log_dir: TensorBoard log directory.
    """

    def __init__(
        self,
        state_size: int = 8,
        action_size: int = 4,
        hidden_layers: list[int] | None = None,
        lr: float = 5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        memory_capacity: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 5,
        min_samples: int = 1000,
        device: torch.device | None = None,
        tb_log_dir: str = "tensorboard_logs/ddqn",
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.min_samples = min_samples

        self.device = device or torch.device("cpu")

        # Networks
        self.policy_net = DQNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.target_net = DQNetwork(state_size, action_size, hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimiser & loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()

        # Replay memory
        self.memory = ReplayBuffer(capacity=memory_capacity)

        # TensorBoard
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir)

        # Bookkeeping
        self.train_step = 0
        self.losses: list[float] = []

    # ── Action selection ─────────────────────────────────────────────────

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select an action using ε-greedy exploration."""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Memory ───────────────────────────────────────────────────────────

    def remember(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition."""
        self.memory.push(state, action, reward, next_state, done)

    # ── Training (DDQN core) ─────────────────────────────────────────────

    def learn(self) -> Optional[float]:
        """Gradient step using the **Double DQN** target.

        Standard DQN:
            target = r + γ · max_a' Q_target(s', a')

        Double DQN:
            a* = argmax_a' Q_online(s', a')
            target = r + γ · Q_target(s', a*)

        Returns:
            Training loss, or ``None`` if buffer too small.
        """
        if len(self.memory) < self.min_samples:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Current Q-values for chosen actions
        q_values = self.policy_net(states).gather(1, actions)

        # ── Double DQN target ────────────────────────────────────────────
        with torch.no_grad():
            # Action selection  → online (policy) network
            best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Action evaluation → target network
            next_q = self.target_net(next_states).gather(1, best_actions)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.losses.append(loss_val)
        self.train_step += 1
        self.writer.add_scalar("train/loss", loss_val, self.train_step)

        return loss_val

    # ── Housekeeping ─────────────────────────────────────────────────────

    def update_target(self) -> None:
        """Hard-copy policy-net weights into target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        log.info("Target network updated (DDQN).")

    def decay_epsilon(self) -> None:
        """Multiplicative epsilon decay."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def log_episode(self, episode: int, reward: float) -> None:
        """Write per-episode metrics to TensorBoard."""
        self.writer.add_scalar("train/episode_reward", reward, episode)
        self.writer.add_scalar("train/epsilon", self.epsilon, episode)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save policy-net weights."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.policy_net.state_dict(), path)
        log.info(f"Model saved -> {path}")

    def load(self, path: str) -> None:
        """Load policy-net weights and sync target net."""
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.target_net.load_state_dict(self.policy_net.state_dict())
        log.info(f"Model loaded <- {path}")

    def close(self) -> None:
        """Flush TensorBoard writer."""
        self.writer.close()
