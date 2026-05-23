"""
PPO Agent wrapper around Stable-Baselines3.

Provides a thin, consistent interface for training, evaluation, and
checkpoint management so that PPO can be used alongside the custom
DQN/DDQN agents with the same ``train → evaluate`` workflow.
"""

import os
from typing import Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from utils.logger import get_logger

log = get_logger("PPOAgent")


class PPOAgent:
    """Stable-Baselines3 PPO wrapper with consistent API.

    Args:
        env_name: Gymnasium environment ID.
        policy: SB3 policy string (``MlpPolicy``).
        learning_rate: Adam learning rate.
        gamma: Discount factor.
        n_steps: Rollout length per update.
        batch_size: Mini-batch size.
        n_epochs: Optimisation epochs per rollout.
        clip_range: PPO clipping parameter.
        ent_coef: Entropy bonus coefficient.
        vf_coef: Value-function loss coefficient.
        max_grad_norm: Gradient clipping threshold.
        gae_lambda: GAE λ.
        seed: Environment + algorithm seed.
        tb_log_dir: TensorBoard directory.
        verbose: SB3 verbosity (0 = silent, 1 = info).
    """

    def __init__(
        self,
        env_name: str = "LunarLander-v3",
        policy: str = "MlpPolicy",
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        clip_range: float = 0.2,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        gae_lambda: float = 0.95,
        seed: int = 42,
        tb_log_dir: str = "tensorboard_logs/ppo",
        verbose: int = 1,
    ) -> None:
        self.env_name = env_name
        self.seed = seed
        self.tb_log_dir = tb_log_dir

        # Wrapped training environment
        env = gym.make(env_name)
        self.env = Monitor(env)

        os.makedirs(tb_log_dir, exist_ok=True)

        self.model = PPO(
            policy=policy,
            env=self.env,
            learning_rate=learning_rate,
            gamma=gamma,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            clip_range=clip_range,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            gae_lambda=gae_lambda,
            seed=seed,
            tensorboard_log=tb_log_dir,
            verbose=verbose,
        )

        log.info("PPOAgent initialised.")

    # ── Training ─────────────────────────────────────────────────────────

    def train(
        self,
        total_timesteps: int = 1_000_000,
        checkpoint_dir: str = "checkpoints",
        checkpoint_freq: int = 50_000,
    ) -> None:
        """Run PPO training with periodic checkpointing.

        Args:
            total_timesteps: Total environment interactions.
            checkpoint_dir: Directory for periodic saves.
            checkpoint_freq: Timesteps between checkpoint saves.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_cb = CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=checkpoint_dir,
            name_prefix="ppo",
        )

        log.info(f"Starting PPO training for {total_timesteps:,} timesteps …")
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=checkpoint_cb,
            progress_bar=True,
        )
        log.info("PPO training complete.")

    # ── Action selection ─────────────────────────────────────────────────

    def act(self, state: np.ndarray, deterministic: bool = True) -> int:
        """Select an action using the learned policy.

        Args:
            state: Observation array.
            deterministic: If True, take the greedy action.

        Returns:
            Action index.
        """
        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action)

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save the full SB3 model (policy + optimizer + stats)."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.model.save(path)
        log.info(f"PPO model saved -> {path}")

    def load(self, path: str) -> None:
        """Load a saved SB3 model."""
        self.model = PPO.load(path, env=self.env)
        log.info(f"PPO model loaded <- {path}")

    @classmethod
    def from_checkpoint(cls, path: str, env_name: str = "LunarLander-v3") -> "PPOAgent":
        """Convenience constructor that loads a saved model.

        Args:
            path: Path to the ``.zip`` checkpoint.
            env_name: Environment name to attach.

        Returns:
            Ready-to-evaluate ``PPOAgent`` instance.
        """
        agent = cls.__new__(cls)
        agent.env_name = env_name
        env = gym.make(env_name)
        agent.env = Monitor(env)
        agent.model = PPO.load(path, env=agent.env)
        agent.seed = 42
        agent.tb_log_dir = "tensorboard_logs/ppo"
        log.info(f"PPOAgent restored from {path}")
        return agent

    def close(self) -> None:
        """Clean up environment."""
        self.env.close()
