"""
Train a PPO agent on LunarLander-v3 using Stable-Baselines3.

Usage:
    python train_ppo.py                          # defaults from configs/ppo.yaml
    python train_ppo.py --config configs/ppo.yaml --timesteps 500000
"""

import argparse
import json
import os

import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.callbacks import BaseCallback

from agents.ppo_agent import PPOAgent
from utils.seed import set_global_seed
from utils.plotting import plot_rewards
from utils.logger import get_logger

log = get_logger("train_ppo")


# ── Reward-tracking callback ────────────────────────────────────────────────

class RewardTracker(BaseCallback):
    """SB3 callback that records per-episode rewards for plotting."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        # Monitor wrapper stores episode info in `infos`
        for info in self.locals.get("infos", []):
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_rewards.append(ep_info["r"])
        return True


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg: dict) -> None:
    """Run PPO training."""

    seed = cfg["env"]["seed"]
    set_global_seed(seed)
    log.info(f"Seed: {seed}")

    agent = PPOAgent(
        env_name=cfg["env"]["name"],
        policy=cfg["training"]["policy"],
        learning_rate=cfg["training"]["learning_rate"],
        gamma=cfg["training"]["gamma"],
        n_steps=cfg["training"]["n_steps"],
        batch_size=cfg["training"]["batch_size"],
        n_epochs=cfg["training"]["n_epochs"],
        clip_range=cfg["training"]["clip_range"],
        ent_coef=cfg["training"]["ent_coef"],
        vf_coef=cfg["training"]["vf_coef"],
        max_grad_norm=cfg["training"]["max_grad_norm"],
        gae_lambda=cfg["training"]["gae_lambda"],
        seed=seed,
        tb_log_dir=cfg["logging"]["tensorboard_dir"],
        verbose=cfg["logging"]["verbose"],
    )

    total_timesteps = cfg["training"]["total_timesteps"]
    results_dir = cfg["logging"]["results_dir"]
    best_path = cfg["checkpoint"]["best_model_path"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cfg["checkpoint"]["save_dir"], exist_ok=True)

    tracker = RewardTracker()

    log.info(f"Starting PPO training for {total_timesteps:,} timesteps …")
    agent.model.learn(
        total_timesteps=total_timesteps,
        callback=tracker,
        progress_bar=True,
    )

    agent.save(best_path)
    log.info("PPO training complete.")

    # ── Post-training outputs ────────────────────────────────────────────
    rewards = tracker.episode_rewards
    if rewards:
        plot_rewards(rewards, "PPO", save_dir=results_dir)
        np.save(os.path.join(results_dir, "ppo_rewards.npy"), np.array(rewards))

    summary = {
        "algorithm": "PPO",
        "total_timesteps": total_timesteps,
        "episodes_completed": len(rewards),
        "best_avg_reward": round(float(np.mean(sorted(rewards)[-100:])), 2) if rewards else 0,
        "final_avg_100": round(float(np.mean(rewards[-100:])), 2) if len(rewards) >= 100 else round(float(np.mean(rewards)), 2) if rewards else 0,
    }
    with open(os.path.join(results_dir, "ppo_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    agent.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on LunarLander-v3")
    parser.add_argument("--config", type=str, default="configs/ppo.yaml")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.timesteps is not None:
        cfg["training"]["total_timesteps"] = args.timesteps
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed

    train(cfg)


if __name__ == "__main__":
    main()
