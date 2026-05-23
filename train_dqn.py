"""
Train a DQN agent on LunarLander-v3.

Usage:
    python train_dqn.py                        # defaults from configs/dqn.yaml
    python train_dqn.py --config configs/dqn.yaml --episodes 500
"""

import argparse
import json
import os
import sys

import gymnasium as gym
import numpy as np
import yaml
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from utils.seed import set_global_seed, get_device
from utils.plotting import plot_rewards, plot_losses
from utils.logger import get_logger

log = get_logger("train_dqn")


def load_config(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train(cfg: dict) -> None:
    """Run the full DQN training loop."""

    # ── Setup ────────────────────────────────────────────────────────────
    seed = cfg["env"]["seed"]
    set_global_seed(seed)
    device = get_device()
    log.info(f"Device: {device} | Seed: {seed}")

    env = gym.make(cfg["env"]["name"])

    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        hidden_layers=cfg["network"]["hidden_layers"],
        lr=cfg["training"]["learning_rate"],
        gamma=cfg["training"]["gamma"],
        batch_size=cfg["training"]["batch_size"],
        memory_capacity=cfg["memory"]["capacity"],
        epsilon_start=cfg["exploration"]["epsilon_start"],
        epsilon_min=cfg["exploration"]["epsilon_min"],
        epsilon_decay=cfg["exploration"]["epsilon_decay"],
        target_update_freq=cfg["training"]["target_update_freq"],
        min_samples=cfg["memory"]["min_samples"],
        device=device,
        tb_log_dir=cfg["logging"]["tensorboard_dir"],
    )

    episodes = cfg["training"]["episodes"]
    max_steps = cfg["training"]["max_steps"]
    save_every = cfg["checkpoint"]["save_every"]
    best_path = cfg["checkpoint"]["best_model_path"]
    results_dir = cfg["logging"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cfg["checkpoint"]["save_dir"], exist_ok=True)

    # ── Training loop ────────────────────────────────────────────────────
    rewards_history: list[float] = []
    best_avg = -float("inf")
    solved_episode: int | None = None

    pbar = tqdm(range(1, episodes + 1), desc="DQN Training", unit="ep")
    for episode in pbar:
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

            if done:
                break

        rewards_history.append(total_reward)
        agent.decay_epsilon()
        agent.log_episode(episode, total_reward)

        # Target network sync
        if episode % agent.target_update_freq == 0:
            agent.update_target()

        # Rolling average
        avg_100 = np.mean(rewards_history[-100:])

        # Best model tracking
        if avg_100 > best_avg:
            best_avg = avg_100
            agent.save(best_path)

        # Periodic checkpoint
        if episode % save_every == 0:
            agent.save(os.path.join(cfg["checkpoint"]["save_dir"], f"dqn_ep{episode}.pth"))

        # Solved check
        if avg_100 >= 200.0 and solved_episode is None:
            solved_episode = episode
            log.info(f"✅ Environment SOLVED at episode {episode}! (avg={avg_100:.1f})")

        pbar.set_postfix(
            reward=f"{total_reward:.0f}",
            avg100=f"{avg_100:.1f}",
            eps=f"{agent.epsilon:.3f}",
        )

    # ── Post-training ────────────────────────────────────────────────────
    env.close()
    agent.close()

    plot_rewards(rewards_history, "DQN", save_dir=results_dir)
    if agent.losses:
        plot_losses(agent.losses, "DQN", save_dir=results_dir)

    # Save raw rewards for comparison script
    np.save(os.path.join(results_dir, "dqn_rewards.npy"), np.array(rewards_history))

    summary = {
        "algorithm": "DQN",
        "episodes": episodes,
        "solved_episode": solved_episode,
        "best_avg_reward": round(best_avg, 2),
        "final_avg_100": round(float(np.mean(rewards_history[-100:])), 2),
    }
    with open(os.path.join(results_dir, "dqn_training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Training complete. Best avg reward: {best_avg:.2f}")
    if solved_episode:
        log.info(f"Solved at episode {solved_episode}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train DQN on LunarLander-v3")
    parser.add_argument("--config", type=str, default="configs/dqn.yaml", help="YAML config path")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.episodes is not None:
        cfg["training"]["episodes"] = args.episodes
    if args.seed is not None:
        cfg["env"]["seed"] = args.seed

    train(cfg)


if __name__ == "__main__":
    main()
