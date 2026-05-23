"""
Unified evaluation and GIF generation for DQN, DDQN, and PPO.

Usage:
    python evaluate.py --agent dqn   --model checkpoints/dqn_best.pth   --episodes 100
    python evaluate.py --agent ddqn  --model checkpoints/ddqn_best.pth  --episodes 100
    python evaluate.py --agent ppo   --model checkpoints/ppo_best       --episodes 100
    python evaluate.py --agent all   --episodes 100   # evaluate all three + comparison plot
"""

import argparse
import json
import os
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

from agents.dqn_agent import DQNAgent
from agents.ddqn_agent import DDQNAgent
from agents.ppo_agent import PPOAgent
from utils.metrics import EvaluationMetrics
from utils.recorder import Recorder
from utils.plotting import plot_comparison
from utils.seed import set_global_seed, get_device
from utils.logger import get_logger

log = get_logger("evaluate")

RESULTS_DIR = "results"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _classify_outcome(
    reward: float,
    terminated: bool,
    final_state: np.ndarray,
) -> tuple[bool, bool]:
    """Determine whether the lander successfully landed or crashed.

    Heuristics based on the LunarLander reward structure:
      • reward >= 200  → very likely a successful, on-pad landing
      • terminated AND reward < 0 → likely a crash
      • otherwise → timeout / off-pad landing

    Returns:
        ``(landed, crashed)`` booleans.
    """
    landed = reward >= 100.0
    crashed = terminated and reward < -50.0
    return landed, crashed


def evaluate_agent(
    agent_type: str,
    model_path: str,
    num_episodes: int = 100,
    record_gif: bool = True,
    seed: int = 42,
) -> EvaluationMetrics:
    """Evaluate a trained agent and optionally record a GIF.

    Args:
        agent_type: One of ``"dqn"``, ``"ddqn"``, ``"ppo"``.
        model_path: Path to the saved model checkpoint.
        num_episodes: Number of evaluation episodes.
        record_gif: Whether to record a GIF of the best episode.
        seed: Random seed.

    Returns:
        :class:`EvaluationMetrics` with all per-episode statistics.
    """
    set_global_seed(seed)
    device = get_device()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── Build agent ──────────────────────────────────────────────────────
    if agent_type in ("dqn", "ddqn"):
        AgentClass = DQNAgent if agent_type == "dqn" else DDQNAgent
        agent = AgentClass(device=device)
        agent.load(model_path)
    elif agent_type == "ppo":
        agent = PPOAgent.from_checkpoint(model_path)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # ── Evaluation loop ──────────────────────────────────────────────────
    render_mode = "rgb_array" if record_gif else None
    env = gym.make("LunarLander-v3", render_mode=render_mode)

    metrics = EvaluationMetrics()
    best_reward = -float("inf")
    best_frames: list[np.ndarray] = []

    for ep in tqdm(range(num_episodes), desc=f"Evaluating {agent_type.upper()}", unit="ep"):
        state, _ = env.reset(seed=seed + ep)
        recorder = Recorder() if record_gif else None
        total_reward = 0.0
        steps = 0
        terminated_flag = False
        final_state = state

        done = False
        while not done:
            if recorder is not None:
                frame = env.render()
                recorder.capture(frame)

            if agent_type == "ppo":
                action = agent.act(state, deterministic=True)
            else:
                action = agent.act(state, eval_mode=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            final_state = next_state
            terminated_flag = terminated
            state = next_state

        landed, crashed = _classify_outcome(total_reward, terminated_flag, final_state)

        metrics.add_episode(
            reward=total_reward,
            landed=landed,
            crashed=crashed,
            x_position=float(final_state[0]),
            y_position=float(final_state[1]),
            steps=steps,
        )

        if record_gif and total_reward > best_reward and recorder is not None:
            best_reward = total_reward
            best_frames = list(recorder.frames)

    env.close()

    # ── Save GIF of best episode ─────────────────────────────────────────
    if record_gif and best_frames:
        gif_path = os.path.join(RESULTS_DIR, f"{agent_type}_landing.gif")
        rec = Recorder()
        rec.frames = best_frames
        rec.save_gif(gif_path, fps=30)
        log.info(f"GIF saved -> {gif_path}")

    # ── Save metrics ─────────────────────────────────────────────────────
    metrics_path = os.path.join(RESULTS_DIR, f"{agent_type}_evaluation_results.json")
    metrics.save(metrics_path)
    log.info(f"\n{agent_type.upper()} Evaluation Results:\n{metrics}")

    if hasattr(agent, "close"):
        agent.close()

    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained RL agents")
    parser.add_argument(
        "--agent",
        type=str,
        choices=["dqn", "ddqn", "ppo", "all"],
        default="all",
        help="Which agent(s) to evaluate",
    )
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--episodes", type=int, default=100, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF recording")
    args = parser.parse_args()

    record_gif = not args.no_gif

    if args.agent == "all":
        # Default model paths for each agent
        defaults = {
            "dqn":  "checkpoints/dqn_best.pth",
            "ddqn": "checkpoints/ddqn_best.pth",
            "ppo":  "checkpoints/ppo_best",
        }

        all_rewards = {}
        for name, path in defaults.items():
            if not os.path.exists(path) and not os.path.exists(path + ".zip"):
                log.warning(f"Checkpoint not found for {name.upper()} at {path} — skipping.")
                continue
            m = evaluate_agent(name, path, args.episodes, record_gif, args.seed)
            all_rewards[name.upper()] = list(m.rewards)

        # Comparison reward plot (using eval rewards as bars is less meaningful,
        # so we load training reward curves if available)
        training_rewards = {}
        for algo in ("dqn", "ddqn", "ppo"):
            npy_path = os.path.join(RESULTS_DIR, f"{algo}_rewards.npy")
            if os.path.exists(npy_path):
                training_rewards[algo.upper()] = list(np.load(npy_path))
        if training_rewards:
            plot_comparison(training_rewards, save_dir=RESULTS_DIR)
            log.info("Comparison plot saved -> results/comparison.png")

    else:
        model_path = args.model
        if model_path is None:
            defaults = {
                "dqn": "checkpoints/dqn_best.pth",
                "ddqn": "checkpoints/ddqn_best.pth",
                "ppo": "checkpoints/ppo_best",
            }
            model_path = defaults[args.agent]

        evaluate_agent(args.agent, model_path, args.episodes, record_gif, args.seed)


if __name__ == "__main__":
    main()
