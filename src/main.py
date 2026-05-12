"""Training driver, evaluation, and figure generation.

Run with default configuration:
    python -m src.main

Run with custom learning rate and seed:
    python -m src.main --alpha 0.3 --seed 7
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from environment import (
    ACTIONS,
    COLS,
    GOAL_REWARD,
    JUMP_FROM,
    JUMP_REWARD,
    OBSTACLES,
    ROWS,
    START,
    STEP_REWARD,
    TERMINAL,
    GridWorld,
    State,
)
from agent import QLearningAgent, SARSAAgent


# --------------------------------------------------------------------------
# Training loops
# --------------------------------------------------------------------------

def train_q(
    env: GridWorld,
    agent: QLearningAgent,
    max_episodes: int = 100,
    max_steps: int = 200,
    window: int = 30,
    threshold: float = 10.0,
) -> Tuple[List[float], List[float], List[float], Optional[int]]:
    """Train a Q-learning agent and return reward history.

    Returns:
        per_episode_rewards, rolling_mean_rewards, epsilon_history, episode_converged
    """
    rewards: List[float] = []
    rolling: List[float] = []
    eps_hist: List[float] = []
    buf: deque = deque(maxlen=window)
    converged: Optional[int] = None

    for ep in range(1, max_episodes + 1):
        s = env.reset()
        total = 0.0
        for _ in range(max_steps):
            a = agent.choose_action(s)
            result = env.step(a)
            agent.update(s, a, result.reward, result.next_state, result.done)
            total += result.reward
            s = result.next_state
            if result.done:
                break

        rewards.append(total)
        buf.append(total)
        rolling.append(float(np.mean(buf)))
        eps_hist.append(agent.epsilon)
        agent.decay_epsilon()

        if (
            len(buf) == window
            and np.mean(buf) > threshold
            and converged is None
        ):
            converged = ep
            break

    return rewards, rolling, eps_hist, converged


def train_sarsa(
    env: GridWorld,
    agent: SARSAAgent,
    max_episodes: int = 100,
    max_steps: int = 200,
    window: int = 30,
    threshold: float = 10.0,
) -> Tuple[List[float], List[float], Optional[int]]:
    """Train a SARSA agent. Returns rewards, rolling mean, and convergence episode."""
    rewards: List[float] = []
    rolling: List[float] = []
    buf: deque = deque(maxlen=window)
    converged: Optional[int] = None

    for ep in range(1, max_episodes + 1):
        s = env.reset()
        a = agent.choose_action(s)
        total = 0.0
        for _ in range(max_steps):
            result = env.step(a)
            a_next = agent.choose_action(result.next_state)
            agent.update(s, a, result.reward, result.next_state, a_next, result.done)
            total += result.reward
            s, a = result.next_state, a_next
            if result.done:
                break

        rewards.append(total)
        buf.append(total)
        rolling.append(float(np.mean(buf)))
        agent.decay_epsilon()

        if (
            len(buf) == window
            and np.mean(buf) > threshold
            and converged is None
        ):
            converged = ep
            break

    return rewards, rolling, converged


# --------------------------------------------------------------------------
# Analytical baseline: Bellman value iteration
# --------------------------------------------------------------------------

def value_iteration(gamma: float = 0.95, theta: float = 1e-6) -> np.ndarray:
    """Compute V*(s) by Bellman optimality iteration."""
    V = np.zeros((ROWS, COLS))
    while True:
        delta = 0.0
        for r in range(1, ROWS + 1):
            for c in range(1, COLS + 1):
                if (r, c) in OBSTACLES or (r, c) == TERMINAL:
                    continue
                old = V[r - 1, c - 1]
                q_candidates = []
                for a in range(1, 5):
                    if (r, c) == JUMP_FROM:
                        nr, nc = (4, 4)  # JUMP_TO
                        rew = JUMP_REWARD
                    else:
                        dr, dc = ACTIONS[a]
                        nr, nc = r + dr, c + dc
                        if not (1 <= nr <= ROWS and 1 <= nc <= COLS) or (nr, nc) in OBSTACLES:
                            nr, nc = r, c
                        rew = GOAL_REWARD if (nr, nc) == TERMINAL else STEP_REWARD

                    if (nr, nc) == TERMINAL:
                        q_candidates.append(rew)
                    else:
                        q_candidates.append(rew + gamma * V[nr - 1, nc - 1])

                V[r - 1, c - 1] = max(q_candidates)
                delta = max(delta, abs(old - V[r - 1, c - 1]))
        if delta < theta:
            break
    return V


# --------------------------------------------------------------------------
# Greedy policy evaluation
# --------------------------------------------------------------------------

def evaluate_greedy(
    env: GridWorld,
    agent: QLearningAgent,
    max_steps: int = 50,
) -> Tuple[List[State], float]:
    """Roll out a greedy episode from start. Returns (path, total_reward)."""
    s = env.reset()
    path: List[State] = [s]
    total = 0.0
    for _ in range(max_steps):
        a = agent.choose_action(s, greedy=True)
        result = env.step(a)
        path.append(result.next_state)
        total += result.reward
        s = result.next_state
        if result.done:
            break
    return path, total


# --------------------------------------------------------------------------
# Learning rate sweep
# --------------------------------------------------------------------------

def alpha_sweep(
    alphas: List[float],
    runs: int = 3,
) -> Dict[float, List[List[float]]]:
    """Run training across multiple alphas with multiple seeds per alpha."""
    results: Dict[float, List[List[float]]] = {}
    for alpha in alphas:
        all_runs = []
        for run in range(runs):
            env = GridWorld()
            agent = QLearningAgent(alpha=alpha, seed=42 + run)
            rewards, _, _, _ = train_q(env, agent)
            all_runs.append(rewards)
        results[alpha] = all_runs
    return results


# --------------------------------------------------------------------------
# Plotting helpers (preserved from original — kept terse for clarity)
# --------------------------------------------------------------------------

def _plot_grid(V: np.ndarray, title: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_v = V.copy()
    for r, c in OBSTACLES:
        plot_v[r - 1, c - 1] = np.nan
    im = ax.imshow(plot_v, cmap="viridis", extent=(0.5, COLS + 0.5, ROWS + 0.5, 0.5))

    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            if (r, c) in OBSTACLES:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
                ax.text(c, r, "X", ha="center", va="center", color="white",
                        fontsize=14, fontweight="bold")
            else:
                colour = "white" if V[r - 1, c - 1] < V.max() / 2 else "black"
                ax.text(c, r - 0.15, f"{V[r - 1, c - 1]:.2f}",
                        ha="center", va="center", fontsize=10, color=colour)
            if (r, c) == START:
                ax.text(c, r + 0.3, "START", ha="center", va="center",
                        color="red", fontsize=8, fontweight="bold")
            if (r, c) == TERMINAL:
                ax.text(c, r + 0.3, "GOAL", ha="center", va="center",
                        color="cyan", fontsize=8, fontweight="bold")
            if (r, c) == JUMP_FROM:
                ax.text(c, r + 0.3, "JUMP", ha="center", va="center",
                        color="magenta", fontsize=8, fontweight="bold")

    ax.set_xticks(range(1, COLS + 1))
    ax.set_yticks(range(1, ROWS + 1))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title)
    ax.set_xlim(0.5, COLS + 0.5)
    ax.set_ylim(ROWS + 0.5, 0.5)
    plt.colorbar(im, ax=ax, label="V(s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {save_path}")


def _plot_policy(policy: np.ndarray, V: np.ndarray, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_v = V.copy()
    for r, c in OBSTACLES:
        plot_v[r - 1, c - 1] = np.nan
    ax.imshow(plot_v, cmap="Blues", extent=(0.5, COLS + 0.5, ROWS + 0.5, 0.5))

    arrows = {1: (0, -0.3), 2: (0, 0.3), 3: (0.3, 0), 4: (-0.3, 0)}
    for r in range(1, ROWS + 1):
        for c in range(1, COLS + 1):
            if (r, c) in OBSTACLES:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))
                continue
            if (r, c) == TERMINAL:
                ax.text(c, r, "GOAL", ha="center", va="center",
                        color="red", fontweight="bold")
                continue
            a = policy[r - 1, c - 1]
            dx, dy = arrows[a]
            ax.arrow(c - dx / 2, r - dy / 2, dx, dy,
                     head_width=0.12, color="black", length_includes_head=True)

    ax.set_xticks(range(1, COLS + 1))
    ax.set_yticks(range(1, ROWS + 1))
    ax.set_title("Greedy Policy")
    ax.set_xlim(0.5, COLS + 0.5)
    ax.set_ylim(ROWS + 0.5, 0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {save_path}")


def _plot_curve(
    rewards: List[float],
    rolling: List[float],
    converged: Optional[int],
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    eps = range(1, len(rewards) + 1)
    ax.plot(eps, rewards, alpha=0.4, color="steelblue", label="Episode reward")
    ax.plot(eps, rolling, color="red", linewidth=2, label="30-ep rolling mean")
    ax.axhline(10, color="green", linestyle="--", label="Threshold = 10")
    if converged:
        ax.axvline(converged, color="green", linestyle=":",
                   label=f"Converged @ ep {converged}")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title("Q-Learning training progress")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {save_path}")


def _plot_alpha(results: Dict[float, List[List[float]]], save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(results)))
    for col, (alpha, runs) in zip(colors, sorted(results.items())):
        max_len = max(len(r) for r in runs)
        padded = np.full((len(runs), max_len), np.nan)
        for i, r in enumerate(runs):
            padded[i, : len(r)] = r
        mean = np.nanmean(padded, axis=0)
        ax.plot(range(1, max_len + 1), mean, color=col,
                linewidth=2, label=f"α = {alpha}")
    ax.axhline(10, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean cumulative reward")
    ax.set_title("Effect of learning rate α on convergence")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {save_path}")


def _plot_q_vs_sarsa(
    q_rolling: List[float],
    s_rolling: List[float],
    save_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(q_rolling, color="red", linewidth=2, label="Q-learning")
    ax.plot(s_rolling, color="blue", linewidth=2, label="SARSA")
    ax.axhline(10, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("30-ep rolling mean reward")
    ax.set_title("Q-learning vs SARSA")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {save_path}")


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train and evaluate a tabular Q-learning agent on a 5x5 grid world.",
    )
    p.add_argument("--alpha", type=float, default=0.5,
                   help="Learning rate for the headline run (default: 0.5)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for reproducibility (default: 42)")
    p.add_argument("--episodes", type=int, default=100,
                   help="Max training episodes (default: 100)")
    p.add_argument("--plot-dir", type=str, default="plots",
                   help="Directory to save figures (default: ./plots)")
    p.add_argument("--skip-sweep", action="store_true",
                   help="Skip the learning-rate sweep to run faster")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.plot_dir, exist_ok=True)

    print("Q-LEARNING GRID WORLD")
    print("-" * 45)

    print(f"\n[1] Training Q-learning (alpha={args.alpha}, seed={args.seed})")
    env = GridWorld()
    agent = QLearningAgent(alpha=args.alpha, seed=args.seed)
    rewards, rolling, _, conv = train_q(env, agent, max_episodes=args.episodes)
    print(f"    episodes: {len(rewards)} | converged at: {conv} | "
          f"final reward: {rewards[-1]:.2f}")
    _plot_curve(rewards, rolling, conv, f"{args.plot_dir}/learning_curve.png")

    print("\n[2] State value visualisation")
    V_learned = agent.state_values()
    _plot_grid(V_learned, "Learned V(s) after Q-learning",
               f"{args.plot_dir}/state_values_learned.png")
    _plot_policy(agent.greedy_policy(), V_learned, f"{args.plot_dir}/policy.png")

    print("\n[3] Optimal V* via value iteration")
    V_star = value_iteration()
    _plot_grid(V_star, "Optimal V* (Value Iteration)",
               f"{args.plot_dir}/state_values_optimal.png")

    print("\n[4] Greedy rollout from start")
    path, total = evaluate_greedy(GridWorld(), agent)
    print(f"    path length: {len(path) - 1} steps | reward: {total:.2f}")
    print(f"    path: {path}")

    print("\n[5] SARSA agent for comparison")
    s_env = GridWorld()
    s_agent = SARSAAgent(alpha=args.alpha, seed=args.seed)
    s_rewards, s_rolling, s_conv = train_sarsa(s_env, s_agent,
                                                max_episodes=args.episodes)
    print(f"    episodes: {len(s_rewards)} | converged at: {s_conv} | "
          f"final reward: {s_rewards[-1]:.2f}")
    _plot_q_vs_sarsa(rolling, s_rolling, f"{args.plot_dir}/q_vs_sarsa.png")

    if not args.skip_sweep:
        print("\n[6] Learning rate sweep")
        sweep = alpha_sweep([0.1, 0.3, 0.5, 0.7, 1.0], runs=3)
        print("    alpha | mean episodes | mean final reward")
        for a in sorted(sweep):
            runs = sweep[a]
            n = np.mean([len(x) for x in runs])
            f = np.mean([x[-1] for x in runs])
            print(f"    {a:.1f}   | {n:13.1f} | {f:17.2f}")
        _plot_alpha(sweep, f"{args.plot_dir}/alpha_comparison.png")

    print(f"\nDone. Figures saved to {args.plot_dir}/")


if __name__ == "__main__":
    main()
