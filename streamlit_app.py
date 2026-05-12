"""Interactive Streamlit demo of the Q-learning grid world.

Lets a recruiter or interviewer:
    - Tweak hyperparameters (alpha, epsilon decay, episodes)
    - Watch the learning curve appear
    - See the learned state-value heatmap and greedy policy
    - Compare against the analytical Bellman optimum
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from environment import (
    COLS,
    JUMP_FROM,
    OBSTACLES,
    ROWS,
    START,
    TERMINAL,
    GridWorld,
)
from agent import QLearningAgent, SARSAAgent
from main import evaluate_greedy, train_q, train_sarsa, value_iteration


# ---------------- page config ----------------

st.set_page_config(
    page_title="Q-Learning Grid World",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 Q-Learning Grid World")
st.markdown(
    "Interactive tabular reinforcement learning demo. A Q-learning agent learns "
    "to navigate a 5×5 grid with obstacles, a goal, and a teleport shortcut."
)

# ---------------- sidebar controls ----------------

st.sidebar.header("Hyperparameters")
alpha = st.sidebar.slider("Learning rate (α)", 0.1, 1.0, 0.5, 0.1)
gamma = st.sidebar.slider("Discount factor (γ)", 0.5, 0.99, 0.95, 0.01)
epsilon_decay = st.sidebar.slider("Epsilon decay", 0.80, 0.99, 0.95, 0.01)
max_episodes = st.sidebar.slider("Max episodes", 20, 200, 100, 10)
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999, value=42)
algorithm = st.sidebar.radio("Algorithm", ["Q-Learning", "SARSA", "Compare both"])

run_button = st.sidebar.button("🚀 Train agent", type="primary")

# ---------------- environment description ----------------

with st.expander("📖 About the environment"):
    st.markdown(
        """
        - **Grid:** 5×5 with 1-indexed (row, column) states
        - **Start:** (2, 1)
        - **Goal:** (5, 5) — reward **+10**
        - **Jump:** (2, 4) → (4, 4) — reward **+5**
        - **Obstacles:** (3, 3) and (3, 4) — block movement
        - **Step cost:** **−1** for all other transitions
        - **Termination:** reaching the goal, or 200 steps per episode

        The optimal policy uses the jump shortcut and reaches the goal in 6 steps
        with a total reward of +11.
        """
    )

# ---------------- training ----------------


def render_value_grid(V: np.ndarray, title: str):
    """Render a value-function heatmap with annotations."""
    fig, ax = plt.subplots(figsize=(6, 6))
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
                ax.text(c, r - 0.1, f"{V[r - 1, c - 1]:.2f}",
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
    ax.set_title(title)
    ax.set_xlim(0.5, COLS + 0.5)
    ax.set_ylim(ROWS + 0.5, 0.5)
    plt.colorbar(im, ax=ax, label="V(s)")
    plt.tight_layout()
    return fig


def render_curve(rewards, rolling, label: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(rewards, alpha=0.4, color="steelblue", label="Episode reward")
    ax.plot(rolling, color="red", linewidth=2, label="30-ep rolling mean")
    ax.axhline(10, color="green", linestyle="--", alpha=0.7, label="Threshold = 10")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative reward")
    ax.set_title(f"{label} — training progress")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


if run_button:
    with st.spinner("Training the agent..."):
        env = GridWorld()
        agent = QLearningAgent(
            alpha=alpha, gamma=gamma, decay=epsilon_decay, seed=seed,
        )
        q_rewards, q_rolling, _, q_conv = train_q(
            env, agent, max_episodes=max_episodes,
        )

        if algorithm in ("SARSA", "Compare both"):
            s_env = GridWorld()
            s_agent = SARSAAgent(
                alpha=alpha, gamma=gamma, decay=epsilon_decay, seed=seed,
            )
            s_rewards, s_rolling, s_conv = train_sarsa(
                s_env, s_agent, max_episodes=max_episodes,
            )

    # Top-line results
    col1, col2, col3 = st.columns(3)
    col1.metric("Episodes to converge", q_conv if q_conv else "Did not converge")
    col2.metric("Final reward", f"{q_rewards[-1]:.1f}")
    path, _ = evaluate_greedy(GridWorld(), agent)
    col3.metric("Path length (greedy)", f"{len(path) - 1} steps")

    # Learning curve
    if algorithm == "Compare both":
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(q_rolling, color="red", linewidth=2, label="Q-learning")
        ax.plot(s_rolling, color="blue", linewidth=2, label="SARSA")
        ax.axhline(10, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("30-ep rolling mean reward")
        ax.set_title("Q-learning vs SARSA")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    else:
        st.pyplot(render_curve(q_rewards, q_rolling, "Q-Learning"))

    # Value heatmaps side by side
    st.subheader("State values: learned vs optimal")
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(render_value_grid(agent.state_values(),
                                    "Learned V(s) after training"))
    with c2:
        V_star = value_iteration(gamma=gamma)
        st.pyplot(render_value_grid(V_star, "Optimal V* (Bellman)"))

    # Greedy path
    st.subheader("Greedy rollout from start")
    st.code(" → ".join(str(s) for s in path))

else:
    st.info("👈 Adjust the hyperparameters in the sidebar and click **Train agent**.")
    st.markdown("---")
    st.markdown(
        "### What you'll see\n"
        "1. **Learning curve** with the 30-episode rolling mean\n"
        "2. **Learned state-value heatmap** compared against the analytical "
        "Bellman optimum (computed by value iteration)\n"
        "3. **Greedy path** the agent takes after training"
    )

st.markdown("---")
st.caption(
    "Built by Salman Khan · MSc Artificial Intelligence, Ulster University · "
    "[GitHub repo](https://github.com/mohammedsalmankhan/q-learning-gridworld)"
)
