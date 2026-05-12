"""Tabular reinforcement learning agents.

Implements two classic temporal-difference control algorithms:
    - QLearningAgent: off-policy, bootstraps with max_a Q(s', a)
    - SARSAAgent: on-policy, bootstraps with Q(s', a') of action actually taken

Both agents share an epsilon-greedy action selection strategy with
multiplicative epsilon decay.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from environment import ROWS, COLS, State, Action


class QLearningAgent:
    """Off-policy tabular Q-learning agent.

    Update rule:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]

    Args:
        alpha: Learning rate in (0, 1].
        gamma: Discount factor in [0, 1).
        epsilon: Initial exploration probability.
        eps_min: Lower bound for epsilon after decay.
        decay: Multiplicative epsilon decay applied per episode.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        eps_min: float = 0.05,
        decay: float = 0.95,
        seed: Optional[int] = None,
    ) -> None:
        self.Q: np.ndarray = np.zeros((ROWS, COLS, 4))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.decay = decay
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state: State, greedy: bool = False) -> Action:
        """Select an action using epsilon-greedy policy.

        Ties among equally valued greedy actions are broken uniformly at
        random to avoid systematic bias toward lower-numbered actions.
        """
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(1, 5))

        r, c = state[0] - 1, state[1] - 1
        q_values = self.Q[r, c]
        best_actions = np.flatnonzero(q_values == q_values.max())
        return int(self.rng.choice(best_actions)) + 1

    def update(
        self,
        s: State,
        a: Action,
        r: float,
        s_next: State,
        done: bool,
    ) -> None:
        """Apply the Q-learning TD update."""
        sr, sc = s[0] - 1, s[1] - 1
        spr, spc = s_next[0] - 1, s_next[1] - 1
        target = r if done else r + self.gamma * self.Q[spr, spc].max()
        td_error = target - self.Q[sr, sc, a - 1]
        self.Q[sr, sc, a - 1] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Apply one step of multiplicative epsilon decay (floored at eps_min)."""
        self.epsilon = max(self.eps_min, self.epsilon * self.decay)

    def state_values(self) -> np.ndarray:
        """Return V(s) = max_a Q(s, a) as a (ROWS, COLS) array."""
        return self.Q.max(axis=2)

    def greedy_policy(self) -> np.ndarray:
        """Return the greedy policy as a (ROWS, COLS) array of actions (1-4)."""
        return self.Q.argmax(axis=2) + 1


class SARSAAgent(QLearningAgent):
    """On-policy SARSA agent.

    Differs from Q-learning only in the update target: SARSA uses the
    Q-value of the action actually chosen in the next state, while
    Q-learning uses the maximum over all next-state actions.
    """

    def update(  # type: ignore[override]
        self,
        s: State,
        a: Action,
        r: float,
        s_next: State,
        a_next: Action,
        done: bool,
    ) -> None:
        """Apply the SARSA TD update using the actually-selected next action."""
        sr, sc = s[0] - 1, s[1] - 1
        spr, spc = s_next[0] - 1, s_next[1] - 1
        target = r if done else r + self.gamma * self.Q[spr, spc, a_next - 1]
        td_error = target - self.Q[sr, sc, a - 1]
        self.Q[sr, sc, a - 1] += self.alpha * td_error
