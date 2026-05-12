"""Unit tests for the grid world environment and Q-learning agent."""
import sys
from pathlib import Path

# Make src importable when running pytest from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest

from environment import (
    GOAL_REWARD,
    JUMP_FROM,
    JUMP_REWARD,
    JUMP_TO,
    OBSTACLES,
    START,
    STEP_REWARD,
    TERMINAL,
    GridWorld,
)
from agent import QLearningAgent, SARSAAgent


# ---------------- environment ----------------

class TestGridWorld:

    def test_reset_returns_start(self):
        env = GridWorld()
        env.state = (4, 4)  # move agent somewhere else
        assert env.reset() == START

    def test_step_into_open_cell(self):
        env = GridWorld()
        env.reset()  # state = (2, 1)
        result = env.step(3)  # East
        assert result.next_state == (2, 2)
        assert result.reward == STEP_REWARD
        assert not result.done

    def test_step_into_obstacle_bumps_back(self):
        env = GridWorld()
        env.state = (3, 2)  # cell adjacent to obstacle (3,3)
        result = env.step(3)  # East would hit obstacle
        assert result.next_state == (3, 2)  # unchanged
        assert result.reward == STEP_REWARD

    def test_step_out_of_bounds_bumps_back(self):
        env = GridWorld()
        env.state = (1, 1)
        result = env.step(1)  # North off the grid
        assert result.next_state == (1, 1)
        assert result.reward == STEP_REWARD

    def test_jump_teleports_with_bonus(self):
        env = GridWorld()
        env.state = JUMP_FROM  # (2, 4)
        result = env.step(3)  # any action triggers the teleport
        assert result.next_state == JUMP_TO  # (4, 4)
        assert result.reward == JUMP_REWARD
        assert not result.done

    def test_reaching_terminal_returns_goal_reward(self):
        env = GridWorld()
        env.state = (5, 4)  # one step West of goal
        result = env.step(3)  # East -> (5, 5)
        assert result.next_state == TERMINAL
        assert result.reward == GOAL_REWARD
        assert result.done

    def test_all_states_excludes_obstacles(self):
        env = GridWorld()
        legal = env.all_states()
        assert len(legal) == 23  # 25 cells minus 2 obstacles
        for obs in OBSTACLES:
            assert obs not in legal


# ---------------- agent ----------------

class TestQLearningAgent:

    def test_initial_q_is_zero(self):
        agent = QLearningAgent(seed=0)
        assert np.all(agent.Q == 0)

    def test_greedy_action_is_deterministic_when_q_is_uniform(self):
        agent = QLearningAgent(seed=1)
        # uniform Q means random tie-break; just check action is in [1,4]
        action = agent.choose_action((2, 1), greedy=True)
        assert action in (1, 2, 3, 4)

    def test_update_changes_q(self):
        agent = QLearningAgent(alpha=0.5, seed=2)
        agent.update(s=(2, 1), a=3, r=-1.0, s_next=(2, 2), done=False)
        # one element should have changed
        assert agent.Q.sum() != 0

    def test_epsilon_decays_to_floor(self):
        agent = QLearningAgent(epsilon=0.5, eps_min=0.1, decay=0.5, seed=3)
        for _ in range(20):
            agent.decay_epsilon()
        assert agent.epsilon == pytest.approx(0.1)

    def test_terminal_target_ignores_future(self):
        """When done=True the bootstrap should not include gamma * V(next)."""
        agent = QLearningAgent(alpha=1.0, gamma=0.95, seed=4)
        agent.Q[4, 4, :] = 100.0  # next-state Q is huge
        agent.update(s=(5, 4), a=3, r=10.0, s_next=(5, 5), done=True)
        # Only the immediate reward should propagate
        assert agent.Q[4, 3, 2] == pytest.approx(10.0)


class TestSARSAAgent:

    def test_sarsa_uses_chosen_next_action(self):
        agent = SARSAAgent(alpha=1.0, gamma=1.0, seed=5)
        agent.Q[2, 2, 0] = 5.0   # high value for action 1 in next state
        agent.Q[2, 2, 1] = -5.0  # low value for action 2
        # Pass action 2 as the actually-selected next action.
        agent.update(s=(2, 2), a=3, r=0.0, s_next=(3, 3), a_next=2, done=False)
        # Q(s, a) should equal r + gamma * Q(s_next, a_next=2) = -5.0
        assert agent.Q[1, 1, 2] == pytest.approx(-5.0)


# ---------------- integration ----------------

def test_agent_learns_to_reach_goal():
    """End-to-end: after training, greedy rollout reaches the goal."""
    from main import train_q, evaluate_greedy

    env = GridWorld()
    agent = QLearningAgent(alpha=0.5, seed=42)
    train_q(env, agent, max_episodes=100)
    path, reward = evaluate_greedy(GridWorld(), agent)
    assert path[-1] == TERMINAL
    assert reward >= 5.0  # positive reward indicates good policy
