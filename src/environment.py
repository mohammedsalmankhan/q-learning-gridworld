"""Grid world environment for tabular reinforcement learning.

A 5x5 deterministic grid world with:
    - Start state at (2, 1)
    - Terminal goal at (5, 5) with reward +10
    - Special teleport from (2, 4) to (4, 4) with reward +5
    - Two obstacles at (3, 3) and (3, 4) that block movement
    - Step cost of -1 for all other transitions

States are 1-indexed (row, column) tuples to match the standard
coursework specification. Actions are encoded as integers:
    1 = North, 2 = South, 3 = East, 4 = West
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple

State = Tuple[int, int]
Action = int

ROWS: int = 5
COLS: int = 5

START: State = (2, 1)
TERMINAL: State = (5, 5)
JUMP_FROM: State = (2, 4)
JUMP_TO: State = (4, 4)
OBSTACLES: Set[State] = {(3, 3), (3, 4)}

GOAL_REWARD: float = 10.0
JUMP_REWARD: float = 5.0
STEP_REWARD: float = -1.0

# Action -> (delta_row, delta_col)
ACTIONS: Dict[Action, Tuple[int, int]] = {
    1: (-1, 0),   # North
    2: (1, 0),    # South
    3: (0, 1),    # East
    4: (0, -1),   # West
}
ACTION_NAMES: Dict[Action, str] = {1: "N", 2: "S", 3: "E", 4: "W"}


@dataclass
class StepResult:
    """Outcome of a single environment step."""
    next_state: State
    reward: float
    done: bool


class GridWorld:
    """Deterministic 5x5 grid world MDP.

    Example:
        >>> env = GridWorld()
        >>> state = env.reset()
        >>> result = env.step(action=3)  # move East
        >>> print(result.next_state, result.reward, result.done)
    """

    def __init__(self) -> None:
        self.state: State = START

    def reset(self) -> State:
        """Reset the agent to the start state and return it."""
        self.state = START
        return self.state

    def step(self, action: Action) -> StepResult:
        """Take an action and return (next_state, reward, done).

        Movement rules:
            - From the jump cell, any action teleports to JUMP_TO with +5
            - Moves into obstacles or out of bounds leave state unchanged
              but still incur the -1 step cost
            - Reaching the terminal yields +10 and ends the episode
        """
        # Special teleport rule applies before normal movement
        if self.state == JUMP_FROM:
            self.state = JUMP_TO
            return StepResult(self.state, JUMP_REWARD, False)

        dr, dc = ACTIONS[action]
        candidate: State = (self.state[0] + dr, self.state[1] + dc)

        # Bump-back on illegal moves
        out_of_bounds = not (1 <= candidate[0] <= ROWS and 1 <= candidate[1] <= COLS)
        if out_of_bounds or candidate in OBSTACLES:
            candidate = self.state

        self.state = candidate
        if self.state == TERMINAL:
            return StepResult(self.state, GOAL_REWARD, True)
        return StepResult(self.state, STEP_REWARD, False)

    def all_states(self) -> list[State]:
        """Return every legal (non-obstacle) state in the grid."""
        return [
            (r, c)
            for r in range(1, ROWS + 1)
            for c in range(1, COLS + 1)
            if (r, c) not in OBSTACLES
        ]
