from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class GridWorldConfig:
    size: int = 7
    max_steps: int = 20
    near_goal: tuple[int, int] = (0, 2)
    far_goal: tuple[int, int] = (6, 0)
    near_reward: float = 1.0
    far_reward: float = 3.0


class TwoGoalGridWorld(gym.Env):
    """Small sparse-reward environment with two valid return modes."""

    metadata = {"render_modes": []}

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        super().__init__()
        self.config = config or GridWorldConfig()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.rng = np.random.default_rng(0)
        self.position = np.array([0, 0], dtype=np.int64)
        self.steps = 0
        self.last_terminal_type = "none"

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = np.array([0, 0], dtype=np.int64)
        self.steps = 0
        self.last_terminal_type = "none"
        return self._observation(), {"terminal_type": self.last_terminal_type}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        if action == 0:
            self.position[1] = max(0, self.position[1] - 1)
        elif action == 1:
            self.position[1] = min(self.config.size - 1, self.position[1] + 1)
        elif action == 2:
            self.position[0] = max(0, self.position[0] - 1)
        elif action == 3:
            self.position[0] = min(self.config.size - 1, self.position[0] + 1)
        else:
            raise ValueError(f"Invalid action {action}")

        self.steps += 1
        reward = 0.0
        terminated = False
        truncated = self.steps >= self.config.max_steps
        pos = tuple(int(value) for value in self.position)

        if pos == self.config.near_goal:
            reward = self.config.near_reward
            terminated = True
            truncated = False
            self.last_terminal_type = "near_goal"
        elif pos == self.config.far_goal:
            reward = self.config.far_reward
            terminated = True
            truncated = False
            self.last_terminal_type = "far_goal"
        elif truncated:
            self.last_terminal_type = "timeout"

        return self._observation(), reward, terminated, truncated, {
            "terminal_type": self.last_terminal_type
        }

    def _observation(self) -> np.ndarray:
        denom = float(self.config.size - 1)
        return np.array(
            [
                self.position[0] / denom,
                self.position[1] / denom,
                self.config.near_goal[0] / denom,
                self.config.near_goal[1] / denom,
                self.config.far_goal[0] / denom,
                self.config.far_goal[1] / denom,
            ],
            dtype=np.float32,
        )


@dataclass(frozen=True)
class KeyDoorConfig:
    size: int = 9
    max_steps: int = 30
    near_goal: tuple[int, int] = (2, 0)
    key_pos: tuple[int, int] = (0, 6)
    door_pos: tuple[int, int] = (4, 6)
    far_goal: tuple[int, int] = (6, 8)
    near_reward: float = 1.0
    far_reward: float = 5.0


class KeyDoorGridWorld(gym.Env):
    """Sparse key-door task with a low-return shortcut and high-return delayed goal."""

    metadata = {"render_modes": []}

    def __init__(self, config: KeyDoorConfig | None = None) -> None:
        super().__init__()
        self.config = config or KeyDoorConfig()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self.rng = np.random.default_rng(0)
        self.position = np.array([0, 0], dtype=np.int64)
        self.has_key = False
        self.steps = 0
        self.last_terminal_type = "none"

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        del options
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.position = np.array([0, 0], dtype=np.int64)
        self.has_key = False
        self.steps = 0
        self.last_terminal_type = "none"
        return self._observation(), {"terminal_type": self.last_terminal_type}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = int(action)
        candidate = self.position.copy()
        if action == 0:
            candidate[1] = max(0, candidate[1] - 1)
        elif action == 1:
            candidate[1] = min(self.config.size - 1, candidate[1] + 1)
        elif action == 2:
            candidate[0] = max(0, candidate[0] - 1)
        elif action == 3:
            candidate[0] = min(self.config.size - 1, candidate[0] + 1)
        else:
            raise ValueError(f"Invalid action {action}")

        if self._can_enter(candidate):
            self.position = candidate

        self.steps += 1
        pos = tuple(int(value) for value in self.position)
        if pos == self.config.key_pos:
            self.has_key = True

        reward = 0.0
        terminated = False
        truncated = self.steps >= self.config.max_steps

        if pos == self.config.near_goal:
            reward = self.config.near_reward
            terminated = True
            truncated = False
            self.last_terminal_type = "near_goal"
        elif pos == self.config.far_goal and self.has_key:
            reward = self.config.far_reward
            terminated = True
            truncated = False
            self.last_terminal_type = "far_goal"
        elif truncated:
            self.last_terminal_type = "timeout"

        return self._observation(), reward, terminated, truncated, {
            "terminal_type": self.last_terminal_type
        }

    def _can_enter(self, candidate: np.ndarray) -> bool:
        x, y = (int(candidate[0]), int(candidate[1]))
        # A vertical wall forces the agent to use the door after collecting the key.
        if x == 4 and y < 8 and (x, y) != self.config.door_pos:
            return False
        if (x, y) == self.config.door_pos and not self.has_key:
            return False
        return True

    def _observation(self) -> np.ndarray:
        denom = float(self.config.size - 1)
        return np.array(
            [
                self.position[0] / denom,
                self.position[1] / denom,
                float(self.has_key),
                self.config.key_pos[0] / denom,
                self.config.key_pos[1] / denom,
                self.config.door_pos[0] / denom,
                self.config.door_pos[1] / denom,
                self.config.near_goal[0] / denom,
                self.config.near_goal[1] / denom,
                self.config.far_goal[0] / denom,
                self.config.far_goal[1] / denom,
            ],
            dtype=np.float32,
        )
