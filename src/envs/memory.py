"""MiniGrid Memory 环境的特征封装与脚本策略。"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

import gymnasium as gym
import minigrid  # noqa: F401  # Registers MiniGrid environments with Gymnasium.
import numpy as np
from gymnasium import spaces


class MemoryVectorObsWrapper(gym.ObservationWrapper):
    """将原始 MiniGrid 观测压缩为课程实验使用的 11 维特征向量。"""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # 固定从起点房间开始，保证每个 episode 的第 0 步都能看到 cue。
        self.unwrapped.agent_pos = np.asarray((2, self.unwrapped.height // 2))
        self.unwrapped.agent_dir = 0
        return self.observation(observation), info

    def observation(self, observation: dict[str, object]) -> np.ndarray:
        """构造紧凑的部分可观测状态表示。"""
        del observation
        env = self.unwrapped
        x, y = env.agent_pos
        key_visible, ball_visible = _visible_start_object_one_hot(env)
        upper_key, upper_ball, lower_key, lower_ball = _visible_terminal_object_one_hots(env)
        return np.asarray(
            [
                float(x) / max(env.width - 1, 1),
                float(y) / max(env.height - 1, 1),
                float(env.agent_dir) / 3.0,
                key_visible,
                ball_visible,
                float(x <= 4 and abs(y - env.height // 2) <= 1),
                float(env.step_count) / max(env.max_steps, 1),
                upper_key,
                upper_ball,
                lower_key,
                lower_ball,
            ],
            dtype=np.float32,
        )


def make_memory_env() -> gym.Env:
    """创建带特征包装器的 MiniGrid Memory 环境。"""
    return MemoryVectorObsWrapper(gym.make("MiniGrid-MemoryS17Random-v0"))


def _visible_start_object_one_hot(env) -> tuple[float, float]:
    """仅在起点房间内返回起始提示物体的 one-hot 编码。"""
    agent_x, agent_y = env.agent_pos
    obj = env.grid.get(1, env.height // 2 - 1)
    if obj is None:
        return 0.0, 0.0
    # 只有仍位于起点房间时，起始提示才属于“当前可见信息”。
    if agent_x > 4 or abs(agent_y - env.height // 2) > 1:
        return 0.0, 0.0
    if obj.type == "key":
        return 1.0, 0.0
    if obj.type == "ball":
        return 0.0, 1.0
    return 0.0, 0.0


def _visible_terminal_object_one_hots(env) -> tuple[float, float, float, float]:
    """仅在接近岔路口/终点区域时暴露上下两个终点物体的类型。"""
    agent_x, agent_y = env.agent_pos
    terminal_x = int(env.success_pos[0])
    if agent_x < terminal_x - 2 or abs(agent_y - env.height // 2) > 2:
        return 0.0, 0.0, 0.0, 0.0

    upper = env.grid.get(terminal_x, env.height // 2 - 2)
    lower = env.grid.get(terminal_x, env.height // 2 + 2)
    return (*_object_one_hot(upper), *_object_one_hot(lower))


def _object_one_hot(obj) -> tuple[float, float]:
    """将终点物体压缩成 key/ball 两维 one-hot。"""
    if obj is None:
        return 0.0, 0.0
    if obj.type == "key":
        return 1.0, 0.0
    if obj.type == "ball":
        return 0.0, 1.0
    return 0.0, 0.0


def memory_random_policy(
    state: np.ndarray,
    rng: np.random.Generator,
    env: gym.Env,
) -> int:
    """完全随机策略，用于制造失败和低回报轨迹。"""
    del state, env
    return int(rng.integers(0, 7))


def make_memory_explore_policy(env: gym.Env, rng: np.random.Generator) -> Callable:
    """返回一个探索脚本策略：到达终点区域，但不总是选对目标。"""
    target = _choose_terminal_target(env, rng)

    def policy(state: np.ndarray, step_rng: np.random.Generator, policy_env: gym.Env) -> int:
        del state, step_rng
        return _shortest_path_action(policy_env, target)

    return policy


def memory_expert_policy(
    state: np.ndarray,
    rng: np.random.Generator,
    env: gym.Env,
) -> int:
    """专家脚本策略：直接朝环境内部标记的正确终点移动。"""
    del state, rng
    return _shortest_path_action(env, tuple(env.unwrapped.success_pos))


def _choose_terminal_target(env: gym.Env, rng: np.random.Generator) -> tuple[int, int]:
    """为探索策略随机选择成功终点或失败终点。"""
    if rng.random() < 0.5:
        return tuple(env.unwrapped.success_pos)
    return tuple(env.unwrapped.failure_pos)


def _shortest_path_action(env: gym.Env, target: tuple[int, int]) -> int:
    """根据最短路径的下一格，转成 MiniGrid 的离散动作。"""
    unwrapped = env.unwrapped
    current = tuple(int(value) for value in unwrapped.agent_pos)
    path = _shortest_path_cells(env, current, target)
    if len(path) < 2:
        return int(unwrapped.actions.done)

    next_cell = path[1]
    dx = next_cell[0] - current[0]
    dy = next_cell[1] - current[1]
    desired_dir = {
        (1, 0): 0,
        (0, 1): 1,
        (-1, 0): 2,
        (0, -1): 3,
    }[(dx, dy)]
    turn_delta = (desired_dir - int(unwrapped.agent_dir)) % 4
    if turn_delta == 0:
        return int(unwrapped.actions.forward)
    if turn_delta == 1:
        return int(unwrapped.actions.right)
    return int(unwrapped.actions.left)


def _shortest_path_cells(
    env: gym.Env,
    start: tuple[int, int],
    target: tuple[int, int],
) -> list[tuple[int, int]]:
    """在网格上做 BFS，求起点到目标的最短可行路径。"""
    queue: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    deltas = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    while queue:
        cell = queue.popleft()
        if cell == target:
            break
        for dx, dy in deltas:
            neighbor = (cell[0] + dx, cell[1] + dy)
            if neighbor in parent or not _is_passable(env, neighbor):
                continue
            parent[neighbor] = cell
            queue.append(neighbor)

    if target not in parent:
        return [start]

    path = [target]
    while path[-1] != start:
        previous = parent[path[-1]]
        if previous is None:
            break
        path.append(previous)
    return list(reversed(path))


def _is_passable(env: gym.Env, cell: tuple[int, int]) -> bool:
    """判断网格单元是否可通行。"""
    unwrapped = env.unwrapped
    x, y = cell
    if x < 0 or y < 0 or x >= unwrapped.width or y >= unwrapped.height:
        return False
    obj = unwrapped.grid.get(x, y)
    return obj is None or bool(obj.can_overlap())
