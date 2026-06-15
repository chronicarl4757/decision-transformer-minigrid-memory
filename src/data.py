"""课程版 Decision Transformer 实验中的离线数据构造工具。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
from torch.utils.data import Dataset

from src.envs.gridworld import KeyDoorGridWorld, TwoGoalGridWorld
from src.envs.memory import (
    make_memory_env,
    make_memory_explore_policy,
    memory_expert_policy,
    memory_random_policy,
)


@dataclass(frozen=True)
class OfflineTrajectory:
    """一条完整的离线轨迹，供 BC、RCBC 和 DT 训练共用。"""

    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns_to_go: np.ndarray
    timesteps: np.ndarray
    total_return: float


@dataclass(frozen=True)
class DTSequenceSample:
    """DT 训练使用的固定长度上下文窗口，采用左侧补零。"""

    returns_to_go: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    timesteps: np.ndarray
    mask: np.ndarray


def compute_returns_to_go(rewards: Sequence[float]) -> np.ndarray:
    """计算轨迹中每个时间步的未折扣 return-to-go。"""
    rewards_array = np.asarray(rewards, dtype=np.float32)
    return np.cumsum(rewards_array[::-1], dtype=np.float32)[::-1].copy()


def random_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    del state
    return int(rng.integers(0, 2))


def weak_heuristic_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    score = state[2] + 0.25 * state[3]
    if rng.random() < 0.25:
        return int(rng.integers(0, 2))
    return int(score > 0.0)


def strong_heuristic_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    score = state[2] + 0.5 * state[3] + 0.02 * state[0] + 0.05 * state[1]
    if rng.random() < 0.05:
        return int(rng.integers(0, 2))
    return int(score > 0.0)


def collect_cartpole_trajectories(
    num_episodes: int,
    seed: int,
    max_steps: int = 500,
    policy_mix: tuple[float, float, float] = (0.35, 0.35, 0.30),
) -> list[OfflineTrajectory]:
    """采集混合质量的 CartPole 轨迹，用于早期基线实验。"""
    rng = np.random.default_rng(seed)
    env = gym.make("CartPole-v1")
    policies: list[Callable[[np.ndarray, np.random.Generator], int]] = [
        random_policy,
        weak_heuristic_policy,
        strong_heuristic_policy,
    ]
    trajectories: list[OfflineTrajectory] = []

    for episode_idx in range(num_episodes):
        policy_idx = int(rng.choice(len(policies), p=np.asarray(policy_mix)))
        policy = policies[policy_idx]
        state, _ = env.reset(seed=seed + episode_idx)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        for _ in range(max_steps):
            action = policy(np.asarray(state, dtype=np.float32), rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            state = next_state
            if terminated or truncated:
                break

        rewards_array = np.asarray(rewards, dtype=np.float32)
        trajectories.append(
            OfflineTrajectory(
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=rewards_array,
                returns_to_go=compute_returns_to_go(rewards_array),
                timesteps=np.arange(len(actions), dtype=np.int64),
                total_return=float(rewards_array.sum()),
            )
        )

    env.close()
    return trajectories


def grid_random_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    del state
    return int(rng.integers(0, 4))


def grid_near_goal_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    del rng
    x, y, near_x, near_y = state[:4]
    if x < near_x:
        return 3
    if y < near_y:
        return 1
    if x > near_x:
        return 2
    return 0


def grid_far_goal_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    del rng
    x, y, _, _, far_x, far_y = state
    if x < far_x:
        return 3
    if y < far_y:
        return 1
    if x > far_x:
        return 2
    return 0


def collect_gridworld_trajectories(
    num_episodes: int,
    seed: int,
    policy_mix: tuple[float, float, float] = (0.20, 0.65, 0.15),
) -> list[OfflineTrajectory]:
    """采集 two-goal 玩具环境轨迹，构造不同回报模式的数据。"""
    rng = np.random.default_rng(seed)
    env = TwoGoalGridWorld()
    policies: list[Callable[[np.ndarray, np.random.Generator], int]] = [
        grid_random_policy,
        grid_near_goal_policy,
        grid_far_goal_policy,
    ]
    trajectories: list[OfflineTrajectory] = []

    for episode_idx in range(num_episodes):
        policy_idx = int(rng.choice(len(policies), p=np.asarray(policy_mix)))
        policy = policies[policy_idx]
        state, _ = env.reset(seed=seed + episode_idx)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(np.asarray(state, dtype=np.float32), rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            state = next_state

        rewards_array = np.asarray(rewards, dtype=np.float32)
        trajectories.append(
            OfflineTrajectory(
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=rewards_array,
                returns_to_go=compute_returns_to_go(rewards_array),
                timesteps=np.arange(len(actions), dtype=np.int64),
                total_return=float(rewards_array.sum()),
            )
        )

    return trajectories


def keydoor_near_goal_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    del rng
    x, y = state[:2]
    near_x, near_y = state[7:9]
    return _move_toward(x, y, near_x, near_y)


def keydoor_far_goal_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    del rng
    x, y, has_key = state[:3]
    key_x, key_y = state[3:5]
    door_x, door_y = state[5:7]
    goal_x, goal_y = state[9:11]
    if has_key < 0.5:
        return _move_toward(x, y, key_x, key_y)
    if x < door_x or (abs(x - door_x) < 1e-6 and y < door_y):
        return _move_toward(x, y, door_x, door_y)
    return _move_toward(x, y, goal_x, goal_y)


def keydoor_random_policy(state: np.ndarray, rng: np.random.Generator) -> int:
    del state
    return int(rng.integers(0, 4))


def _move_toward(x: float, y: float, target_x: float, target_y: float) -> int:
    if x < target_x:
        return 3
    if y < target_y:
        return 1
    if x > target_x:
        return 2
    return 0


def collect_keydoor_trajectories(
    num_episodes: int,
    seed: int,
    policy_mix: tuple[float, float, float] = (0.20, 0.65, 0.15),
) -> list[OfflineTrajectory]:
    """采集 key-door 稀疏奖励轨迹，用于中期原型实验。"""
    rng = np.random.default_rng(seed)
    env = KeyDoorGridWorld()
    policies: list[Callable[[np.ndarray, np.random.Generator], int]] = [
        keydoor_random_policy,
        keydoor_near_goal_policy,
        keydoor_far_goal_policy,
    ]
    trajectories: list[OfflineTrajectory] = []

    for episode_idx in range(num_episodes):
        policy_idx = int(rng.choice(len(policies), p=np.asarray(policy_mix)))
        policy = policies[policy_idx]
        state, _ = env.reset(seed=seed + episode_idx)
        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []

        terminated = truncated = False
        while not (terminated or truncated):
            action = policy(np.asarray(state, dtype=np.float32), rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            state = next_state

        rewards_array = np.asarray(rewards, dtype=np.float32)
        trajectories.append(
            OfflineTrajectory(
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=rewards_array,
                returns_to_go=compute_returns_to_go(rewards_array),
                timesteps=np.arange(len(actions), dtype=np.int64),
                total_return=float(rewards_array.sum()),
            )
        )

    return trajectories


def collect_memory_trajectories(
    num_episodes: int,
    seed: int,
    policy_mix: tuple[float, float, float] = (0.0, 0.35, 0.65),
) -> list[OfflineTrajectory]:
    """使用随机、探索、专家脚本策略采集 MiniGrid Memory 离线轨迹。"""
    rng = np.random.default_rng(seed)
    trajectories: list[OfflineTrajectory] = []

    for episode_idx in range(num_episodes):
        env = make_memory_env()
        state, _ = env.reset(seed=seed + episode_idx)
        policy_idx = int(rng.choice(3, p=np.asarray(policy_mix)))
        if policy_idx == 0:
            policy = memory_random_policy
        elif policy_idx == 1:
            policy = make_memory_explore_policy(env, rng)
        else:
            policy = memory_expert_policy

        states: list[np.ndarray] = []
        actions: list[int] = []
        rewards: list[float] = []
        terminated = truncated = False

        while not (terminated or truncated):
            action = policy(np.asarray(state, dtype=np.float32), rng, env)
            next_state, reward, terminated, truncated, _ = env.step(action)
            states.append(np.asarray(state, dtype=np.float32))
            actions.append(action)
            rewards.append(float(reward))
            state = next_state

        rewards_array = np.asarray(rewards, dtype=np.float32)
        trajectories.append(
            OfflineTrajectory(
                states=np.asarray(states, dtype=np.float32),
                actions=np.asarray(actions, dtype=np.int64),
                rewards=rewards_array,
                returns_to_go=compute_returns_to_go(rewards_array),
                timesteps=np.arange(len(actions), dtype=np.int64),
                total_return=float(rewards_array.sum()),
            )
        )
        env.close()

    return trajectories


def build_decision_transformer_samples(
    trajectories: Sequence[OfflineTrajectory],
    context_length: int,
    state_dim: int,
    rtg_scale: float = 1.0,
) -> list[DTSequenceSample]:
    """将变长轨迹切成 DT 训练所需的定长窗口。"""
    samples: list[DTSequenceSample] = []
    for trajectory in trajectories:
        for end in range(len(trajectory.actions) - 1, -1, -1):
            start = max(0, end - context_length + 1)
            length = end - start + 1
            pad = context_length - length

            # 左侧补零后，真实时间步会对齐在窗口末尾，这样训练时的输入布局
            # 与评估时“只看最近 K 步历史”的自回归决策逻辑保持一致。
            states = np.zeros((context_length, state_dim), dtype=np.float32)
            actions = np.zeros(context_length, dtype=np.int64)
            returns_to_go = np.zeros(context_length, dtype=np.float32)
            timesteps = np.zeros(context_length, dtype=np.int64)
            mask = np.zeros(context_length, dtype=bool)

            states[pad:] = trajectory.states[start : end + 1]
            actions[pad:] = trajectory.actions[start : end + 1]
            returns_to_go[pad:] = trajectory.returns_to_go[start : end + 1] / rtg_scale
            timesteps[pad:] = trajectory.timesteps[start : end + 1]
            mask[pad:] = True

            samples.append(
                DTSequenceSample(
                    returns_to_go=returns_to_go,
                    states=states,
                    actions=actions,
                    timesteps=timesteps,
                    mask=mask,
                )
            )
    return samples


class BCDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """将轨迹展平成独立的状态-动作对，供普通 BC 训练。"""

    def __init__(self, trajectories: Sequence[OfflineTrajectory]) -> None:
        self.states = torch.as_tensor(
            np.concatenate([trajectory.states for trajectory in trajectories], axis=0),
            dtype=torch.float32,
        )
        self.actions = torch.as_tensor(
            np.concatenate([trajectory.actions for trajectory in trajectories], axis=0),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


class ReturnConditionedBCDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """将轨迹展平成 (state, RTG, action) 三元组，供 RCBC 训练。"""

    def __init__(self, trajectories: Sequence[OfflineTrajectory], rtg_scale: float) -> None:
        self.states = torch.as_tensor(
            np.concatenate([trajectory.states for trajectory in trajectories], axis=0),
            dtype=torch.float32,
        )
        self.returns_to_go = torch.as_tensor(
            np.concatenate([trajectory.returns_to_go for trajectory in trajectories], axis=0)
            / rtg_scale,
            dtype=torch.float32,
        )
        self.actions = torch.as_tensor(
            np.concatenate([trajectory.actions for trajectory in trajectories], axis=0),
            dtype=torch.long,
        )

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.states[index], self.returns_to_go[index], self.actions[index]


class StackedBCDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """将轨迹展平成短状态窗口，供 stacked BC 基线训练。"""

    def __init__(
        self,
        trajectories: Sequence[OfflineTrajectory],
        context_length: int,
        state_dim: int,
    ) -> None:
        stacked_states: list[np.ndarray] = []
        actions: list[int] = []
        for trajectory in trajectories:
            for end in range(len(trajectory.actions)):
                start = max(0, end - context_length + 1)
                length = end - start + 1
                pad = context_length - length
                # 轨迹开头不足 context_length 的部分统一补零，保证所有样本
                # 维度一致，同时保留“最近状态在窗口末尾”的布局。
                window = np.zeros((context_length, state_dim), dtype=np.float32)
                window[pad:] = trajectory.states[start : end + 1]
                stacked_states.append(window.reshape(-1))
                actions.append(int(trajectory.actions[end]))

        self.stacked_states = torch.as_tensor(np.asarray(stacked_states), dtype=torch.float32)
        self.actions = torch.as_tensor(np.asarray(actions), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.stacked_states[index], self.actions[index]


class DecisionTransformerDataset(Dataset[dict[str, torch.Tensor]]):
    """向 PyTorch 暴露固定长度 DT 训练窗口的数据集。"""

    def __init__(
        self,
        trajectories: Sequence[OfflineTrajectory],
        context_length: int,
        state_dim: int,
        rtg_scale: float = 500.0,
    ) -> None:
        self.samples = build_decision_transformer_samples(
            trajectories=trajectories,
            context_length=context_length,
            state_dim=state_dim,
            rtg_scale=rtg_scale,
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "returns_to_go": torch.as_tensor(sample.returns_to_go, dtype=torch.float32),
            "states": torch.as_tensor(sample.states, dtype=torch.float32),
            "actions": torch.as_tensor(sample.actions, dtype=torch.long),
            "timesteps": torch.as_tensor(sample.timesteps, dtype=torch.long),
            "mask": torch.as_tensor(sample.mask, dtype=torch.bool),
        }


def summarize_trajectories(trajectories: Sequence[OfflineTrajectory]) -> dict[str, float]:
    """统计轨迹集合的整体特征，用于报告与 sanity check。"""
    returns = np.asarray([trajectory.total_return for trajectory in trajectories], dtype=np.float32)
    lengths = np.asarray([len(trajectory.actions) for trajectory in trajectories], dtype=np.float32)
    return {
        "episodes": float(len(trajectories)),
        "transitions": float(lengths.sum()),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
    }
