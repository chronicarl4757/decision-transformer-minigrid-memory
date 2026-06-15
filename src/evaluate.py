"""模型 rollout 评估与轨迹录制工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from src.envs.memory import make_memory_env
from src.models import (
    BehaviorCloningPolicy,
    DecisionTransformerPolicy,
    ReturnConditionedBCPolicy,
    StackedBCPolicy,
)


@dataclass
class TrajectoryStep:
    """导出演示轨迹时记录的单步信息。"""

    x: int
    y: int
    direction: int
    observation: list[float]
    action: int
    reward: float


@dataclass
class RecordedTrajectory:
    """用于 JSON 导出的完整评估轨迹。"""

    episode_id: int
    model: str
    target_return: float
    steps: list[TrajectoryStep]
    total_return: float
    success: bool
    terminal_type: str


@torch.no_grad()
def evaluate_bc(
    model: BehaviorCloningPolicy,
    episodes: int,
    seed: int,
    device: torch.device,
    env_name: str = "memory",
    success_threshold: float = 0.80,
) -> dict[str, float]:
    """评估普通 BC 在指定环境中的 rollout 表现。"""
    env = make_env(env_name)
    returns: list[float] = []
    terminal_types: list[str] = []
    model.eval()
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action = model.act(state_tensor)
            state, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        returns.append(total)
        terminal_types.append(str(info.get("terminal_type", "none")))
    env.close()
    return _return_stats(returns, terminal_types, success_threshold)


@torch.no_grad()
def evaluate_return_conditioned_bc(
    model: ReturnConditionedBCPolicy,
    episodes: int,
    seed: int,
    device: torch.device,
    target_return: float,
    rtg_scale: float,
    env_name: str = "memory",
    success_threshold: float = 0.80,
) -> dict[str, float]:
    """评估 RCBC，并在 rollout 过程中持续更新剩余目标回报。"""
    env = make_env(env_name)
    returns: list[float] = []
    terminal_types: list[str] = []
    model.eval()
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        desired_return = float(target_return)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action = model.act(state_tensor, desired_return / rtg_scale)
            state, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
            desired_return = max(desired_return - float(reward), 0.0)
        returns.append(total)
        terminal_types.append(str(info.get("terminal_type", "none")))
    env.close()
    return _return_stats(returns, terminal_types, success_threshold)


@torch.no_grad()
def evaluate_stacked_bc(
    model: StackedBCPolicy,
    episodes: int,
    seed: int,
    device: torch.device,
    env_name: str = "memory",
    success_threshold: float = 0.80,
) -> dict[str, float]:
    """评估 stacked BC 基线。"""
    env = make_env(env_name)
    returns: list[float] = []
    terminal_types: list[str] = []
    model.eval()
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        states: list[np.ndarray] = []
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            states.append(np.asarray(state, dtype=np.float32))
            stacked_state = _stack_recent_states(
                states=states,
                context_length=model.context_length,
                state_dim=model.state_dim,
            )
            action = model.act(stacked_state, device=device)
            state, reward, terminated, truncated, info = env.step(action)
            total += float(reward)
        returns.append(total)
        terminal_types.append(str(info.get("terminal_type", "none")))
    env.close()
    return _return_stats(returns, terminal_types, success_threshold)


@torch.no_grad()
def evaluate_dt(
    model: DecisionTransformerPolicy,
    episodes: int,
    seed: int,
    device: torch.device,
    target_return: float = 0.95,
    rtg_scale: float = 1.0,
    env_name: str = "memory",
    success_threshold: float = 0.80,
) -> dict[str, float]:
    """评估 DT：每一步都用最近上下文重新构造一次模型输入。"""
    env = make_env(env_name)
    returns: list[float] = []
    terminal_types: list[str] = []
    model.eval()

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        states: list[np.ndarray] = []
        actions: list[int] = []
        returns_to_go: list[float] = []
        timesteps: list[int] = []
        desired_return = float(target_return)
        total = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            states.append(np.asarray(state, dtype=np.float32))
            returns_to_go.append(desired_return / rtg_scale)
            timesteps.append(len(timesteps))
            # 评估时当前时刻的动作还未知，因此在动作序列末尾先补一个占位，
            # 再让模型预测最后一个 state token 对应的动作。
            action_context = actions + [0]
            action = _predict_dt_action(
                model=model,
                states=states,
                actions=action_context,
                returns_to_go=returns_to_go,
                timesteps=timesteps,
                device=device,
            )
            state, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            total += float(reward)
            desired_return = max(desired_return - float(reward), 0.0)
        returns.append(total)
        terminal_types.append(str(info.get("terminal_type", "none")))

    env.close()
    return _return_stats(returns, terminal_types, success_threshold)


def _stack_recent_states(
    states: list[np.ndarray],
    context_length: int,
    state_dim: int,
) -> np.ndarray:
    """将最近若干步状态整理成固定长度窗口，供 stacked BC 使用。"""
    start = max(0, len(states) - context_length)
    actual_length = len(states) - start
    pad = context_length - actual_length
    state_array = np.zeros((context_length, state_dim), dtype=np.float32)
    state_array[pad:] = np.asarray(states[start:], dtype=np.float32)
    return state_array.reshape(-1)


def _predict_dt_action(
    model: DecisionTransformerPolicy,
    states: list[np.ndarray],
    actions: list[int],
    returns_to_go: list[float],
    timesteps: list[int],
    device: torch.device,
) -> int:
    """将最近历史整理成 DT 输入，并读取最后一个动作预测。"""
    context_length = model.context_length
    state_dim = states[0].shape[0]
    start = max(0, len(states) - context_length)
    actual_length = len(states) - start
    pad = context_length - actual_length

    # 这里的补零方式必须与训练数据构造保持一致，否则评估时的输入分布
    # 会和训练阶段不匹配。
    state_array = np.zeros((context_length, state_dim), dtype=np.float32)
    action_array = np.zeros(context_length, dtype=np.int64)
    rtg_array = np.zeros(context_length, dtype=np.float32)
    timestep_array = np.zeros(context_length, dtype=np.int64)
    mask_array = np.zeros(context_length, dtype=bool)

    state_array[pad:] = np.asarray(states[start:], dtype=np.float32)
    action_array[pad:] = np.asarray(actions[start:], dtype=np.int64)
    rtg_array[pad:] = np.asarray(returns_to_go[start:], dtype=np.float32)
    timestep_array[pad:] = np.asarray(timesteps[start:], dtype=np.int64)
    mask_array[pad:] = True

    logits = model(
        returns_to_go=torch.as_tensor(rtg_array, dtype=torch.float32, device=device).unsqueeze(0),
        states=torch.as_tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0),
        actions=torch.as_tensor(action_array, dtype=torch.long, device=device).unsqueeze(0),
        timesteps=torch.as_tensor(timestep_array, dtype=torch.long, device=device).unsqueeze(0),
        padding_mask=torch.as_tensor(mask_array, dtype=torch.bool, device=device).unsqueeze(0),
    )
    return int(torch.argmax(logits[0, -1], dim=-1).item())


def make_env(env_name: str):
    """根据环境名创建评估环境。"""
    if env_name == "memory":
        return make_memory_env()
    raise ValueError(f"Unsupported env_name: {env_name}")


def _return_stats(
    returns: list[float],
    terminal_types: list[str],
    success_threshold: float,
) -> dict[str, float]:
    """将多条 rollout 的回报与终止类型汇总成统一指标。"""
    values = np.asarray(returns, dtype=np.float32)
    return {
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "return_min": float(values.min()),
        "return_max": float(values.max()),
        "success_rate": float(np.mean(values >= success_threshold)),
        "near_goal_rate": float(np.mean(np.asarray(terminal_types) == "near_goal")),
        "far_goal_rate": float(np.mean(np.asarray(terminal_types) == "far_goal")),
    }


# ──────────────────────────────────────────────
# Trajectory-recording evaluation (for demo export)
# ──────────────────────────────────────────────

def _agent_position(env: gym.Env) -> tuple[int, int, int]:
    """Return (x, y, direction) from the unwrapped MiniGrid env."""
    unwrapped = env.unwrapped
    return (int(unwrapped.agent_pos[0]), int(unwrapped.agent_pos[1]), int(unwrapped.agent_dir))


@torch.no_grad()
def record_bc(
    model: BehaviorCloningPolicy,
    episodes: int,
    seed: int,
    device: torch.device,
    target_return: float = 0.0,
    success_threshold: float = 0.80,
) -> list[RecordedTrajectory]:
    env = make_env("memory")
    trajectories: list[RecordedTrajectory] = []
    model.eval()

    for ep_idx in range(episodes):
        state, _ = env.reset(seed=seed + ep_idx)
        steps: list[TrajectoryStep] = []
        total = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            x, y, d = _agent_position(env)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action = model.act(state_tensor)
            next_state, reward, terminated, truncated, info = env.step(action)
            steps.append(TrajectoryStep(
                x=x, y=y, direction=d,
                observation=state.tolist() if isinstance(state, np.ndarray) else list(state),
                action=int(action), reward=float(reward),
            ))
            state = next_state
            total += float(reward)

        trajectories.append(RecordedTrajectory(
            episode_id=ep_idx, model="bc", target_return=float(target_return),
            steps=steps, total_return=total,
            success=total >= success_threshold,
            terminal_type=str(info.get("terminal_type", "none")),
        ))

    env.close()
    return trajectories


@torch.no_grad()
def record_rcbc(
    model: ReturnConditionedBCPolicy,
    episodes: int,
    seed: int,
    device: torch.device,
    target_return: float = 0.95,
    rtg_scale: float = 1.0,
    success_threshold: float = 0.80,
) -> list[RecordedTrajectory]:
    env = make_env("memory")
    trajectories: list[RecordedTrajectory] = []
    model.eval()

    for ep_idx in range(episodes):
        state, _ = env.reset(seed=seed + ep_idx)
        desired_return = float(target_return)
        steps: list[TrajectoryStep] = []
        total = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            x, y, d = _agent_position(env)
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
            action = model.act(state_tensor, desired_return / rtg_scale)
            next_state, reward, terminated, truncated, info = env.step(action)
            steps.append(TrajectoryStep(
                x=x, y=y, direction=d,
                observation=state.tolist() if isinstance(state, np.ndarray) else list(state),
                action=int(action), reward=float(reward),
            ))
            state = next_state
            total += float(reward)
            desired_return = max(desired_return - float(reward), 0.0)

        trajectories.append(RecordedTrajectory(
            episode_id=ep_idx, model="rcbc", target_return=float(target_return),
            steps=steps, total_return=total,
            success=total >= success_threshold,
            terminal_type=str(info.get("terminal_type", "none")),
        ))

    env.close()
    return trajectories


@torch.no_grad()
def record_dt(
    model: DecisionTransformerPolicy,
    episodes: int,
    seed: int,
    device: torch.device,
    target_return: float = 0.95,
    rtg_scale: float = 1.0,
    success_threshold: float = 0.80,
) -> list[RecordedTrajectory]:
    env = make_env("memory")
    trajectories: list[RecordedTrajectory] = []
    model.eval()

    for ep_idx in range(episodes):
        state, _ = env.reset(seed=seed + ep_idx)
        states: list[np.ndarray] = []
        actions: list[int] = []
        returns_to_go: list[float] = []
        timesteps: list[int] = []
        desired_return = float(target_return)
        recorded_steps: list[TrajectoryStep] = []
        total = 0.0
        terminated = truncated = False

        while not (terminated or truncated):
            x, y, d = _agent_position(env)
            states.append(np.asarray(state, dtype=np.float32))
            returns_to_go.append(desired_return / rtg_scale)
            timesteps.append(len(timesteps))
            action_context = actions + [0]
            action = _predict_dt_action(
                model=model, states=states, actions=action_context,
                returns_to_go=returns_to_go, timesteps=timesteps, device=device,
            )
            next_state, reward, terminated, truncated, info = env.step(action)
            recorded_steps.append(TrajectoryStep(
                x=x, y=y, direction=d,
                observation=state.tolist() if isinstance(state, np.ndarray) else list(state),
                action=int(action), reward=float(reward),
            ))
            state = next_state
            actions.append(action)
            total += float(reward)
            desired_return = max(desired_return - float(reward), 0.0)

        trajectories.append(RecordedTrajectory(
            episode_id=ep_idx, model="dt", target_return=float(target_return),
            steps=recorded_steps, total_return=total,
            success=total >= success_threshold,
            terminal_type=str(info.get("terminal_type", "none")),
        ))

    env.close()
    return trajectories


def trajectories_to_json(trajectories: list[RecordedTrajectory]) -> list[dict[str, Any]]:
    return [
        {
            "episode_id": t.episode_id,
            "model": t.model,
            "target_return": t.target_return,
            "steps": [
                {
                    "x": s.x, "y": s.y, "direction": s.direction,
                    "observation": s.observation,
                    "action": s.action, "reward": s.reward,
                }
                for s in t.steps
            ],
            "total_return": t.total_return,
            "success": t.success,
            "terminal_type": t.terminal_type,
        }
        for t in trajectories
    ]
