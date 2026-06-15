from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import compute_returns_to_go

ANTMAZE_DATASET_CANDIDATES = (
    "D4RL/antmaze/umaze-v2",
    "D4RL/antmaze/umaze-v1",
)


@dataclass(frozen=True)
class ContinuousTrajectory:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    returns_to_go: np.ndarray
    timesteps: np.ndarray
    total_return: float


@dataclass(frozen=True)
class AntMazeDatasetBundle:
    dataset_id: str
    trajectories: list[ContinuousTrajectory]
    state_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray


def flatten_antmaze_observation(observation) -> np.ndarray:
    if isinstance(observation, dict):
        parts = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in observation:
                parts.append(np.asarray(observation[key], dtype=np.float32).reshape(-1))
        if parts:
            return np.concatenate(parts, axis=0).astype(np.float32)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def normalize_trajectories(
    trajectories: Sequence[ContinuousTrajectory],
) -> tuple[list[ContinuousTrajectory], np.ndarray, np.ndarray]:
    states = np.concatenate([trajectory.states for trajectory in trajectories], axis=0)
    mean = states.mean(axis=0).astype(np.float32)
    std = (states.std(axis=0) + 1e-6).astype(np.float32)
    normalized = [
        ContinuousTrajectory(
            states=((trajectory.states - mean) / std).astype(np.float32),
            actions=trajectory.actions,
            rewards=trajectory.rewards,
            returns_to_go=trajectory.returns_to_go,
            timesteps=trajectory.timesteps,
            total_return=trajectory.total_return,
        )
        for trajectory in trajectories
    ]
    return normalized, mean, std


def load_antmaze_dataset(
    dataset_ids: Sequence[str] = ANTMAZE_DATASET_CANDIDATES,
    max_episodes: int | None = None,
    normalize_states: bool = True,
) -> AntMazeDatasetBundle:
    try:
        import minari
    except ImportError as exc:
        raise ImportError(
            "AntMaze extension requires Minari. Install Farama dependencies on the remote "
            "GPU host, e.g. `pip install minari gymnasium-robotics[mujoco] mujoco`."
        ) from exc

    last_error: Exception | None = None
    for dataset_id in dataset_ids:
        try:
            dataset = minari.load_dataset(dataset_id, download=True)
            trajectories = _trajectories_from_minari_dataset(dataset, max_episodes=max_episodes)
            env = dataset.recover_environment()
            low = np.asarray(env.action_space.low, dtype=np.float32)
            high = np.asarray(env.action_space.high, dtype=np.float32)
            env.close()
            if normalize_states:
                trajectories, mean, std = normalize_trajectories(trajectories)
            else:
                state_dim = trajectories[0].states.shape[1]
                mean = np.zeros(state_dim, dtype=np.float32)
                std = np.ones(state_dim, dtype=np.float32)
            return AntMazeDatasetBundle(
                dataset_id=dataset_id,
                trajectories=trajectories,
                state_dim=int(trajectories[0].states.shape[1]),
                action_dim=int(trajectories[0].actions.shape[1]),
                action_low=low,
                action_high=high,
                state_mean=mean,
                state_std=std,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    raise RuntimeError(f"Could not load AntMaze dataset candidates: {dataset_ids}") from last_error


def _trajectories_from_minari_dataset(
    dataset,
    max_episodes: int | None,
) -> list[ContinuousTrajectory]:
    episodes = dataset.iterate_episodes()
    return trajectories_from_minari_episodes(episodes, max_episodes=max_episodes)


def trajectories_from_minari_episodes(
    episodes: Iterable,
    max_episodes: int | None = None,
) -> list[ContinuousTrajectory]:
    trajectories: list[ContinuousTrajectory] = []
    for episode_idx, episode in enumerate(episodes):
        if max_episodes is not None and episode_idx >= max_episodes:
            break
        observations = _episode_field(episode, "observations")
        actions = np.asarray(_episode_field(episode, "actions"), dtype=np.float32)
        rewards = np.asarray(_episode_field(episode, "rewards"), dtype=np.float32)
        states = np.asarray(
            [
                flatten_antmaze_observation(_index_observation(observations, idx))
                for idx in range(len(actions))
            ],
            dtype=np.float32,
        )
        trajectories.append(
            ContinuousTrajectory(
                states=states,
                actions=actions,
                rewards=rewards,
                returns_to_go=compute_returns_to_go(rewards),
                timesteps=np.arange(len(actions), dtype=np.int64),
                total_return=float(rewards.sum()),
            )
        )
    if not trajectories:
        raise ValueError("No trajectories were loaded from the AntMaze dataset.")
    return trajectories


def _episode_field(episode, name: str):
    if hasattr(episode, name):
        return getattr(episode, name)
    return episode[name]


def _index_observation(observations, index: int):
    if isinstance(observations, dict):
        return {key: value[index] for key, value in observations.items()}
    return observations[index]


class ContinuousBCDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, trajectories: Sequence[ContinuousTrajectory]) -> None:
        self.states = torch.as_tensor(
            np.concatenate([trajectory.states for trajectory in trajectories], axis=0),
            dtype=torch.float32,
        )
        self.actions = torch.as_tensor(
            np.concatenate([trajectory.actions for trajectory in trajectories], axis=0),
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.states[index], self.actions[index]


class ContinuousDecisionTransformerDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        trajectories: Sequence[ContinuousTrajectory],
        context_length: int,
        state_dim: int,
        action_dim: int,
        rtg_scale: float = 1.0,
    ) -> None:
        self.samples = []
        for trajectory in trajectories:
            for end in range(len(trajectory.actions) - 1, -1, -1):
                start = max(0, end - context_length + 1)
                length = end - start + 1
                pad = context_length - length

                states = np.zeros((context_length, state_dim), dtype=np.float32)
                actions = np.zeros((context_length, action_dim), dtype=np.float32)
                returns_to_go = np.zeros(context_length, dtype=np.float32)
                timesteps = np.zeros(context_length, dtype=np.int64)
                mask = np.zeros(context_length, dtype=bool)

                states[pad:] = trajectory.states[start : end + 1]
                actions[pad:] = trajectory.actions[start : end + 1]
                returns_to_go[pad:] = trajectory.returns_to_go[start : end + 1] / rtg_scale
                timesteps[pad:] = trajectory.timesteps[start : end + 1]
                mask[pad:] = True

                self.samples.append((returns_to_go, states, actions, timesteps, mask))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        returns_to_go, states, actions, timesteps, mask = self.samples[index]
        return {
            "returns_to_go": torch.as_tensor(returns_to_go, dtype=torch.float32),
            "states": torch.as_tensor(states, dtype=torch.float32),
            "actions": torch.as_tensor(actions, dtype=torch.float32),
            "timesteps": torch.as_tensor(timesteps, dtype=torch.long),
            "mask": torch.as_tensor(mask, dtype=torch.bool),
        }
