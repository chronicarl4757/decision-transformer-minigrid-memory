from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.antmaze import (
    ContinuousBCDataset,
    ContinuousDecisionTransformerDataset,
    ContinuousTrajectory,
    normalize_trajectories,
)
from src.data import compute_returns_to_go

HOPPER_DATASET_CANDIDATES = (
    "mujoco/hopper/medium-v0",
)


@dataclass(frozen=True)
class ContinuousDatasetBundle:
    dataset_id: str
    env_id: str
    trajectories: list[ContinuousTrajectory]
    state_dim: int
    action_dim: int
    action_low: np.ndarray
    action_high: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray


def flatten_continuous_observation(observation) -> np.ndarray:
    if isinstance(observation, dict):
        parts = []
        for key in ("observation", "achieved_goal", "desired_goal"):
            if key in observation:
                parts.append(np.asarray(observation[key], dtype=np.float32).reshape(-1))
        if parts:
            return np.concatenate(parts, axis=0).astype(np.float32)
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def load_continuous_dataset(
    dataset_ids: Sequence[str],
    max_episodes: int | None = None,
    normalize_states: bool = True,
) -> ContinuousDatasetBundle:
    try:
        import minari
    except ImportError as exc:
        raise ImportError(
            "D4RL reproduction requires Minari. Install Farama dependencies on the remote "
            "GPU host without changing the installed CUDA torch."
        ) from exc

    last_error: Exception | None = None
    for dataset_id in dataset_ids:
        try:
            dataset = minari.load_dataset(dataset_id, download=True)
            trajectories = trajectories_from_minari_episodes(
                dataset.iterate_episodes(),
                max_episodes=max_episodes,
            )
            env = dataset.recover_environment()
            low = np.asarray(env.action_space.low, dtype=np.float32)
            high = np.asarray(env.action_space.high, dtype=np.float32)
            env_id = getattr(getattr(env, "spec", None), "id", dataset_id)
            env.close()
            if normalize_states:
                trajectories, mean, std = normalize_trajectories(trajectories)
            else:
                state_dim = trajectories[0].states.shape[1]
                mean = np.zeros(state_dim, dtype=np.float32)
                std = np.ones(state_dim, dtype=np.float32)
            return ContinuousDatasetBundle(
                dataset_id=dataset_id,
                env_id=str(env_id),
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
    raise RuntimeError(
        f"Could not load continuous dataset candidates: {dataset_ids}"
    ) from last_error


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
                flatten_continuous_observation(_index_observation(observations, idx))
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
        raise ValueError("No trajectories were loaded from the continuous-control dataset.")
    return trajectories


def _episode_field(episode, name: str):
    if hasattr(episode, name):
        return getattr(episode, name)
    return episode[name]


def _index_observation(observations, index: int):
    if isinstance(observations, dict):
        return {key: value[index] for key, value in observations.items()}
    return observations[index]


__all__ = [
    "ContinuousBCDataset",
    "ContinuousDatasetBundle",
    "ContinuousDecisionTransformerDataset",
    "ContinuousTrajectory",
    "HOPPER_DATASET_CANDIDATES",
    "flatten_continuous_observation",
    "load_continuous_dataset",
    "trajectories_from_minari_episodes",
]
