from __future__ import annotations

import numpy as np
import torch

from src.data import collect_memory_trajectories
from src.evaluate import evaluate_stacked_bc, make_env
from src.models import StackedBCPolicy


def test_memory_env_observation_shape_and_action_space() -> None:
    env = make_env("memory")
    state, _ = env.reset(seed=0)

    assert np.asarray(state).shape == (11,)
    assert env.action_space.n == 7

    env.close()


def test_memory_reset_starts_with_visible_one_hot_cue() -> None:
    env = make_env("memory")

    for seed in range(20):
        state, _ = env.reset(seed=seed)
        key_visible, ball_visible = state[3:5]

        assert tuple(env.unwrapped.agent_pos) == (2, env.unwrapped.height // 2)
        assert key_visible + ball_visible == 1.0
        assert {float(key_visible), float(ball_visible)} == {0.0, 1.0}
        assert state[7:11].sum() == 0.0

    env.close()


def test_memory_cue_disappears_after_leaving_start_room() -> None:
    env = make_env("memory")
    state, _ = env.reset(seed=0)
    assert state[3] + state[4] == 1.0

    env.unwrapped.agent_pos = np.asarray((5, env.unwrapped.height // 2))
    state = env.observation({})

    assert state[3] == 0.0
    assert state[4] == 0.0

    env.close()


def test_memory_terminal_objects_are_visible_at_fork_without_start_cue() -> None:
    env = make_env("memory")
    env.reset(seed=0)
    hallway_x = env.unwrapped.width - 4
    env.unwrapped.agent_pos = np.asarray((hallway_x, env.unwrapped.height // 2))
    state = env.observation({})

    assert state[3] == 0.0
    assert state[4] == 0.0
    assert state[7] + state[8] == 1.0
    assert state[9] + state[10] == 1.0

    env.close()


def test_collect_memory_trajectories_contains_mixed_returns() -> None:
    trajectories = collect_memory_trajectories(num_episodes=80, seed=7)
    returns = [trajectory.total_return for trajectory in trajectories]

    assert min(returns) <= 0.05
    assert max(returns) >= 0.80
    assert all(trajectory.states.shape[1] == 11 for trajectory in trajectories)
    assert all(trajectory.actions.max(initial=0) < 7 for trajectory in trajectories)


def test_memory_expert_policy_achieves_high_return() -> None:
    trajectories = collect_memory_trajectories(
        num_episodes=8,
        seed=11,
        policy_mix=(0.0, 0.0, 1.0),
    )

    assert np.mean([trajectory.total_return for trajectory in trajectories]) >= 0.80


def test_evaluate_stacked_bc_returns_memory_metrics() -> None:
    model = StackedBCPolicy(state_dim=11, action_dim=7, context_length=5, hidden_dim=16)

    metrics = evaluate_stacked_bc(
        model=model,
        episodes=1,
        seed=17,
        device=torch.device("cpu"),
        env_name="memory",
        success_threshold=0.80,
    )

    assert "return_mean" in metrics
    assert "success_rate" in metrics
