import numpy as np

from src.data import collect_gridworld_trajectories, collect_keydoor_trajectories
from src.envs.gridworld import KeyDoorGridWorld, TwoGoalGridWorld


def test_gridworld_near_and_far_goals_have_distinct_returns() -> None:
    env = TwoGoalGridWorld()
    env.reset(seed=0)

    near_actions = [1, 1]
    near_reward = 0.0
    for action in near_actions:
        _, reward, terminated, truncated, info = env.step(action)
        near_reward += reward
    assert terminated
    assert not truncated
    assert near_reward == 1.0
    assert info["terminal_type"] == "near_goal"

    env.reset(seed=0)
    far_actions = [3, 3, 3, 3, 3, 3]
    far_reward = 0.0
    for action in far_actions:
        _, reward, terminated, truncated, info = env.step(action)
        far_reward += reward
    assert terminated
    assert not truncated
    assert far_reward == 3.0
    assert info["terminal_type"] == "far_goal"


def test_gridworld_observation_exposes_goal_ambiguity_at_start() -> None:
    env = TwoGoalGridWorld()
    observation, _ = env.reset(seed=0)

    assert observation.shape == (6,)
    np.testing.assert_allclose(observation[:2], np.array([0.0, 0.0], dtype=np.float32))
    assert env.action_space.n == 4


def test_collect_gridworld_trajectories_contains_mixed_returns() -> None:
    trajectories = collect_gridworld_trajectories(num_episodes=30, seed=7)
    returns = {trajectory.total_return for trajectory in trajectories}

    assert 0.0 in returns
    assert 1.0 in returns
    assert 3.0 in returns
    assert all(trajectory.states.shape[1] == 6 for trajectory in trajectories)
    assert all(trajectory.actions.max(initial=0) < 4 for trajectory in trajectories)


def test_keydoor_blocks_door_until_key_is_collected() -> None:
    env = KeyDoorGridWorld()
    env.reset(seed=0)

    # Move to the door without collecting the key.
    for action in [1, 1, 1, 1, 3, 3, 3]:
        observation, _, terminated, truncated, _ = env.step(action)
    assert not terminated
    assert not truncated
    assert observation[0] < 4 / 8
    assert observation[2] == 0.0


def test_keydoor_success_requires_key_then_door_then_goal() -> None:
    env = KeyDoorGridWorld()
    env.reset(seed=0)

    actions = [1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 1, 1]
    total = 0.0
    for action in actions:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward

    assert terminated
    assert not truncated
    assert total == 5.0
    assert info["terminal_type"] == "far_goal"


def test_collect_keydoor_trajectories_contains_sparse_mixed_returns() -> None:
    trajectories = collect_keydoor_trajectories(num_episodes=40, seed=11)
    returns = {trajectory.total_return for trajectory in trajectories}

    assert 0.0 in returns
    assert 1.0 in returns
    assert 5.0 in returns
    assert all(trajectory.states.shape[1] == 11 for trajectory in trajectories)
