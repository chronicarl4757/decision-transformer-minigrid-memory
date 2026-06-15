import numpy as np
import torch

from src.antmaze import (
    ContinuousDecisionTransformerDataset,
    ContinuousTrajectory,
    flatten_antmaze_observation,
)
from src.d4rl_continuous import flatten_continuous_observation
from src.models import ContinuousBCPolicy, ContinuousDecisionTransformerPolicy


def test_flatten_antmaze_observation_concatenates_goal_dict_fields() -> None:
    observation = {
        "observation": np.array([1.0, 2.0], dtype=np.float32),
        "achieved_goal": np.array([3.0, 4.0], dtype=np.float32),
        "desired_goal": np.array([5.0, 6.0], dtype=np.float32),
    }

    flat = flatten_antmaze_observation(observation)

    np.testing.assert_allclose(flat, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))


def test_continuous_dt_dataset_left_pads_state_action_windows() -> None:
    trajectory = ContinuousTrajectory(
        states=np.array([[1.0, 10.0], [2.0, 20.0]], dtype=np.float32),
        actions=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        rewards=np.array([0.0, 1.0], dtype=np.float32),
        returns_to_go=np.array([1.0, 1.0], dtype=np.float32),
        timesteps=np.array([0, 1], dtype=np.int64),
        total_return=1.0,
    )

    dataset = ContinuousDecisionTransformerDataset(
        trajectories=[trajectory],
        context_length=3,
        state_dim=2,
        action_dim=2,
        rtg_scale=1.0,
    )
    sample = dataset[0]

    assert sample["states"].shape == (3, 2)
    assert sample["actions"].shape == (3, 2)
    assert sample["mask"].tolist() == [False, True, True]
    np.testing.assert_allclose(sample["states"][1:].numpy(), trajectory.states)
    np.testing.assert_allclose(sample["actions"][1:].numpy(), trajectory.actions)




def test_flatten_continuous_observation_handles_flat_array() -> None:
    observation = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    flat = flatten_continuous_observation(observation)

    np.testing.assert_allclose(flat, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_continuous_models_return_action_vectors_and_clip_actions() -> None:
    action_low = np.array([-1.0, -0.5], dtype=np.float32)
    action_high = np.array([1.0, 0.5], dtype=np.float32)
    bc = ContinuousBCPolicy(state_dim=6, action_dim=2, hidden_dim=16)
    dt = ContinuousDecisionTransformerPolicy(
        state_dim=6,
        action_dim=2,
        context_length=4,
        embed_dim=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_timestep=50,
    )

    bc_actions = bc(torch.randn(5, 6))
    dt_actions = dt(
        returns_to_go=torch.randn(5, 4),
        states=torch.randn(5, 4, 6),
        actions=torch.randn(5, 4, 2),
        timesteps=torch.arange(4).repeat(5, 1),
        padding_mask=torch.ones(5, 4, dtype=torch.bool),
    )
    action = dt.act(
        returns_to_go=[1.0],
        states=[np.zeros(6, dtype=np.float32)],
        actions=[np.zeros(2, dtype=np.float32)],
        timesteps=[0],
        action_low=action_low,
        action_high=action_high,
        device=torch.device("cpu"),
    )

    assert bc_actions.shape == (5, 2)
    assert dt_actions.shape == (5, 4, 2)
    assert action.shape == (2,)
    assert np.all(action >= action_low)
    assert np.all(action <= action_high)
