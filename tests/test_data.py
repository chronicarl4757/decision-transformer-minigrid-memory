import numpy as np

from src.data import (
    OfflineTrajectory,
    StackedBCDataset,
    build_decision_transformer_samples,
    compute_returns_to_go,
)


def test_compute_returns_to_go_uses_future_rewards() -> None:
    rewards = [1.0, 0.5, 2.0]

    returns = compute_returns_to_go(rewards)

    np.testing.assert_allclose(returns, np.array([3.5, 2.5, 2.0], dtype=np.float32))


def test_decision_transformer_samples_pad_left_and_mask_real_steps() -> None:
    trajectory = OfflineTrajectory(
        states=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        actions=np.array([0, 1, 0], dtype=np.int64),
        rewards=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        returns_to_go=np.array([3.0, 2.0, 1.0], dtype=np.float32),
        timesteps=np.array([0, 1, 2], dtype=np.int64),
        total_return=3.0,
    )

    samples = build_decision_transformer_samples([trajectory], context_length=5, state_dim=4)
    sample = samples[0]

    assert sample.states.shape == (5, 4)
    assert sample.actions.tolist() == [0, 0, 0, 1, 0]
    assert sample.mask.tolist() == [False, False, True, True, True]
    np.testing.assert_allclose(sample.returns_to_go, np.array([0.0, 0.0, 3.0, 2.0, 1.0]))
    assert sample.timesteps.tolist() == [0, 0, 0, 1, 2]


def test_stacked_bc_dataset_flattens_left_padded_recent_states() -> None:
    trajectory = OfflineTrajectory(
        states=np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=np.float32,
        ),
        actions=np.array([0, 1, 2], dtype=np.int64),
        rewards=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        returns_to_go=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        timesteps=np.array([0, 1, 2], dtype=np.int64),
        total_return=1.0,
    )

    dataset = StackedBCDataset([trajectory], context_length=3, state_dim=2)
    first_states, first_action = dataset[0]
    last_states, last_action = dataset[2]

    assert first_states.shape == (6,)
    np.testing.assert_allclose(first_states.numpy(), np.array([0.0, 0.0, 0.0, 0.0, 1.0, 10.0]))
    assert int(first_action) == 0
    np.testing.assert_allclose(last_states.numpy(), np.array([1.0, 10.0, 2.0, 20.0, 3.0, 30.0]))
    assert int(last_action) == 2
