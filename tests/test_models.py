import numpy as np
import torch

from src.models import (
    BehaviorCloningPolicy,
    DecisionTransformerPolicy,
    ReturnConditionedBCPolicy,
    StackedBCPolicy,
)


def test_behavior_cloning_forward_returns_action_logits() -> None:
    model = BehaviorCloningPolicy(state_dim=4, action_dim=2, hidden_dim=16)
    states = torch.randn(7, 4)

    logits = model(states)

    assert logits.shape == (7, 2)


def test_return_conditioned_bc_forward_uses_state_and_return_condition() -> None:
    model = ReturnConditionedBCPolicy(state_dim=6, action_dim=4, hidden_dim=16)
    states = torch.randn(7, 6)
    returns_to_go = torch.randn(7)

    logits = model(states, returns_to_go)

    assert logits.shape == (7, 4)


def test_stacked_bc_forward_returns_action_logits_and_discrete_action() -> None:
    model = StackedBCPolicy(state_dim=6, action_dim=4, context_length=5, hidden_dim=16)
    stacked_states = torch.randn(7, 30)

    logits = model(stacked_states)
    action = model.act(np.zeros(30, dtype=np.float32), device=torch.device("cpu"))

    assert logits.shape == (7, 4)
    assert 0 <= action < 4


def test_decision_transformer_forward_returns_per_step_action_logits() -> None:
    model = DecisionTransformerPolicy(
        state_dim=6,
        action_dim=4,
        context_length=6,
        embed_dim=16,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        max_timestep=500,
    )
    batch_size = 3
    context_length = 6

    logits = model(
        returns_to_go=torch.randn(batch_size, context_length),
        states=torch.randn(batch_size, context_length, 6),
        actions=torch.randint(0, 4, (batch_size, context_length)),
        timesteps=torch.arange(context_length).repeat(batch_size, 1),
        padding_mask=torch.ones(batch_size, context_length, dtype=torch.bool),
    )

    assert logits.shape == (batch_size, context_length, 4)
