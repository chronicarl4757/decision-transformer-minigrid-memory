"""离线强化学习复现实验中使用的策略网络模块。"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class BehaviorCloningPolicy(nn.Module):
    """普通行为克隆策略：当前状态直接映射到动作 logits。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> int:
        logits = self.forward(state.unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())


class ReturnConditionedBCPolicy(nn.Module):
    """带回报条件的前馈基线：输入当前状态和标量 RTG。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor, returns_to_go: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([states, returns_to_go.unsqueeze(-1)], dim=-1)
        return self.net(inputs)

    @torch.no_grad()
    def act(self, state: torch.Tensor, desired_return: float) -> int:
        rtg = torch.as_tensor([desired_return], dtype=state.dtype, device=state.device)
        logits = self.forward(state.unsqueeze(0), rtg)
        return int(torch.argmax(logits, dim=-1).item())


class StackedBCPolicy(nn.Module):
    """短历史前馈基线：输入最近若干步状态拼接后的向量。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_length: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.net = nn.Sequential(
            nn.Linear(state_dim * context_length, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, stacked_states: torch.Tensor) -> torch.Tensor:
        return self.net(stacked_states)

    @torch.no_grad()
    def act(
        self,
        stacked_state: np.ndarray | torch.Tensor,
        device: torch.device | None = None,
    ) -> int:
        stacked_state = torch.as_tensor(stacked_state, dtype=torch.float32, device=device)
        logits = self.forward(stacked_state.unsqueeze(0))
        return int(torch.argmax(logits, dim=-1).item())


class ContinuousBCPolicy(nn.Module):
    """连续动作 BC 基线，用于 Hopper、AntMaze 等任务。"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)

    @torch.no_grad()
    def act(
        self,
        state: np.ndarray | torch.Tensor,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
        action = self.forward(state_tensor.unsqueeze(0))[0].detach().cpu().numpy()
        return np.clip(action, action_low, action_high)


class DecisionTransformerPolicy(nn.Module):
    """离散动作版本的 Decision Transformer，用于 Memory/GridWorld 任务。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_length: int,
        embed_dim: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_timestep: int = 500,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.context_length = context_length
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Embedding(action_dim, embed_dim)
        self.embed_timestep = nn.Embedding(max_timestep + 1, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.predict_action = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, context_length = returns_to_go.shape
        timesteps = timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)
        time_embeddings = self.embed_timestep(timesteps)

        # 每个真实时间步都会展开成三个 token：RTG、state、action。
        # 三者共享同一个 timestep embedding，用来同时保留时间顺序
        # 和模态差异。
        return_embeddings = self.embed_return(returns_to_go.unsqueeze(-1)) + time_embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions.clamp(min=0)) + time_embeddings

        tokens = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings),
            dim=2,
        ).reshape(batch_size, 3 * context_length, -1)
        tokens = self.embed_ln(tokens)

        # 因果掩码禁止当前 token 访问未来 token，保证训练阶段
        # 与 rollout 阶段的自回归决策方式一致。
        causal_mask = torch.triu(
            torch.ones(
                3 * context_length,
                3 * context_length,
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        src_key_padding_mask = None
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask.repeat_interleave(3, dim=1)

        hidden = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        # 按原论文设定，只使用每个 state token 位置的隐藏表示
        # 来预测对应时间步的动作。
        hidden = hidden.reshape(batch_size, context_length, 3, -1)
        state_hidden = hidden[:, :, 1, :]
        return self.predict_action(state_hidden)


class ContinuousDecisionTransformerPolicy(nn.Module):
    """连续动作版本的 Decision Transformer，用于 D4RL 类控制任务。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        context_length: int,
        embed_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        dropout: float = 0.1,
        max_timestep: int = 1000,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.context_length = context_length
        self.embed_return = nn.Linear(1, embed_dim)
        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_timestep = nn.Embedding(max_timestep + 1, embed_dim)
        self.embed_ln = nn.LayerNorm(embed_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.predict_action = nn.Linear(embed_dim, action_dim)

    def forward(
        self,
        returns_to_go: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, context_length = returns_to_go.shape
        timesteps = timesteps.clamp(min=0, max=self.embed_timestep.num_embeddings - 1)
        time_embeddings = self.embed_timestep(timesteps)

        # 连续动作任务不能使用离散动作 embedding 表，因此这里改为
        # 线性层映射动作向量；但 token 的组织方式与离散 DT 保持一致。
        return_embeddings = self.embed_return(returns_to_go.unsqueeze(-1)) + time_embeddings
        state_embeddings = self.embed_state(states) + time_embeddings
        action_embeddings = self.embed_action(actions) + time_embeddings

        tokens = torch.stack(
            (return_embeddings, state_embeddings, action_embeddings),
            dim=2,
        ).reshape(batch_size, 3 * context_length, -1)
        tokens = self.embed_ln(tokens)

        causal_mask = torch.triu(
            torch.ones(
                3 * context_length,
                3 * context_length,
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        src_key_padding_mask = None
        if padding_mask is not None:
            src_key_padding_mask = ~padding_mask.repeat_interleave(3, dim=1)

        hidden = self.transformer(
            tokens,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
        )
        hidden = hidden.reshape(batch_size, context_length, 3, -1)
        state_hidden = hidden[:, :, 1, :]
        return self.predict_action(state_hidden)

    @torch.no_grad()
    def act(
        self,
        returns_to_go: list[float],
        states: list[np.ndarray],
        actions: list[np.ndarray],
        timesteps: list[int],
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        start = max(0, len(states) - self.context_length)
        actual_length = len(states) - start
        pad = self.context_length - actual_length

        state_array = np.zeros((self.context_length, self.state_dim), dtype=np.float32)
        action_array = np.zeros((self.context_length, self.action_dim), dtype=np.float32)
        rtg_array = np.zeros(self.context_length, dtype=np.float32)
        timestep_array = np.zeros(self.context_length, dtype=np.int64)
        mask_array = np.zeros(self.context_length, dtype=bool)

        state_array[pad:] = np.asarray(states[start:], dtype=np.float32)
        action_array[pad:] = np.asarray(actions[start:], dtype=np.float32)
        rtg_array[pad:] = np.asarray(returns_to_go[start:], dtype=np.float32)
        timestep_array[pad:] = np.asarray(timesteps[start:], dtype=np.int64)
        mask_array[pad:] = True

        rtg_tensor = torch.as_tensor(rtg_array, dtype=torch.float32, device=device).unsqueeze(0)
        state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=device).unsqueeze(0)
        action_tensor = torch.as_tensor(
            action_array,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        timestep_tensor = torch.as_tensor(
            timestep_array,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask_array, dtype=torch.bool, device=device).unsqueeze(0)
        predicted = self.forward(
            returns_to_go=rtg_tensor,
            states=state_tensor,
            actions=action_tensor,
            timesteps=timestep_tensor,
            padding_mask=mask_tensor,
        )
        action = predicted[0, -1].detach().cpu().numpy()
        return np.clip(action, action_low, action_high)
