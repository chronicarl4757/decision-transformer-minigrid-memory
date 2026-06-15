"""课程版 DT 复现实验的统一训练入口。"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import (
    BCDataset,
    DecisionTransformerDataset,
    ReturnConditionedBCDataset,
    StackedBCDataset,
    collect_memory_trajectories,
    summarize_trajectories,
)
from src.evaluate import (
    evaluate_bc,
    evaluate_dt,
    evaluate_return_conditioned_bc,
    evaluate_stacked_bc,
)
from src.models import (
    BehaviorCloningPolicy,
    DecisionTransformerPolicy,
    ReturnConditionedBCPolicy,
    StackedBCPolicy,
)
from src.plots import plot_results
from src.progress import tqdm

ENV_CONFIGS = {
    "memory": {
        "state_dim": 11,
        "action_dim": 7,
        "max_timestep": 605,
        "target_return": 0.95,
        "rtg_scale": 1.0,
        "success_threshold": 0.80,
    },
}

STACKED_BC_CONTEXTS = (3, 5)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，统一管理不同环境下的训练配置。"""
    parser = argparse.ArgumentParser(description="Train BC and Decision Transformer offline.")
    parser.add_argument("--env", choices=sorted(ENV_CONFIGS), default="memory")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--context-length", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--target-return", type=float, default=None)
    parser.add_argument("--eval-target-returns", type=float, nargs="*", default=None)
    parser.add_argument("--rtg-scale", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default"))
    parser.add_argument("--quick", action="store_true", help="Use a tiny run for smoke testing.")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""
    args = parse_args()
    if args.quick:
        args.episodes = 60
        args.epochs = 3
        args.batch_size = 64
        args.eval_episodes = 3
        args.embed_dim = 32
        args.n_layers = 1
        args.n_heads = 2

    run_experiment(args)


def run_experiment(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """执行一次完整实验：采集数据、训练模型、评估并导出结果。"""
    set_seed(args.seed)
    env_config = ENV_CONFIGS[args.env]
    args.state_dim = int(env_config["state_dim"])
    args.action_dim = int(env_config["action_dim"])
    args.max_timestep = int(env_config["max_timestep"])
    args.success_threshold = float(env_config["success_threshold"])
    if args.target_return is None:
        args.target_return = float(env_config["target_return"])
    if args.rtg_scale is None:
        args.rtg_scale = float(env_config["rtg_scale"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    trajectories = collect_trajectories(args.env, num_episodes=args.episodes, seed=args.seed)
    dataset_summary = summarize_trajectories(trajectories)
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(dataset_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    bc_model, bc_losses = train_bc(args, trajectories, device)
    torch.save(bc_model.state_dict(), args.output_dir / "bc_model.pt")
    rcbc_model, rcbc_losses = train_return_conditioned_bc(args, trajectories, device)
    torch.save(rcbc_model.state_dict(), args.output_dir / "rcbc_model.pt")
    stacked_models: dict[int, StackedBCPolicy] = {}
    stacked_losses: dict[int, list[float]] = {}
    # 两个 stacked BC 基线共享同一套训练流程，只是历史窗口长度不同。
    for stacked_context in STACKED_BC_CONTEXTS:
        stacked_model, losses = train_stacked_bc(args, trajectories, device, stacked_context)
        stacked_models[stacked_context] = stacked_model
        stacked_losses[stacked_context] = losses
        model_path = args.output_dir / f"stacked_bc_k{stacked_context}_model.pt"
        torch.save(stacked_model.state_dict(), model_path)
    dt_model, dt_losses = train_dt(args, trajectories, device)
    torch.save(dt_model.state_dict(), args.output_dir / "dt_model.pt")

    bc_metrics = evaluate_bc(
        model=bc_model,
        episodes=args.eval_episodes,
        seed=args.seed + 10_000,
        device=device,
        env_name=args.env,
        success_threshold=args.success_threshold,
    )
    dt_metrics = evaluate_dt(
        model=dt_model,
        episodes=args.eval_episodes,
        seed=args.seed + 20_000,
        device=device,
        target_return=args.target_return,
        rtg_scale=args.rtg_scale,
        env_name=args.env,
        success_threshold=args.success_threshold,
    )
    rcbc_metrics = evaluate_return_conditioned_bc(
        model=rcbc_model,
        episodes=args.eval_episodes,
        seed=args.seed + 15_000,
        device=device,
        target_return=args.target_return,
        rtg_scale=args.rtg_scale,
        env_name=args.env,
        success_threshold=args.success_threshold,
    )
    stacked_metrics = {
        context: evaluate_stacked_bc(
            model=model,
            episodes=args.eval_episodes,
            seed=args.seed + 16_000 + context,
            device=device,
            env_name=args.env,
            success_threshold=args.success_threshold,
        )
        for context, model in stacked_models.items()
    }

    metrics = {
        "bc": bc_metrics,
        "return_conditioned_bc": rcbc_metrics,
        **{f"stacked_bc_k{context}": value for context, value in stacked_metrics.items()},
        "decision_transformer": dt_metrics,
    }
    if args.eval_target_returns:
        target_rows = []
        for eval_target in args.eval_target_returns:
            # target sweep 用同一个训练好的 DT，在不同目标回报条件下
            # 重复 rollout，观察条件化是否真正生效。
            target_metrics = evaluate_dt(
                model=dt_model,
                episodes=args.eval_episodes,
                seed=args.seed + 30_000,
                device=device,
                target_return=float(eval_target),
                rtg_scale=args.rtg_scale,
                env_name=args.env,
                success_threshold=args.success_threshold,
            )
            target_rows.append({"target_return": float(eval_target), **target_metrics})
        pd.DataFrame(target_rows).to_csv(args.output_dir / "target_sweep.csv", index=False)
        metrics["target_sweep"] = {
            str(row["target_return"]): {
                key: value for key, value in row.items() if key != "target_return"
            }
            for row in target_rows
        }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    losses = pd.DataFrame(
        [
            *(
                {"model": "BC", "epoch": idx + 1, "loss": loss}
                for idx, loss in enumerate(bc_losses)
            ),
            *(
                {"model": "RCBC", "epoch": idx + 1, "loss": loss}
                for idx, loss in enumerate(rcbc_losses)
            ),
            *(
                {"model": f"SBC-K{context}", "epoch": idx + 1, "loss": loss}
                for context, losses_for_context in stacked_losses.items()
                for idx, loss in enumerate(losses_for_context)
            ),
            *(
                {"model": "DT", "epoch": idx + 1, "loss": loss}
                for idx, loss in enumerate(dt_losses)
            ),
        ]
    )
    losses.to_csv(args.output_dir / "losses.csv", index=False)
    plot_results(args.output_dir)
    return metrics


def collect_trajectories(env_name: str, num_episodes: int, seed: int):
    """根据环境名分派对应的离线数据采集函数。"""
    if env_name == "memory":
        return collect_memory_trajectories(num_episodes=num_episodes, seed=seed)
    raise ValueError(f"Unsupported env: {env_name}")


def train_bc(
    args: argparse.Namespace,
    trajectories,
    device: torch.device,
) -> tuple[BehaviorCloningPolicy, list[float]]:
    """训练普通行为克隆基线。"""
    dataset = BCDataset(trajectories)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = BehaviorCloningPolicy(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=128,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for _ in tqdm(range(args.epochs), desc="Training BC"):
        model.train()
        epoch_losses: list[float] = []
        for states, actions in loader:
            states = states.to(device)
            actions = actions.to(device)
            loss = criterion(model(states), actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses


def train_return_conditioned_bc(
    args: argparse.Namespace,
    trajectories,
    device: torch.device,
) -> tuple[ReturnConditionedBCPolicy, list[float]]:
    """训练带回报条件的前馈基线 RCBC。"""
    dataset = ReturnConditionedBCDataset(trajectories, rtg_scale=args.rtg_scale)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = ReturnConditionedBCPolicy(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        hidden_dim=128,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for _ in tqdm(range(args.epochs), desc="Training RCBC"):
        model.train()
        epoch_losses: list[float] = []
        for states, returns_to_go, actions in loader:
            states = states.to(device)
            returns_to_go = returns_to_go.to(device)
            actions = actions.to(device)
            loss = criterion(model(states, returns_to_go), actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses


def train_stacked_bc(
    args: argparse.Namespace,
    trajectories,
    device: torch.device,
    context_length: int,
) -> tuple[StackedBCPolicy, list[float]]:
    """训练短历史状态窗口基线 SBC。"""
    dataset = StackedBCDataset(
        trajectories=trajectories,
        context_length=context_length,
        state_dim=args.state_dim,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = StackedBCPolicy(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        context_length=context_length,
        hidden_dim=128,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for _ in tqdm(range(args.epochs), desc=f"Training SBC-K{context_length}"):
        model.train()
        epoch_losses: list[float] = []
        for stacked_states, actions in loader:
            stacked_states = stacked_states.to(device)
            actions = actions.to(device)
            loss = criterion(model(stacked_states), actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses


def train_dt(
    args: argparse.Namespace,
    trajectories,
    device: torch.device,
) -> tuple[DecisionTransformerPolicy, list[float]]:
    """训练离散动作版本的 Decision Transformer。"""
    dataset = DecisionTransformerDataset(
        trajectories=trajectories,
        context_length=args.context_length,
        state_dim=args.state_dim,
        rtg_scale=args.rtg_scale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DecisionTransformerPolicy(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        context_length=args.context_length,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_timestep=args.max_timestep,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")
    losses: list[float] = []

    for _ in tqdm(range(args.epochs), desc="Training DT"):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            returns_to_go = batch["returns_to_go"].to(device)
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["mask"].to(device)

            logits = model(
                returns_to_go=returns_to_go,
                states=states,
                actions=actions,
                timesteps=timesteps,
                padding_mask=mask,
            )
            raw_loss = criterion(
                logits.reshape(-1, args.action_dim),
                actions.reshape(-1),
            ).reshape_as(actions)
            # 只有真实时间步参与损失；左侧补零位置对应的 token 不计入训练。
            loss = (raw_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            # 课程规模实验里梯度裁剪能显著减少 DT 训练初期的不稳定。
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
