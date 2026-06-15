from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.antmaze import (
    ANTMAZE_DATASET_CANDIDATES,
    ContinuousBCDataset,
    ContinuousDecisionTransformerDataset,
    flatten_antmaze_observation,
    load_antmaze_dataset,
)
from src.models import ContinuousBCPolicy, ContinuousDecisionTransformerPolicy
from src.progress import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train continuous-action DT on AntMaze Umaze.")
    parser.add_argument("--dataset-id", action="append", default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--target-return", type=float, default=1.0)
    parser.add_argument("--rtg-scale", type=float, default=1.0)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/antmaze_umaze/seed0"))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-normalize-states", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.quick:
        args.max_episodes = 20
        args.epochs = 2
        args.batch_size = 64
        args.num_workers = 0
        args.eval_episodes = 2
        args.embed_dim = 64
        args.n_layers = 1
        args.n_heads = 2
    run_experiment(args)


def run_experiment(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    bundle = load_antmaze_dataset(
        dataset_ids=tuple(args.dataset_id) if args.dataset_id else ANTMAZE_DATASET_CANDIDATES,
        max_episodes=args.max_episodes,
        normalize_states=not args.no_normalize_states,
    )
    _write_dataset_summary(bundle, args.output_dir)

    bc_model, bc_losses = train_bc(args, bundle, device)
    torch.save(bc_model.state_dict(), args.output_dir / "continuous_bc_model.pt")
    dt_model, dt_losses = train_dt(args, bundle, device)
    torch.save(dt_model.state_dict(), args.output_dir / "continuous_dt_model.pt")

    bc_metrics = evaluate_bc(args, bundle, bc_model, device)
    dt_metrics = evaluate_dt(args, bundle, dt_model, device)
    metrics = {
        "dataset_id": bundle.dataset_id,
        "continuous_bc": bc_metrics,
        "continuous_decision_transformer": dt_metrics,
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
                {"model": "DT", "epoch": idx + 1, "loss": loss}
                for idx, loss in enumerate(dt_losses)
            ),
        ]
    )
    losses.to_csv(args.output_dir / "losses.csv", index=False)
    _plot_losses(losses, args.output_dir)
    pd.DataFrame(
        [
            {"model": "BC", **bc_metrics},
            {"model": "DT", **dt_metrics},
        ]
    ).to_csv(args.output_dir / "evaluation_summary.csv", index=False)
    return metrics


def train_bc(
    args: argparse.Namespace,
    bundle,
    device: torch.device,
) -> tuple[ContinuousBCPolicy, list[float]]:
    dataset = ContinuousBCDataset(bundle.trajectories)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = ContinuousBCPolicy(
        state_dim=bundle.state_dim,
        action_dim=bundle.action_dim,
        hidden_dim=256,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    losses: list[float] = []

    for _ in tqdm(range(args.epochs), desc="Training AntMaze BC"):
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


def train_dt(
    args: argparse.Namespace,
    bundle,
    device: torch.device,
) -> tuple[ContinuousDecisionTransformerPolicy, list[float]]:
    dataset = ContinuousDecisionTransformerDataset(
        trajectories=bundle.trajectories,
        context_length=args.context_length,
        state_dim=bundle.state_dim,
        action_dim=bundle.action_dim,
        rtg_scale=args.rtg_scale,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = ContinuousDecisionTransformerPolicy(
        state_dim=bundle.state_dim,
        action_dim=bundle.action_dim,
        context_length=args.context_length,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_timestep=max(1000, max(len(t.actions) for t in bundle.trajectories) + 1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction="none")
    losses: list[float] = []

    for _ in tqdm(range(args.epochs), desc="Training AntMaze DT"):
        model.train()
        epoch_losses: list[float] = []
        for batch in loader:
            returns_to_go = batch["returns_to_go"].to(device)
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["mask"].to(device)
            predicted = model(
                returns_to_go=returns_to_go,
                states=states,
                actions=actions,
                timesteps=timesteps,
                padding_mask=mask,
            )
            raw_loss = criterion(predicted, actions).mean(dim=-1)
            loss = (raw_loss * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)))
    return model, losses


@torch.no_grad()
def evaluate_bc(
    args: argparse.Namespace,
    bundle,
    model: ContinuousBCPolicy,
    device: torch.device,
) -> dict[str, float]:
    env = _recover_env(bundle.dataset_id)
    returns, lengths, successes = [], [], []
    model.eval()
    for episode in range(args.eval_episodes):
        observation, _ = env.reset(seed=args.seed + 40_000 + episode)
        state = _normalize_state(flatten_antmaze_observation(observation), bundle)
        total = 0.0
        terminated = truncated = False
        length = 0
        info = {}
        while not (terminated or truncated):
            action = model.act(state, bundle.action_low, bundle.action_high, device)
            observation, reward, terminated, truncated, info = env.step(action)
            state = _normalize_state(flatten_antmaze_observation(observation), bundle)
            total += float(reward)
            length += 1
        returns.append(total)
        lengths.append(length)
        successes.append(_success_from_episode(total, info, args.success_threshold))
    env.close()
    return _continuous_return_stats(returns, lengths, successes)


@torch.no_grad()
def evaluate_dt(
    args: argparse.Namespace,
    bundle,
    model: ContinuousDecisionTransformerPolicy,
    device: torch.device,
) -> dict[str, float]:
    env = _recover_env(bundle.dataset_id)
    returns, lengths, successes = [], [], []
    model.eval()
    for episode in range(args.eval_episodes):
        observation, _ = env.reset(seed=args.seed + 50_000 + episode)
        state = _normalize_state(flatten_antmaze_observation(observation), bundle)
        states: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        returns_to_go: list[float] = []
        timesteps: list[int] = []
        desired_return = float(args.target_return)
        total = 0.0
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            states.append(state)
            returns_to_go.append(desired_return / args.rtg_scale)
            timesteps.append(len(timesteps))
            action_context = actions + [np.zeros(bundle.action_dim, dtype=np.float32)]
            action = model.act(
                returns_to_go=returns_to_go,
                states=states,
                actions=action_context,
                timesteps=timesteps,
                action_low=bundle.action_low,
                action_high=bundle.action_high,
                device=device,
            )
            observation, reward, terminated, truncated, info = env.step(action)
            actions.append(action.astype(np.float32))
            state = _normalize_state(flatten_antmaze_observation(observation), bundle)
            total += float(reward)
            desired_return = max(desired_return - float(reward), 0.0)
        returns.append(total)
        lengths.append(len(actions))
        successes.append(_success_from_episode(total, info, args.success_threshold))
    env.close()
    return _continuous_return_stats(returns, lengths, successes)


def _recover_env(dataset_id: str):
    import gymnasium as gym
    import gymnasium_robotics  # noqa: F401  # registers AntMaze environments

    # Use dense reward variant to match the dataset's reward scale (the data was
    # collected with dense rewards; the default recover_environment() returns a
    # sparse-reward env that gives +1 only at the goal).
    env_name = _dense_env_name(dataset_id)
    return gym.make(env_name)


def _dense_env_name(dataset_id: str) -> str:
    # Map dataset ID like "D4RL/antmaze/umaze-v1" to dense env like "AntMaze_UMazeDense-v4"
    parts = dataset_id.split("/")
    env_part = parts[-1] if len(parts) == 3 else parts[0]
    name_version = env_part.split("-")
    name = name_version[0]
    version_num = name_version[1] if len(name_version) > 1 else "v1"
    # Convert short name to gymnasium name: umaze -> UMaze
    env_base = name.replace("umaze", "UMaze")
    return f"AntMaze_{env_base}Dense-v4"


def _normalize_state(state: np.ndarray, bundle) -> np.ndarray:
    return ((state - bundle.state_mean) / bundle.state_std).astype(np.float32)


def _success_from_episode(total_return: float, info: dict, success_threshold: float) -> float:
    if "success" in info:
        return float(bool(info["success"]))
    if "is_success" in info:
        return float(bool(info["is_success"]))
    return float(total_return >= success_threshold)


def _continuous_return_stats(
    returns: list[float],
    lengths: list[int],
    successes: list[float],
) -> dict[str, float]:
    values = np.asarray(returns, dtype=np.float32)
    return {
        "return_mean": float(values.mean()),
        "return_std": float(values.std()),
        "return_min": float(values.min()),
        "return_max": float(values.max()),
        "success_rate": float(np.mean(successes)),
        "episode_length_mean": float(np.mean(lengths)),
    }


def _write_dataset_summary(bundle, output_dir: Path) -> None:
    returns = np.asarray(
        [trajectory.total_return for trajectory in bundle.trajectories],
        dtype=np.float32,
    )
    lengths = np.asarray(
        [len(trajectory.actions) for trajectory in bundle.trajectories],
        dtype=np.float32,
    )
    summary = {
        "dataset_id": bundle.dataset_id,
        "episodes": float(len(bundle.trajectories)),
        "transitions": float(lengths.sum()),
        "state_dim": float(bundle.state_dim),
        "action_dim": float(bundle.action_dim),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "length_mean": float(lengths.mean()),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _plot_losses(losses: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    for model_name, group in losses.groupby("model"):
        plt.plot(group["epoch"], group["loss"], marker="o", label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("AntMaze Continuous Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
