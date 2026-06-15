from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.models import ContinuousBCPolicy, ContinuousDecisionTransformerPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Hopper D4RL reproduction outputs.")
    parser.add_argument("--runs-dir", type=Path, default=Path("outputs/hopper_medium"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hopper_medium/visualizations"),
    )
    parser.add_argument("--success-threshold", type=float, default=3000.0)
    parser.add_argument("--context-length", type=int, default=20)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.runs_dir / "summary.csv")
    plot_seed_returns(summary, args.output_dir, args.success_threshold)
    plot_mean_return(summary, args.output_dir)
    plot_return_ranges(summary, args.output_dir, args.success_threshold)
    plot_success_rates(summary, args.output_dir)
    plot_losses(args.runs_dir, args.output_dir)
    plot_replay_check(args.runs_dir, args.output_dir)

    weight_notes = plot_weight_summaries(args)
    write_visualization_notes(summary, args, weight_notes)


def plot_seed_returns(summary: pd.DataFrame, output_dir: Path, success_threshold: float) -> None:
    seeds = sorted(summary["seed"].unique())
    models = ["BC", "DT"]
    width = 0.36
    x = np.arange(len(seeds))

    plt.figure(figsize=(7.2, 4.2))
    for idx, model in enumerate(models):
        values = [
            float(summary[(summary["seed"] == seed) & (summary["model"] == model)][
                "return_mean"
            ].iloc[0])
            for seed in seeds
        ]
        plt.bar(x + (idx - 0.5) * width, values, width=width, label=model)
    plt.axhline(
        success_threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold={success_threshold:.0f}",
    )
    plt.xticks(x, [f"seed {seed}" for seed in seeds])
    plt.ylabel("Evaluation return")
    plt.title("Hopper Evaluation Return by Seed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hopper_seed_returns.png", dpi=200)
    plt.close()


def plot_mean_return(summary: pd.DataFrame, output_dir: Path) -> None:
    grouped = summary.groupby("model")["return_mean"].agg(["mean", "std"]).loc[["BC", "DT"]]
    plt.figure(figsize=(5.2, 4.2))
    plt.bar(
        grouped.index,
        grouped["mean"],
        yerr=grouped["std"],
        capsize=8,
        color=["#6C8EBF", "#D79B00"],
    )
    plt.ylabel("Return mean across seeds")
    plt.title("Hopper Cross-Seed Mean ± Std")
    for idx, (_model, row) in enumerate(grouped.iterrows()):
        plt.text(
            idx,
            row["mean"] + row["std"] + 80,
            f"{row['mean']:.0f}±{row['std']:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(output_dir / "hopper_mean_return_errorbar.png", dpi=200)
    plt.close()


def plot_return_ranges(summary: pd.DataFrame, output_dir: Path, success_threshold: float) -> None:
    rows = summary.sort_values(["seed", "model"]).reset_index(drop=True)
    labels = [f"s{int(row.seed)} {row.model}" for row in rows.itertuples()]
    y = np.arange(len(rows))
    means = rows["return_mean"].to_numpy(dtype=float)
    mins = rows["return_min"].to_numpy(dtype=float)
    maxs = rows["return_max"].to_numpy(dtype=float)
    colors = ["#6C8EBF" if model == "BC" else "#D79B00" for model in rows["model"]]

    plt.figure(figsize=(7.6, 4.8))
    plt.hlines(y, mins, maxs, color=colors, linewidth=4, alpha=0.65)
    plt.scatter(means, y, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    plt.axvline(
        success_threshold,
        color="#555555",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold={success_threshold:.0f}",
    )
    plt.yticks(y, labels)
    plt.xlabel("Episode return range over evaluation episodes")
    plt.title("Return Spread Explains Unstable Threshold Success")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hopper_return_ranges.png", dpi=200)
    plt.close()


def plot_success_rates(summary: pd.DataFrame, output_dir: Path) -> None:
    seeds = sorted(summary["seed"].unique())
    models = ["BC", "DT"]
    width = 0.36
    x = np.arange(len(seeds))

    plt.figure(figsize=(7.2, 4.2))
    for idx, model in enumerate(models):
        values = [
            float(summary[(summary["seed"] == seed) & (summary["model"] == model)][
                "success_rate"
            ].iloc[0])
            for seed in seeds
        ]
        plt.bar(x + (idx - 0.5) * width, values, width=width, label=model)
    plt.xticks(x, [f"seed {seed}" for seed in seeds])
    plt.ylabel("Rate of return >= 3000")
    plt.ylim(0, 1.05)
    plt.title("Threshold-Based Success Rate Is Highly Sensitive")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hopper_threshold_success.png", dpi=200)
    plt.close()


def plot_losses(runs_dir: Path, output_dir: Path) -> None:
    plt.figure(figsize=(7.4, 4.6))
    for losses_path in sorted(runs_dir.glob("seed*/losses.csv")):
        seed = losses_path.parent.name.replace("seed", "")
        losses = pd.read_csv(losses_path)
        for model, group in losses.groupby("model"):
            linestyle = "-" if model == "DT" else "--"
            plt.plot(
                group["epoch"],
                group["loss"],
                linestyle=linestyle,
                linewidth=1.5,
                label=f"{model} seed {seed}",
            )
    plt.xlabel("Epoch")
    plt.ylabel("Action MSE")
    plt.title("Training Loss Converges but Rollout Return Still Varies")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "hopper_training_losses_all_seeds.png", dpi=200)
    plt.close()


def plot_replay_check(runs_dir: Path, output_dir: Path) -> None:
    replay_path = runs_dir / "replay_check" / "replay_check.json"
    if not replay_path.exists():
        return
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    dataset_returns = np.asarray(data["dataset_returns"], dtype=float)
    replayed_returns = np.asarray(data["replayed_returns"], dtype=float)
    x = np.arange(len(dataset_returns))
    width = 0.36

    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - width / 2, dataset_returns, width=width, label="dataset return")
    plt.bar(x + width / 2, replayed_returns, width=width, label="replayed return")
    plt.xticks(x, [f"traj {idx}" for idx in x])
    plt.ylabel("Return")
    plt.title("Dataset-Environment Replay Mismatch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "hopper_replay_mismatch.png", dpi=200)
    plt.close()


def plot_weight_summaries(args: argparse.Namespace) -> list[str]:
    notes: list[str] = []
    seed0_dir = args.runs_dir / "seed0"
    dataset_summary_path = seed0_dir / "dataset_summary.json"
    if not dataset_summary_path.exists():
        return ["Skipped weight plots: dataset summary was not found."]
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    state_dim = int(dataset_summary["state_dim"])
    action_dim = int(dataset_summary["action_dim"])

    bc_path = seed0_dir / "continuous_bc_model.pt"
    dt_path = seed0_dir / "continuous_dt_model.pt"
    if not bc_path.exists() or not dt_path.exists():
        return [
            "Skipped weight plots: local Hopper .pt model files were not found. "
            "Copy continuous_bc_model.pt and continuous_dt_model.pt into "
            "outputs/hopper_medium/seed0/ and rerun this script to generate them."
        ]

    bc_state = torch.load(bc_path, map_location="cpu", weights_only=True)
    dt_state = torch.load(dt_path, map_location="cpu", weights_only=True)
    max_timestep = int(dt_state["embed_timestep.weight"].shape[0] - 1)

    bc = ContinuousBCPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dim=256)
    dt = ContinuousDecisionTransformerPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        context_length=args.context_length,
        embed_dim=args.embed_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_timestep=max_timestep,
    )
    bc.load_state_dict(bc_state)
    dt.load_state_dict(dt_state)

    plot_module_weight_norms(
        {
            "BC net.0": bc.net[0].weight,
            "BC net.2": bc.net[2].weight,
            "BC out": bc.net[4].weight,
            "DT return": dt.embed_return.weight,
            "DT state": dt.embed_state.weight,
            "DT action": dt.embed_action.weight,
            "DT out": dt.predict_action.weight,
        },
        args.output_dir,
    )
    notes.append("Generated weight plots from local seed0 model checkpoints.")
    return notes


def plot_module_weight_norms(weights: dict[str, torch.Tensor], output_dir: Path) -> None:
    names = list(weights.keys())
    norms = [float(weight.detach().float().norm().item()) for weight in weights.values()]

    plt.figure(figsize=(8, 4.4))
    plt.bar(names, norms, color="#82B366")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Frobenius norm")
    plt.title("Learned Weight Norms by Module")
    plt.tight_layout()
    plt.savefig(output_dir / "hopper_weight_norms_seed0.png", dpi=200)
    plt.close()


def write_visualization_notes(
    summary: pd.DataFrame,
    args: argparse.Namespace,
    weight_notes: list[str],
) -> None:
    grouped = summary.groupby("model")["return_mean"].agg(["mean", "std"]).loc[["BC", "DT"]]
    notes = [
        "# Hopper Visualization Notes",
        "",
        "These figures are generated from the local Hopper outputs.",
        "",
        "## Key Readings",
        "",
        (
            f"- BC return: {grouped.loc['BC', 'mean']:.1f} ± "
            f"{grouped.loc['BC', 'std']:.1f} across seeds."
        ),
        (
            f"- DT return: {grouped.loc['DT', 'mean']:.1f} ± "
            f"{grouped.loc['DT', 'std']:.1f} across seeds."
        ),
        (
            "- The success rate is threshold-based, not an environment-provided "
            f"success flag: it counts episodes with return >= {args.success_threshold:.0f}."
        ),
        "- The replay check shows dataset/environment mismatch and should be disclosed.",
        "",
        "## Generated Figures",
        "",
        "- `hopper_seed_returns.png`: per-seed BC/DT return comparison.",
        "- `hopper_mean_return_errorbar.png`: cross-seed mean with error bars.",
        "- `hopper_return_ranges.png`: min-to-max return ranges over evaluation episodes.",
        "- `hopper_threshold_success.png`: threshold-based success rate by seed.",
        "- `hopper_training_losses_all_seeds.png`: BC/DT action MSE over epochs.",
        "- `hopper_replay_mismatch.png`: dataset action replay mismatch.",
        "- `hopper_weight_norms_seed0.png`: generated only when local `.pt` weights exist.",
        "",
        "## Weight Plot Status",
        "",
        *[f"- {note}" for note in weight_notes],
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(notes), encoding="utf-8")


if __name__ == "__main__":
    main()
