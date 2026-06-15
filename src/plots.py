from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    losses_path = output_dir / "losses.csv"
    metrics_path = output_dir / "metrics.json"

    if losses_path.exists():
        losses = pd.read_csv(losses_path)
        plt.figure(figsize=(7, 4))
        for model_name, group in losses.groupby("model"):
            plt.plot(group["epoch"], group["loss"], marker="o", label=model_name)
        plt.xlabel("Epoch")
        plt.ylabel("Training loss")
        plt.title("Model Training Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss.png", dpi=180)
        plt.close()

    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        names = [
            name
            for name in (
                "bc",
                "return_conditioned_bc",
                "stacked_bc_k3",
                "stacked_bc_k5",
                "decision_transformer",
            )
            if name in metrics
        ]
        means = [metrics[name]["return_mean"] for name in names]
        y_max = max(1.0, max(means) * 1.2)
        plt.figure(figsize=(6, 4))
        label_map = {
            "bc": "BC",
            "return_conditioned_bc": "RCBC",
            "stacked_bc_k3": "SBC-K3",
            "stacked_bc_k5": "SBC-K5",
            "decision_transformer": "DT",
        }
        color_map = {
            "bc": "#4C78A8",
            "return_conditioned_bc": "#54A24B",
            "stacked_bc_k3": "#72B7B2",
            "stacked_bc_k5": "#B279A2",
            "decision_transformer": "#F58518",
        }
        labels = [label_map[name] for name in names]
        colors = [color_map[name] for name in names]
        plt.bar(labels, means, color=colors)
        plt.ylabel("Mean episode return")
        plt.title("Evaluation Return")
        plt.ylim(0, y_max)
        plt.tight_layout()
        plt.savefig(output_dir / "evaluation_return.png", dpi=180)
        plt.close()


def plot_ablation_summary(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    summary_path = output_dir / "summary.csv"
    if not summary_path.exists():
        return

    summary = pd.read_csv(summary_path)
    for kind, group in summary.groupby("kind"):
        if kind == "context":
            labels = [f"K={value}" for value in group["context_length"]]
        elif kind == "target_return":
            labels = [f"R={int(value)}" for value in group["target_return"]]
        else:
            labels = [f"L={value}" for value in group["n_layers"]]

        plt.figure(figsize=(7, 4))
        plt.bar(labels, group["dt_return_mean"], color="#54A24B")
        plt.ylabel("DT mean episode return")
        plt.title(f"Ablation: {kind}")
        plt.ylim(0, 500)
        plt.tight_layout()
        plt.savefig(output_dir / f"ablation_{kind}.png", dpi=180)
        plt.close()
