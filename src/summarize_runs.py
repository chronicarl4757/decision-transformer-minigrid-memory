from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_ORDER = ["BC", "RCBC", "SBC-K3", "SBC-K5", "DT"]
MODEL_COLORS = {
    "BC": "#4C78A8",
    "RCBC": "#54A24B",
    "SBC-K3": "#72B7B2",
    "SBC-K5": "#B279A2",
    "DT": "#F58518",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed experiment outputs.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_runs(args.runs_dir, args.output_dir)


def aggregate_runs(runs_dir: Path, output_dir: Path) -> None:
    metric_rows = []
    target_rows = []

    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        seed = _seed_from_name(run_dir.name)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for model_key, model_name in [
            ("bc", "BC"),
            ("return_conditioned_bc", "RCBC"),
            ("stacked_bc_k3", "SBC-K3"),
            ("stacked_bc_k5", "SBC-K5"),
            ("decision_transformer", "DT"),
        ]:
            if model_key not in metrics:
                continue
            row = {"seed": seed, "model": model_name, **metrics[model_key]}
            metric_rows.append(row)

        sweep_path = run_dir / "target_sweep.csv"
        if sweep_path.exists():
            sweep = pd.read_csv(sweep_path)
            sweep.insert(0, "seed", seed)
            target_rows.extend(sweep.to_dict("records"))

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "main_metrics_by_seed.csv", index=False)
    summary = (
        metrics_df.groupby("model")
        .agg(
            return_mean=("return_mean", "mean"),
            return_std_across_seeds=("return_mean", "std"),
            success_rate_mean=("success_rate", "mean"),
            far_goal_rate_mean=("far_goal_rate", "mean"),
        )
        .reset_index()
    )
    summary["model"] = pd.Categorical(summary["model"], categories=MODEL_ORDER, ordered=True)
    summary = summary.sort_values("model")
    summary.to_csv(output_dir / "main_metrics_summary.csv", index=False)
    _plot_main_summary(summary, output_dir)

    if target_rows:
        target_df = pd.DataFrame(target_rows)
        target_df.to_csv(output_dir / "target_sweep_by_seed.csv", index=False)
        target_summary = (
            target_df.groupby("target_return")
            .agg(
                return_mean=("return_mean", "mean"),
                return_std_across_seeds=("return_mean", "std"),
                success_rate_mean=("success_rate", "mean"),
                far_goal_rate_mean=("far_goal_rate", "mean"),
            )
            .reset_index()
        )
        target_summary = target_summary.sort_values("target_return")
        target_summary.to_csv(output_dir / "target_sweep_summary.csv", index=False)
        _plot_target_summary(target_summary, output_dir)


def _seed_from_name(name: str) -> int:
    if name.startswith("seed"):
        return int(name.removeprefix("seed"))
    return 0


def _plot_main_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    y_max = max(1.0, float(summary["return_mean"].max()) * 1.2)
    plt.figure(figsize=(7, 4))
    plt.bar(
        summary["model"].astype(str),
        summary["return_mean"],
        yerr=summary["return_std_across_seeds"].fillna(0.0),
        capsize=5,
        color=[MODEL_COLORS[str(model)] for model in summary["model"]],
    )
    plt.ylabel("Mean return")
    plt.title("Model Comparison on MiniGrid Memory")
    plt.ylim(0, y_max)
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close()


def _plot_target_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    y_max = max(1.0, float(summary["return_mean"].max()) * 1.2)
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        summary["target_return"],
        summary["return_mean"],
        yerr=summary["return_std_across_seeds"].fillna(0.0),
        marker="o",
        capsize=5,
        color="#1B9E77",
    )
    plt.xlabel("Target return")
    plt.ylabel("Actual mean return")
    plt.title("Target Return Sweep")
    plt.xticks(summary["target_return"])
    plt.ylim(-0.1, y_max)
    plt.tight_layout()
    plt.savefig(output_dir / "target_sweep.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
