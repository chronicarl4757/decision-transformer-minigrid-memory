from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

MODEL_KEYS = [
    ("bc", "BC"),
    ("return_conditioned_bc", "RCBC"),
    ("stacked_bc_k3", "SBC-K3"),
    ("stacked_bc_k5", "SBC-K5"),
    ("decision_transformer", "DT"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate epoch-sweep experiment outputs.")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_epoch_sweep(args.runs_dir, args.output_dir)


def aggregate_epoch_sweep(runs_dir: Path, output_dir: Path) -> None:
    rows = []
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        epoch = _epoch_from_name(run_dir.name)
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        for model_key, model_name in MODEL_KEYS:
            if model_key not in metrics:
                continue
            rows.append({"epoch": epoch, "model": model_name, **metrics[model_key]})

    df = pd.DataFrame(rows).sort_values(["model", "epoch"])
    df.to_csv(output_dir / "epoch_sweep_metrics.csv", index=False)
    _plot_dt_success(df, output_dir)
    _plot_model_returns(df, output_dir)


def _epoch_from_name(name: str) -> int:
    if name.startswith("epoch"):
        return int(name.removeprefix("epoch"))
    return 0


def _plot_dt_success(df: pd.DataFrame, output_dir: Path) -> None:
    dt = df[df["model"] == "DT"].sort_values("epoch")
    if dt.empty:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(dt["epoch"], dt["success_rate"], marker="o", color="#F58518")
    plt.xlabel("Training epochs")
    plt.ylabel("Success rate")
    plt.title("DT Success Rate by Training Epoch")
    plt.xticks(dt["epoch"])
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / "dt_success_by_epoch.png", dpi=180)
    plt.close()


def _plot_model_returns(df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    for model_name, group in df.groupby("model"):
        group = group.sort_values("epoch")
        plt.plot(group["epoch"], group["return_mean"], marker="o", label=model_name)
    plt.xlabel("Training epochs")
    plt.ylabel("Mean return")
    plt.title("Model Return by Training Epoch")
    plt.legend()
    plt.ylim(-0.05, max(1.0, float(df["return_mean"].max()) * 1.15))
    plt.tight_layout()
    plt.savefig(output_dir / "model_return_by_epoch.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
