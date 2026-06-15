from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.plots import plot_ablation_summary
from src.train import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run compact Decision Transformer ablations.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ablations"))
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str | int]] = []

    base = {
        "env": "memory",
        "episodes": 400 if not args.quick else 60,
        "epochs": 40 if not args.quick else 3,
        "batch_size": 128,
        "seed": 0,
        "eval_episodes": 100 if not args.quick else 3,
        "embed_dim": 96 if not args.quick else 32,
        "n_heads": 4 if not args.quick else 2,
        "lr": 1e-3,
        "target_return": 0.95,
        "quick": False,
    }

    settings = [
        ("context", {"context_length": 5, "n_layers": 1}),
        ("context", {"context_length": 10, "n_layers": 1}),
        ("context", {"context_length": 20, "n_layers": 1}),
        ("target_return", {"context_length": 20, "n_layers": 1, "target_return": 0.0}),
        ("target_return", {"context_length": 20, "n_layers": 1, "target_return": 0.3}),
        ("target_return", {"context_length": 20, "n_layers": 1, "target_return": 0.95}),
        ("model_size", {"context_length": 10, "n_layers": 1}),
        ("model_size", {"context_length": 10, "n_layers": 2}),
        ("model_size", {"context_length": 10, "n_layers": 3}),
    ]

    for index, (kind, overrides) in enumerate(settings):
        config = {**base, **overrides}
        config["output_dir"] = args.output_dir / f"{index:02d}_{kind}"
        metrics = run_experiment(argparse.Namespace(**config))
        rows.append(
            {
                "kind": kind,
                "context_length": config["context_length"],
                "target_return": config["target_return"],
                "n_layers": config["n_layers"],
                "bc_return_mean": metrics["bc"]["return_mean"],
                "dt_return_mean": metrics["decision_transformer"]["return_mean"],
                "dt_success_rate": metrics["decision_transformer"]["success_rate"],
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_ablation_summary(args.output_dir)


if __name__ == "__main__":
    main()
