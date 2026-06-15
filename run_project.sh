#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
MODE="${1:---quick}"

echo "[1/4] Python: $($PYTHON_BIN -V)"
echo "[2/4] Running tests"
"$PYTHON_BIN" -m pytest

echo "[3/4] Running lint"
"$PYTHON_BIN" -m ruff check .

if [[ "$MODE" == "--full" ]]; then
  echo "[4/4] Re-running the 3-seed final experiment"
  rm -rf outputs/memory_final_multiseed outputs/memory_final_summary
  for seed in 0 1 2; do
    "$PYTHON_BIN" -m src.train \
      --env memory \
      --episodes 400 \
      --epochs 40 \
      --batch-size 128 \
      --context-length 20 \
      --n-layers 1 \
      --embed-dim 96 \
      --n-heads 4 \
      --lr 1e-3 \
      --eval-episodes 100 \
      --seed "$seed" \
      --target-return 0.95 \
      --eval-target-returns 0 0.3 0.95 \
      --output-dir "outputs/memory_final_multiseed/seed${seed}"
  done
  "$PYTHON_BIN" -m src.summarize_runs \
    --runs-dir outputs/memory_final_multiseed \
    --output-dir outputs/memory_final_summary
else
  echo "[4/4] Running quick smoke experiment"
  rm -rf outputs/quick_smoke
  "$PYTHON_BIN" -m src.train \
    --env memory \
    --quick \
    --context-length 20 \
    --target-return 0.95 \
    --eval-target-returns 0 0.3 0.95 \
    --output-dir outputs/quick_smoke
fi

echo
echo "Done."
echo "Final report: dist/final_submission_report.pdf"
echo "Literature review: dist/literature_review.pdf"
echo "Final metrics: outputs/memory_final_summary/main_metrics_summary.csv"
