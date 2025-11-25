"""
Analyze in-domain performance on GoEmotions.

This script expects:
- data/goemotions_train_full.csv           (original data)
- results/pred_goemotions_train_full.csv   (with model predictions)

It will:
- reconstruct a single gold label per sample (same as in train.py)
- compute overall accuracy
- compute per-label support and accuracy
- save a summary table to results/label_stats_goemotions.csv
"""

from pathlib import Path
from typing import Any

import ast
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

ORIG_FILE = DATA_DIR / "goemotions_train_full.csv"
PRED_FILE = RESULTS_DIR / "pred_goemotions_train_full.csv"
OUT_FILE = RESULTS_DIR / "label_stats_goemotions.csv"


def extract_first_label(x: Any) -> int:
    """
    Take the first label from the 'labels' cell.

    The value can be:
    - a list of ints
    - a string like "[3, 5]"
    - a single int (int or string)
    """
    # Already a list
    if isinstance(x, list) and len(x) > 0:
        return int(x[0])

    # String representation
    if isinstance(x, str):
        x = x.strip()
        if not x:
            return 0
        try:
            value = ast.literal_eval(x)
            if isinstance(value, list) and len(value) > 0:
                return int(value[0])
            return int(value)
        except Exception:
            # Try direct int conversion
            try:
                return int(x)
            except Exception:
                return 0

    # Fallback
    try:
        return int(x)
    except Exception:
        return 0


def main():
    if not ORIG_FILE.exists():
        raise FileNotFoundError(f"Original file not found: {ORIG_FILE}")
    if not PRED_FILE.exists():
        raise FileNotFoundError(f"Prediction file not found: {PRED_FILE}")

    print(f"[INFO] Loading original data from: {ORIG_FILE}")
    df_orig = pd.read_csv(ORIG_FILE)

    print(f"[INFO] Loading predictions from: {PRED_FILE}")
    df_pred = pd.read_csv(PRED_FILE)

    # We assume both files have the same order of rows.
    # If you want to be safe, you could merge on an 'id' column instead.
    assert len(df_orig) == len(df_pred), "Row counts do not match!"

    # Rebuild gold label from 'labels' column (same logic as in train.py)
    if "labels" not in df_orig.columns:
        raise ValueError("Expected a 'labels' column in the original file.")
    df_orig["gold_label"] = df_orig["labels"].apply(extract_first_label)

    if "pred_label" not in df_pred.columns:
        raise ValueError("Expected a 'pred_label' column in the prediction file.")
    df_orig["pred_label"] = df_pred["pred_label"].astype(int)

    # Overall accuracy
    correct = (df_orig["gold_label"] == df_orig["pred_label"]).sum()
    total = len(df_orig)
    overall_acc = correct / total
    print(f"[RESULT] Overall training accuracy (first-label): {overall_acc:.3f}")

    # Per-label support and accuracy
    rows = []
    for label_id, group in df_orig.groupby("gold_label"):
        support = len(group)
        acc = (group["gold_label"] == group["pred_label"]).mean()
        rows.append({"label_id": int(label_id), "support": int(support), "accuracy": float(acc)})

    df_stats = pd.DataFrame(rows).sort_values(by="support", ascending=False)

    # Print a few lines to the console
    print("[INFO] Per-label support and accuracy (top 10 by support):")
    print(df_stats.head(10))

    # Save to CSV for report
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_stats.to_csv(OUT_FILE, index=False)
    print(f"[INFO] Saved full label stats to: {OUT_FILE}")


if __name__ == "__main__":
    main()
