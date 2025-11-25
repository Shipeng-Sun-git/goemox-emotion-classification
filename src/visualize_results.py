from pathlib import Path
from typing import Any

import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------
# Paths and basic helpers
# ----------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def extract_first_label(val: Any) -> int:

    # Already a list
    if isinstance(val, list):
        return int(val[0]) if val else 0

    # String representation
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return 0
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, list) and parsed:
                return int(parsed[0])
            return int(parsed)
        except Exception:
            # Fallback to simple int conversion
            try:
                return int(val)
            except Exception:
                return 0

    # Any other type
    try:
        return int(val)
    except Exception:
        return 0


# ----------------------------
# Visualization functions
# ----------------------------

def plot_top_label_support_and_accuracy(goe: pd.DataFrame) -> Path:
    """
    Plot support and accuracy for the most frequent labels in GoEmotions.
    Returns the path to the saved figure.
    """
    # Build gold labels and correctness flag
    goe["gold_label"] = goe["labels"].apply(extract_first_label)
    goe["correct"] = (goe["gold_label"] == goe["pred_label"]).astype(int)

    # Aggregate support and accuracy per label, pick top 6 by support
    stats = (
        goe.groupby("gold_label")
        .agg(support=("gold_label", "size"), accuracy=("correct", "mean"))
        .reset_index()
    )
    top_stats = stats.sort_values("support", ascending=False).head(6)

    labels = top_stats["gold_label"].astype(str).tolist()
    support = top_stats["support"].tolist()
    acc = top_stats["accuracy"].tolist()

    x = np.arange(len(labels))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, support, width, label="Support (count)")
    plt.bar(x + width / 2, acc, width, label="Accuracy")

    plt.xticks(x, labels)
    plt.xlabel("Label ID (gold_label)")
    plt.ylabel("Support / Accuracy")
    plt.title("Top GoEmotions Labels: Support and Accuracy\n(first-label, train subset)")
    plt.legend()
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "goemotions_top_labels_support_accuracy.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {out_path}")
    return out_path


def _label_distribution(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Compute the normalized predicted label distribution for a dataset.
    Returns a DataFrame with columns ['label', name].
    """
    counts = df["pred_label"].value_counts().sort_index()
    total = counts.sum()
    probs = counts / total if total > 0 else counts
    return pd.DataFrame({"label": probs.index, name: probs.values})


def plot_label_distribution_across_datasets(
    goe: pd.DataFrame, hn: pd.DataFrame, gh: pd.DataFrame
) -> Path:
    """
    Plot predicted label distribution across GoEmotions, HackerNews, and GitHub.
    We collapse labels into three groups: 27 (neutral), 0 (admiration), others.
    """
    dist_goe = _label_distribution(goe, "GoEmotions")
    dist_hn = _label_distribution(hn, "HackerNews")
    dist_gh = _label_distribution(gh, "GitHub")

    # Merge distributions
    dist = (
        dist_goe.merge(dist_hn, on="label", how="outer")
        .merge(dist_gh, on="label", how="outer")
        .fillna(0)
    )

    # Collapse into 27 / 0 / others
    rows = []
    for lab in [27, 0]:
        row = {"label_group": str(lab)}
        for col in dist.columns:
            if col == "label":
                continue
            row[col] = dist.loc[dist["label"] == lab, col].sum()
        rows.append(row)

    # "others" group
    row = {"label_group": "others"}
    for col in dist.columns:
        if col == "label":
            continue
        row[col] = dist.loc[~dist["label"].isin([0, 27]), col].sum()
    rows.append(row)

    collapsed = pd.DataFrame(rows)

    # Plot
    groups = collapsed["label_group"].tolist()
    datasets = ["GoEmotions", "HackerNews", "GitHub"]
    x = np.arange(len(groups))
    width = 0.2

    plt.figure()
    for i, ds in enumerate(datasets):
        plt.bar(x + (i - 1) * width, collapsed[ds].tolist(), width, label=ds)

    plt.xticks(x, groups)
    plt.xlabel("Predicted label group")
    plt.ylabel("Proportion of samples")
    plt.title("Predicted Label Distribution across Datasets")
    plt.legend()
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "pred_label_distribution_datasets.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {out_path}")
    return out_path


def plot_confusion_heatmap_top_labels(goe: pd.DataFrame, top_k: int = 10) -> Path:
    """
    Plot a confusion heatmap for the top-k most frequent labels in GoEmotions.
    Rows: gold labels; Columns: predicted labels. Rows are normalized to sum to 1.
    """
    goe["gold_label"] = goe["labels"].apply(extract_first_label)
    goe["pred_label"] = goe["pred_label"].astype(int)

    # Choose top-k labels by support
    label_counts = goe["gold_label"].value_counts().sort_values(ascending=False)
    top_labels = label_counts.head(top_k).index.tolist()
    n = len(top_labels)

    # Build confusion matrix for top-k labels
    label_to_idx = {lab: i for i, lab in enumerate(top_labels)}
    conf_mat = np.zeros((n, n), dtype=float)

    for _, row in goe.iterrows():
        g = row["gold_label"]
        p = row["pred_label"]
        if g in label_to_idx and p in label_to_idx:
            conf_mat[label_to_idx[g], label_to_idx[p]] += 1

    # Normalize each row to sum to 1 (avoid division by zero)
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    conf_norm = conf_mat / row_sums

    plt.figure()
    plt.imshow(conf_norm, aspect="auto")
    plt.colorbar(label="P(predicted | true)")
    plt.xticks(range(n), [str(l) for l in top_labels], rotation=45)
    plt.yticks(range(n), [str(l) for l in top_labels])
    plt.xlabel("Predicted label")
    plt.ylabel("True label (gold_label)")
    plt.title(f"GoEmotions Confusion Heatmap (Top {n} Labels)")
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "goemotions_confusion_top_labels_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {out_path}")
    return out_path


def plot_dataset_label_distribution_correlation(
    goe: pd.DataFrame, hn: pd.DataFrame, gh: pd.DataFrame
) -> Path:
    """
    Compute and plot a correlation heatmap between predicted label
    distributions of GoEmotions, HackerNews and GitHub.
    """

    def label_dist_vector(df: pd.DataFrame) -> np.ndarray:
        counts = df["pred_label"].value_counts().sort_index()
        # Ensure 28 positions (labels 0–27)
        full_index = pd.Index(range(28))
        counts = counts.reindex(full_index, fill_value=0)
        vec = counts.values.astype(float)
        total = vec.sum()
        if total > 0:
            vec /= total
        return vec

    vec_goe = label_dist_vector(goe)
    vec_hn = label_dist_vector(hn)
    vec_gh = label_dist_vector(gh)

    data = np.vstack([vec_goe, vec_hn, vec_gh])
    corr = np.corrcoef(data)

    labels = ["GoEmotions", "HackerNews", "GitHub"]

    plt.figure()
    plt.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(3), labels, rotation=45)
    plt.yticks(range(3), labels)
    plt.title("Correlation Heatmap of Predicted Label Distributions")
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "dataset_label_distribution_correlation_heatmap.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {out_path}")
    return out_path


def plot_confidence_histograms(goe: pd.DataFrame, hn: pd.DataFrame, gh: pd.DataFrame) -> Path:
    """
    Plot histograms of prediction confidence for the three datasets.
    This is useful for discussing calibration / over-confidence.
    """
    plt.figure()

    plt.hist(goe["pred_confidence"], bins=20, alpha=0.5, label="GoEmotions")
    plt.hist(hn["pred_confidence"], bins=20, alpha=0.5, label="HackerNews")
    plt.hist(gh["pred_confidence"], bins=20, alpha=0.5, label="GitHub")

    plt.xlabel("Prediction confidence (max softmax probability)")
    plt.ylabel("Count")
    plt.title("Prediction Confidence Histograms")
    plt.legend()
    plt.tight_layout()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIG_DIR / "prediction_confidence_histograms.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved figure: {out_path}")
    return out_path


# ----------------------------
# Main
# ----------------------------

def main():
    # Load CSVs
    pred_goe_path = RESULTS_DIR / "pred_goemotions_train_full.csv"
    pred_hn_path = RESULTS_DIR / "pred_hackernews_sample_full.csv"
    pred_gh_path = RESULTS_DIR / "pred_github_flask_issues_full.csv"

    if not pred_goe_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_goe_path}")
    if not pred_hn_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_hn_path}")
    if not pred_gh_path.exists():
        raise FileNotFoundError(f"Missing file: {pred_gh_path}")

    goe = pd.read_csv(pred_goe_path)
    hn = pd.read_csv(pred_hn_path)
    gh = pd.read_csv(pred_gh_path)

    # Ensure pred_label is integer where needed
    goe["pred_label"] = goe["pred_label"].astype(int)
    hn["pred_label"] = hn["pred_label"].astype(int)
    gh["pred_label"] = gh["pred_label"].astype(int)

    # Generate all figures
    plot_top_label_support_and_accuracy(goe.copy())
    plot_label_distribution_across_datasets(goe.copy(), hn.copy(), gh.copy())
    plot_confusion_heatmap_top_labels(goe.copy(), top_k=10)
    plot_dataset_label_distribution_correlation(goe.copy(), hn.copy(), gh.copy())
    plot_confidence_histograms(goe.copy(), hn.copy(), gh.copy())

    print("[INFO] All visualizations generated in results/figures/.")


if __name__ == "__main__":
    main()
