import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import ticker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["savefig.pad_inches"] = 0.15

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
COL_NAMES = {
    "em": "EM",
    "f1": "F1",
    "latency_ms": "latency_ms",
    "recall@5": "recall@5",
    "recall@10": "recall@10",
}


def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    numeric_cols = ["em", "f1", "latency_ms", "recall@5", "recall@10"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for old, new in COL_NAMES.items():
        if old in df.columns and old != new:
            df.rename(columns={old: new}, inplace=True)
    str_cols = ["dataset", "question", "model", "distractor_type", "prediction"]
    for c in str_cols:
        if c not in df.columns:
            df[c] = ""
    return df


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "distractor_type", "distractor_count"]
    summary = (
        df.groupby(group_cols, observed=True)
        .agg(
            mean_EM=("EM", "mean"),
            mean_F1=("F1", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
        )
        .reset_index()
    )
    summary["distractor_count"] = summary["distractor_count"].astype(int)
    return summary.sort_values(["model", "distractor_count", "distractor_type"])


def compute_answer_flip_rate(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["distractor_type"] == "none"].copy()
    baseline = baseline[["dataset", "question", "model", "EM"]].rename(
        columns={"EM": "baseline_EM"}
    )
    treated = df[df["distractor_type"] != "none"].copy()
    merged = treated.merge(
        baseline, on=["dataset", "question", "model"], how="inner"
    )
    merged["flipped"] = (merged["baseline_EM"] == 1) & (merged["EM"] == 0)
    group_cols = ["model", "distractor_type", "distractor_count", "dataset"]
    flip_rate = (
        merged.groupby(group_cols, observed=True)
        .agg(flip_rate=("flipped", "mean"))
        .reset_index()
    )
    flip_rate["distractor_count"] = flip_rate["distractor_count"].astype(int)
    return flip_rate


def compute_ffs(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df[df["distractor_type"] == "none"][
        ["dataset", "question", "model", "F1"]
    ].rename(columns={"F1": "baseline_F1"})
    treated = df[df["distractor_type"] != "none"].copy()
    merged = treated.merge(
        baseline, on=["dataset", "question", "model"], how="inner"
    )
    merged["F1_drop"] = merged["baseline_F1"] - merged["F1"]
    merged["FFS"] = 0.0
    has_baseline = merged["baseline_F1"] > 0
    merged.loc[has_baseline, "FFS"] = (
        merged.loc[has_baseline, "F1_drop"].clip(lower=0)
        / merged.loc[has_baseline, "baseline_F1"]
    )
    group_cols = ["model", "distractor_type", "distractor_count", "dataset"]
    ffs = (
        merged.groupby(group_cols, observed=True)
        .agg(
            mean_FFS=("FFS", "mean"),
            mean_F1_drop=("F1_drop", "mean"),
            mean_baseline_F1=("baseline_F1", "mean"),
        )
        .reset_index()
    )
    ffs["distractor_count"] = ffs["distractor_count"].astype(int)
    return ffs


def compute_overall_metrics(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model", "distractor_type", "distractor_count", "dataset"]
    agg = (
        df.groupby(group_cols, observed=True)
        .agg(
            mean_EM=("EM", "mean"),
            mean_F1=("F1", "mean"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_recall_5=("recall@5", "mean"),
            mean_recall_10=("recall@10", "mean"),
            count=("EM", "count"),
        )
        .reset_index()
    )
    agg["distractor_count"] = agg["distractor_count"].astype(int)
    return agg.sort_values(["model", "distractor_count", "distractor_type", "dataset"])


def save_summary_tables(metrics: pd.DataFrame, flip_rate: pd.DataFrame, ffs: pd.DataFrame):
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(ANALYSIS_DIR / "overall_metrics.csv", index=False)
    (ANALYSIS_DIR / "overall_metrics.txt").write_text(
        metrics.to_string(index=False)
    )
    print(f"  Saved: {ANALYSIS_DIR / 'overall_metrics.csv'}")

    flip_rate.to_csv(ANALYSIS_DIR / "answer_flip_rate.csv", index=False)
    (ANALYSIS_DIR / "answer_flip_rate.txt").write_text(
        flip_rate.to_string(index=False)
    )
    print(f"  Saved: {ANALYSIS_DIR / 'answer_flip_rate.csv'}")

    ffs.to_csv(ANALYSIS_DIR / "ffs_scores.csv", index=False)
    (ANALYSIS_DIR / "ffs_scores.txt").write_text(
        ffs.to_string(index=False)
    )
    print(f"  Saved: {ANALYSIS_DIR / 'ffs_scores.csv'}")

    ffs_leaderboard = (
        ffs.groupby("model", observed=True)
        .agg(mean_FFS=("mean_FFS", "mean"))
        .sort_values("mean_FFS")
        .reset_index()
    )
    ffs_leaderboard.to_csv(ANALYSIS_DIR / "ffs_leaderboard.csv", index=False)
    print(f"  Saved: {ANALYSIS_DIR / 'ffs_leaderboard.csv'}")

    per_dataset = (
        metrics.groupby(["model", "dataset"], observed=True)
        .agg(mean_EM=("mean_EM", "mean"), mean_F1=("mean_F1", "mean"))
        .reset_index()
    )
    per_dataset.to_csv(ANALYSIS_DIR / "per_dataset_metrics.csv", index=False)
    print(f"  Saved: {ANALYSIS_DIR / 'per_dataset_metrics.csv'}")


def plot_accuracy_vs_distractor_count(metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    datasets = metrics["dataset"].unique()
    for ax, ds in zip(axes, datasets):
        subset = metrics[metrics["dataset"] == ds]
        for model in subset["model"].unique():
            m_sub = subset[subset["model"] == model]
            for dtype in m_sub["distractor_type"].unique():
                d_sub = m_sub[m_sub["distractor_type"] == dtype]
                d_sub = d_sub.sort_values("distractor_count")
                label = f"{model} ({dtype})" if dtype != "none" else f"{model} (baseline)"
                marker = "o" if dtype == "none" else None
                linestyle = "-" if dtype == "none" else "--"
                ax.plot(
                    d_sub["distractor_count"],
                    d_sub["mean_EM"],
                    marker=marker,
                    linestyle=linestyle,
                    label=label,
                    linewidth=1.5,
                )
        ax.set_xlabel("Distractor Count")
        ax.set_ylabel("Mean EM")
        ax.set_title(f"{ds.upper()} - EM vs Distractor Count")
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks(sorted(metrics["distractor_count"].unique()))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = PLOTS_DIR / "accuracy_vs_distractor_count.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_f1_vs_distractor_count(metrics: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    datasets = metrics["dataset"].unique()
    for ax, ds in zip(axes, datasets):
        subset = metrics[metrics["dataset"] == ds]
        for model in subset["model"].unique():
            m_sub = subset[subset["model"] == model]
            for dtype in m_sub["distractor_type"].unique():
                d_sub = m_sub[m_sub["distractor_type"] == dtype]
                d_sub = d_sub.sort_values("distractor_count")
                label = f"{model} ({dtype})" if dtype != "none" else f"{model} (baseline)"
                marker = "o" if dtype == "none" else None
                linestyle = "-" if dtype == "none" else "--"
                ax.plot(
                    d_sub["distractor_count"],
                    d_sub["mean_F1"],
                    marker=marker,
                    linestyle=linestyle,
                    label=label,
                    linewidth=1.5,
                )
        ax.set_xlabel("Distractor Count")
        ax.set_ylabel("Mean F1")
        ax.set_title(f"{ds.upper()} - F1 vs Distractor Count")
        ax.legend(fontsize=8, loc="best")
        ax.set_xticks(sorted(metrics["distractor_count"].unique()))
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    plt.tight_layout()
    path = PLOTS_DIR / "f1_vs_distractor_count.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_ffs_leaderboard(ffs: pd.DataFrame):
    leaderboard = (
        ffs.groupby(["model", "distractor_type", "distractor_count"], observed=True)
        .agg(mean_FFS=("mean_FFS", "mean"))
        .groupby("model", observed=True)
        .agg(mean_FFS=("mean_FFS", "mean"))
        .sort_values("mean_FFS")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#2ecc71" if v < 0.3 else "#f39c12" if v < 0.6 else "#e74c3c" for v in leaderboard["mean_FFS"]]
    bars = ax.barh(leaderboard["model"], leaderboard["mean_FFS"], color=colors, edgecolor="gray")
    ax.set_xlabel("Mean Faithfulness Fragility Score (FFS)")
    ax.set_title("FFS Leaderboard (lower is better)")
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(1.0))

    for bar, val in zip(bars, leaderboard["mean_FFS"]):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1%}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    path = PLOTS_DIR / "ffs_leaderboard.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_distractor_heatmap(df: pd.DataFrame):
    baseline = df[df["distractor_type"] == "none"][
        ["dataset", "question", "model", "F1"]
    ].rename(columns={"F1": "baseline_F1"})
    treated = df[df["distractor_type"] != "none"].copy()
    merged = treated.merge(
        baseline, on=["dataset", "question", "model"], how="inner"
    )
    merged["F1_drop"] = merged["baseline_F1"] - merged["F1"]

    heatmap_data = (
        merged.groupby(["model", "distractor_type", "distractor_count"], observed=True)
        .agg(mean_F1_drop=("F1_drop", "mean"))
        .reset_index()
    )
    heatmap_data["distractor_count"] = heatmap_data["distractor_count"].astype(int)

    models = sorted(heatmap_data["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), squeeze=False)

    for ax, model in zip(axes[0], models):
        pivot = heatmap_data[heatmap_data["model"] == model].pivot_table(
            index="distractor_type",
            columns="distractor_count",
            values="mean_F1_drop",
            aggfunc="mean",
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            cbar_kws={"label": "Mean F1 Drop"},
            linewidths=0.5,
        )
        ax.set_title(f"{model}\nF1 Drop by Distractor Type & Count")
        ax.set_ylabel("Distractor Type")
        ax.set_xlabel("Distractor Count")

    plt.tight_layout()
    path = PLOTS_DIR / "distractor_type_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze RAG experiment results")
    parser.add_argument(
        "--csv",
        type=str,
        default=str(RESULTS_DIR / "baseline.csv"),
        help="Path to experiment CSV",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run experiments first.")
        sys.exit(1)

    print(f"Loading: {csv_path}")
    df = load_data(csv_path)
    print(f"  Rows: {len(df)}")

    # drop cached rows to avoid double-counting
    if "cached" in df.columns:
        n_cached = (df["cached"] == "1").sum()
        print(f"  Cached rows (excluded): {n_cached}")
        df = df[df["cached"] != "1"].copy()

    df["distractor_count"] = df["distractor_count"].astype(int)

    print(f"\n{'=' * 60}")
    print("COMPUTING METRICS")
    print(f"{'=' * 60}")

    metrics = compute_overall_metrics(df)
    flip_rate = compute_answer_flip_rate(df)
    ffs = compute_ffs(df)

    print("\n--- Answer Flip Rate ---")
    print(flip_rate.to_string(index=False))

    print("\n--- Faithfulness Fragility Score (FFS) ---")
    print(ffs.to_string(index=False))

    print(f"\n{'=' * 60}")
    print("SAVING SUMMARY TABLES")
    print(f"{'=' * 60}")
    save_summary_tables(metrics, flip_rate, ffs)

    print(f"\n{'=' * 60}")
    print("GENERATING PLOTS")
    print(f"{'=' * 60}")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plot_accuracy_vs_distractor_count(metrics)
    plot_f1_vs_distractor_count(metrics)
    plot_ffs_leaderboard(ffs)
    plot_distractor_heatmap(df)

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")
    print(f"  Results: file:///{RESULTS_DIR.resolve().as_posix()}")


if __name__ == "__main__":
    main()
