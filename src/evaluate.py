import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from scipy import stats

PRIMARY_METRIC = "accuracy"  # Must match train.py WandB summary key


def _plot_learning_curve(history_df: pd.DataFrame, run_id: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    for col in history_df.columns:
        if col.startswith("train_") and col.endswith("acc"):
            sns.lineplot(data=history_df, x="epoch", y=col, ax=ax, label=col)
        if col.startswith("val_") and col.endswith("acc"):
            sns.lineplot(data=history_df, x="epoch", y=col, ax=ax, label=col)
    ax.set_title(f"Learning Curve – {run_id}")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Epoch")
    ax.legend()
    fig.tight_layout()
    path = out_dir / f"{run_id}_learning_curve.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _plot_confusion_matrix(cm: np.ndarray, run_id: str, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix – {run_id}")
    fig.tight_layout()
    path = out_dir / f"{run_id}_confusion_matrix.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def _export_run_metrics(run, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    history_df = run.history()  # pandas DataFrame
    summary = dict(run.summary)
    config = dict(run.config)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as fp:
        json.dump(
            {
                "history": history_df.to_dict(orient="list"),
                "summary": summary,
                "config": config,
            },
            fp,
            indent=2,
        )

    generated_paths: List[Path] = []
    if not history_df.empty and "epoch" in history_df:
        generated_paths.append(_plot_learning_curve(history_df, run.id, out_dir))

    # Confusion matrix figure (prefer test, else val)
    cm_key = "test_confusion_matrix" if "test_confusion_matrix" in summary else "val_confusion_matrix"
    if cm_key in summary and summary[cm_key] is not None:
        cm_arr = np.array(summary[cm_key])
        generated_paths.append(_plot_confusion_matrix(cm_arr, run.id, out_dir))

    for p in generated_paths:
        print(p)

    # Return numeric summary metrics only
    numeric_metrics = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
    return numeric_metrics


def _is_metric_to_minimise(metric_name: str) -> bool:
    low_better_tokens = ["loss", "error", "ece", "variance", "perplexity"]
    return any(tok in metric_name.lower() for tok in low_better_tokens)


def main():
    parser = argparse.ArgumentParser("Independent evaluation & visualisation script")
    parser.add_argument("results_dir", type=str)
    parser.add_argument("run_ids", type=str, help="JSON list string e.g. '[\"run1\",\"run2\"]'")
    args = parser.parse_args()

    results_root = Path(args.results_dir).expanduser().resolve()
    run_ids: List[str] = json.loads(args.run_ids)

    # Load global WandB config
    import yaml
    base_cfg = yaml.safe_load(
        open(Path(__file__).resolve().parent.parent / "config" / "config.yaml", "r")
    )
    entity = base_cfg["wandb"]["entity"]
    project = base_cfg["wandb"]["project"]

    api = wandb.Api()

    aggregated_metrics: Dict[str, Dict[str, float]] = {}
    per_run_primary: Dict[str, float] = {}
    val_acc_time_series: Dict[str, List[float]] = {}

    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        run_dir = results_root / rid
        numeric_metrics = _export_run_metrics(run, run_dir)
        for mname, mval in numeric_metrics.items():
            aggregated_metrics.setdefault(mname, {})[rid] = float(mval)
        per_run_primary[rid] = numeric_metrics.get(PRIMARY_METRIC, 0.0)

        # Collect validation accuracy series for significance testing / box plots
        hist_df = run.history()
        if "val_acc" in hist_df:
            val_acc_time_series[rid] = list(hist_df["val_acc"].dropna().values)

    # ---------------- Aggregated metrics file ---------------- #
    comparison_dir = results_root / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Identify best proposed / baseline
    best_proposed_id = max(
        (rid for rid in run_ids if "proposed" in rid), key=lambda x: per_run_primary.get(x, -np.inf)
    )
    best_baseline_id = max(
        (rid for rid in run_ids if ("comparative" in rid or "baseline" in rid)),
        key=lambda x: per_run_primary.get(x, -np.inf),
    )

    gap_direction = -1.0 if _is_metric_to_minimise(PRIMARY_METRIC) else 1.0
    raw_gap = (
        per_run_primary[best_proposed_id] - per_run_primary[best_baseline_id]
    )
    gap = gap_direction * raw_gap / max(1e-8, abs(per_run_primary[best_baseline_id])) * 100.0

    # Statistical significance test (Welch t-test) using val_acc time series
    p_value = None
    if best_proposed_id in val_acc_time_series and best_baseline_id in val_acc_time_series:
        try:
            t_stat, p_val = stats.ttest_ind(
                val_acc_time_series[best_proposed_id],
                val_acc_time_series[best_baseline_id],
                equal_var=False,
            )
            p_value = float(p_val)
        except Exception:
            p_value = None

    aggregated_json = {
        "primary_metric": PRIMARY_METRIC,
        "metrics": aggregated_metrics,
        "best_proposed": {
            "run_id": best_proposed_id,
            "value": per_run_primary[best_proposed_id],
        },
        "best_baseline": {
            "run_id": best_baseline_id,
            "value": per_run_primary[best_baseline_id],
        },
        "gap": gap,
        "p_value": p_value,
    }
    with open(comparison_dir / "aggregated_metrics.json", "w") as fp:
        json.dump(aggregated_json, fp, indent=2)
    print(comparison_dir / "aggregated_metrics.json")

    # ---------------- Comparison figures ---------------- #
    # 1. Bar charts for all collected metrics
    for metric_name, run_dict in aggregated_metrics.items():
        df_plot = pd.DataFrame(list(run_dict.items()), columns=["run_id", metric_name])
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=df_plot, x="run_id", y=metric_name, ax=ax)
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f"{height:.3f}", (p.get_x() + p.get_width() / 2., height),
                        ha="center", va="bottom")
        ax.set_title(f"{metric_name} Comparison")
        ax.set_xlabel("Run ID")
        ax.set_ylabel(metric_name)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        path = comparison_dir / f"comparison_{metric_name}_bar_chart.pdf"
        fig.savefig(path)
        plt.close(fig)
        print(path)

    # 2. Box plot for primary metric across epochs
    if val_acc_time_series:
        df_box = pd.DataFrame(
            [(rid, acc) for rid, accs in val_acc_time_series.items() for acc in accs],
            columns=["run_id", "val_acc"],
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df_box, x="run_id", y="val_acc", ax=ax)
        ax.set_title("Validation Accuracy Distribution Across Epochs")
        ax.set_ylabel("val_acc")
        ax.set_xlabel("Run ID")
        fig.tight_layout()
        path_box = comparison_dir / "comparison_accuracy_box_plot.pdf"
        fig.savefig(path_box)
        plt.close(fig)
        print(path_box)

    # 3. Append significance result to a text file for transparency
    if p_value is not None:
        sig_path = comparison_dir / "statistical_significance.txt"
        with open(sig_path, "w") as fp:
            fp.write(
                f"Welch t-test between {best_proposed_id} and {best_baseline_id} on val_acc\n"
                f"p-value: {p_value:.6f}\n"
            )
        print(sig_path)


if __name__ == "__main__":
    main()
