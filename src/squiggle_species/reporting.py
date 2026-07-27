from __future__ import annotations

import csv
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from .constants import SPECIES
from .metrics import classification_summary
from .utils import save_json, write_csv


def load_predictions(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction table is empty: {path}")
    for row in rows:
        if "predicted_label" not in row and "pred_label" in row:
            row["predicted_label"] = row["pred_label"]
    required = {"read_id", "predicted_label", "confidence"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    return rows


def build_report(
    rows: list[dict[str, str]],
    output_dir: str | Path,
    threshold: float = 0.0,
    threshold_enabled: bool | None = None,
    calibration: dict | None = None,
    class_names: tuple[str, ...] | list[str] | None = None,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = tuple(class_names or SPECIES)
    confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)
    y_pred = np.asarray([int(row["predicted_label"]) for row in rows], dtype=np.int64)
    if threshold_enabled is None:
        threshold_enabled = threshold > 0
    accepted = confidence >= threshold if threshold_enabled else np.ones(len(rows), dtype=bool)
    abundance = Counter(int(label) for label in y_pred[accepted])
    abundance_rows = [
        {
            "label": label,
            "species": class_names[label] if 0 <= label < len(class_names) else f"class_{label}",
            "reads": abundance.get(label, 0),
            "fraction_of_accepted": abundance.get(label, 0) / max(1, int(accepted.sum())),
        }
        for label in range(len(class_names))
    ]
    write_csv(output_dir / "species_abundance.csv", abundance_rows, list(abundance_rows[0]))
    summary = {
        "status": "complete",
        "total_reads": len(rows),
        "threshold_enabled": bool(threshold_enabled),
        "threshold": float(threshold),
        "calibration_status": calibration.get("selection_status") if calibration else None,
        "accepted_reads": int(accepted.sum()),
        "accepted_rate": float(accepted.mean()),
        "abstained_reads": int((~accepted).sum()),
        "abundance": abundance_rows,
    }
    has_truth = "true_label" in rows[0] and all(row.get("true_label", "") != "" for row in rows)
    if has_truth:
        y_true = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
        full_metrics = classification_summary(y_true, y_pred, confidence, class_names)
        accepted_metrics = (
            classification_summary(y_true[accepted], y_pred[accepted], confidence[accepted], class_names)
            if accepted.any()
            else {}
        )
        summary["full_metrics"] = full_metrics
        summary["accepted_metrics"] = accepted_metrics
        if accepted_metrics:
            matrix = np.asarray(accepted_metrics["confusion_matrix"], dtype=np.int64)
            np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")
            per_species = _per_species_rows(
                y_true,
                accepted,
                full_metrics,
                accepted_metrics,
                class_names,
            )
            write_csv(output_dir / "per_species_metrics.csv", per_species, list(per_species[0]))
            summary["per_species_acceptance"] = {
                row["species"]: {
                    "accepted_reads": row["accepted_reads"],
                    "total_reads": row["total_reads"],
                    "acceptance_rate": row["acceptance_rate"],
                }
                for row in per_species
            }
            _plot_confusion(matrix, class_names, output_dir / "confusion_matrix.png")
            _plot_per_species(per_species, output_dir / "per_species_metrics.png")
            correctness_auroc = None
            correct = y_true == y_pred
            if np.unique(correct).size == 2:
                correctness_auroc = float(roc_auc_score(correct.astype(np.int8), confidence))
            summary["confidence_correctness_auroc"] = correctness_auroc
            _plot_confidence(
                confidence,
                correct,
                threshold,
                bool(threshold_enabled),
                correctness_auroc,
                output_dir / "confidence_distribution.png",
            )
            metric_rows = [
                {"metric": "full_accuracy", "value": full_metrics["accuracy"]},
                {"metric": "full_macro_f1", "value": full_metrics["macro_f1"]},
                {"metric": "accepted_accuracy", "value": accepted_metrics["accuracy"]},
                {"metric": "accepted_macro_f1", "value": accepted_metrics["macro_f1"]},
                {"metric": "accepted_rate", "value": float(accepted.mean())},
                {"metric": "confidence_correctness_auroc", "value": correctness_auroc},
            ]
            write_csv(output_dir / "metrics_summary.csv", metric_rows, ["metric", "value"])
    else:
        _plot_abundance(
            abundance_rows,
            output_dir / "species_abundance.png",
        )
        _plot_unlabeled_confidence(
            confidence,
            accepted,
            threshold,
            bool(threshold_enabled),
            output_dir / "confidence_distribution.png",
        )
    save_json(output_dir / "report_summary.json", summary)
    return summary


def _per_species_rows(
    y_true: np.ndarray,
    accepted: np.ndarray,
    full_metrics: dict,
    accepted_metrics: dict,
    class_names: tuple[str, ...],
) -> list[dict]:
    rows = []
    for label, species in enumerate(class_names):
        total_reads = int(np.sum(y_true == label))
        accepted_reads = int(np.sum((y_true == label) & accepted))
        full = full_metrics["per_species"][species]
        selected = accepted_metrics["per_species"][species]
        rows.append(
            {
                "species": species,
                "precision": selected["precision"],
                "recall": selected["recall"],
                "f1": selected["f1"],
                "support": selected["support"],
                "full_precision": full["precision"],
                "full_recall": full["recall"],
                "full_f1": full["f1"],
                "accepted_precision": selected["precision"],
                "accepted_recall": selected["recall"],
                "accepted_f1": selected["f1"],
                "total_reads": total_reads,
                "accepted_reads": accepted_reads,
                "acceptance_rate": accepted_reads / max(1, total_reads),
            }
        )
    return rows


def _plot_confusion(matrix: np.ndarray, class_names: tuple[str, ...], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "squiggle_species_mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reference label")
    ax.set_title("Accepted read-level confusion matrix")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_per_species(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    species = [row["species"] for row in rows]
    x = np.arange(len(species))
    width = 0.34
    fig, (metric_ax, coverage_ax) = plt.subplots(
        2,
        1,
        figsize=(10, 7.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]},
    )
    metric_ax.bar(x - width / 2, [row["full_f1"] for row in rows], width, label="Full-coverage F1", color="#64748B")
    metric_ax.bar(x + width / 2, [row["accepted_f1"] for row in rows], width, label="Accepted-set F1", color="#2A9D6F")
    metric_ax.set_ylim(0, 1.02)
    metric_ax.set_ylabel("F1")
    metric_ax.set_title("Per-species performance and selective coverage")
    metric_ax.legend(frameon=False, ncol=2)
    metric_ax.grid(axis="y", alpha=0.25)

    rates = [row["acceptance_rate"] for row in rows]
    coverage_ax.bar(x, rates, width=0.58, color="#D97706")
    coverage_ax.set_ylim(0, 1.02)
    coverage_ax.set_ylabel("Acceptance rate")
    coverage_ax.set_xticks(x, species, rotation=35, ha="right")
    coverage_ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(rates):
        coverage_ax.text(index, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_confidence(
    confidence: np.ndarray,
    correct: np.ndarray,
    threshold: float,
    threshold_enabled: bool,
    correctness_auroc: float | None,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, 31)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    ax.hist(confidence[correct], bins=bins, density=True, alpha=0.65, color="#2A9D6F", label="Correct")
    ax.hist(confidence[~correct], bins=bins, density=True, alpha=0.65, color="#C84A4A", label="Incorrect")
    if threshold_enabled:
        ax.axvline(threshold, color="#1F2937", linestyle="--", linewidth=2, label=f"Threshold={threshold:.3f}")
    diagnostic = "Correctness AUROC: unavailable" if correctness_auroc is None else f"Correctness AUROC={correctness_auroc:.3f}"
    ax.text(
        0.02,
        0.95,
        diagnostic,
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Density")
    ax.set_title("Confidence distribution by prediction correctness")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_abundance(rows: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [row["species"] for row in rows]
    values = [row["fraction_of_accepted"] for row in rows]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    bars = ax.bar(labels, values, color="#2A9D6F")
    ax.set_ylim(0, max(1.0, max(values, default=0.0) * 1.15))
    ax.set_ylabel("Fraction of accepted reads")
    ax.set_title("Predicted species composition")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_unlabeled_confidence(
    confidence: np.ndarray,
    accepted: np.ndarray,
    threshold: float,
    threshold_enabled: bool,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.hist(confidence, bins=np.linspace(0, 1, 31), color="#64748B", alpha=0.8)
    if threshold_enabled:
        ax.axvline(
            threshold,
            color="#C84A4A",
            linestyle="--",
            linewidth=2,
            label=f"Frozen threshold={threshold:.3f}",
        )
        ax.legend(frameon=False)
    ax.text(
        0.02,
        0.95,
        f"Accepted: {accepted.mean():.1%}",
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#CBD5E1"},
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("Reads")
    ax.set_title("Read-level confidence distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
