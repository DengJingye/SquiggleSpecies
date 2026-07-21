from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .constants import SPECIES


def read_prediction_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Prediction table is empty: {path}")
    for row in rows:
        if "predicted_label" not in row and "pred_label" in row:
            row["predicted_label"] = row["pred_label"]
    required = {"true_label", "predicted_label", "confidence"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Prediction table is missing columns: {missing}")
    return rows


def _correctness_auroc(correct: np.ndarray, confidence: np.ndarray) -> float | None:
    if np.unique(correct).size < 2:
        return None
    return float(roc_auc_score(correct.astype(np.int8), confidence))


def _aurc(correct: np.ndarray, confidence: np.ndarray) -> float:
    order = np.argsort(-confidence, kind="stable")
    cumulative_errors = np.cumsum(~correct[order], dtype=np.float64)
    accepted = np.arange(1, len(correct) + 1, dtype=np.float64)
    coverage = accepted / len(correct)
    selective_risk = cumulative_errors / accepted
    return float(np.trapz(np.r_[0.0, selective_risk], np.r_[0.0, coverage]))


def _curve_from_ranked_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    grid_size: int,
) -> list[dict]:
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    num_classes = len(SPECIES)
    order = np.argsort(-confidence, kind="stable")
    sorted_true = y_true[order]
    sorted_pred = y_pred[order]
    sorted_confidence = confidence[order]
    boundary_indices = np.flatnonzero(
        np.r_[sorted_confidence[:-1] != sorted_confidence[1:], True]
    )
    if len(boundary_indices) > grid_size:
        positions = np.linspace(0, len(boundary_indices) - 1, grid_size, dtype=np.int64)
        boundary_indices = boundary_indices[np.unique(positions)]
    accepted_sizes = boundary_indices + 1
    if accepted_sizes[-1] != len(y_true):
        accepted_sizes = np.r_[accepted_sizes, len(y_true)]

    class_totals = np.bincount(y_true, minlength=num_classes)
    pair_ids = sorted_true * num_classes + sorted_pred
    cumulative_pairs = np.zeros(num_classes * num_classes, dtype=np.int64)
    curve = []
    previous = 0
    for n in accepted_sizes:
        cumulative_pairs += np.bincount(pair_ids[previous:n], minlength=num_classes * num_classes)
        previous = int(n)
        matrix = cumulative_pairs.reshape(num_classes, num_classes)
        support = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        true_positive = np.diag(matrix)
        precision = np.divide(
            true_positive,
            predicted,
            out=np.zeros(num_classes, dtype=np.float64),
            where=predicted > 0,
        )
        recall = np.divide(
            true_positive,
            support,
            out=np.zeros(num_classes, dtype=np.float64),
            where=support > 0,
        )
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros(num_classes, dtype=np.float64),
            where=(precision + recall) > 0,
        )
        per_class_coverage = support / class_totals
        accuracy = float(true_positive.sum() / n)
        row = {
            "threshold": 0.0 if n == len(y_true) else float(sorted_confidence[n - 1]),
            "accepted_reads": int(n),
            "coverage": float(n / len(y_true)),
            "accepted_accuracy": accuracy,
            "accepted_macro_f1": float(f1.mean()),
            "selective_risk": 1.0 - accuracy,
            "min_per_class_coverage": float(per_class_coverage.min()),
        }
        row.update(
            {
                f"coverage_{species}": float(per_class_coverage[index])
                for index, species in enumerate(SPECIES)
            }
        )
        curve.append(row)
    return curve


def calibrate_threshold(
    rows: list[dict[str, str]],
    min_coverage: float = 0.5,
    target_accuracy: float | None = 0.90,
    min_per_class_coverage: float = 0.5,
    min_accuracy_gain: float = 0.01,
    grid_size: int = 1001,
) -> tuple[dict, list[dict]]:
    if not 0 < min_coverage <= 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if target_accuracy is not None and not 0 < target_accuracy <= 1:
        raise ValueError("target_accuracy must be in (0, 1]")
    if not 0 <= min_per_class_coverage <= 1:
        raise ValueError("min_per_class_coverage must be in [0, 1]")
    if min_accuracy_gain < 0:
        raise ValueError("min_accuracy_gain must be non-negative")

    confidence = np.asarray([float(row["confidence"]) for row in rows], dtype=np.float64)
    y_true = np.asarray([int(row["true_label"]) for row in rows], dtype=np.int64)
    y_pred = np.asarray([int(row["predicted_label"]) for row in rows], dtype=np.int64)
    labels = np.arange(len(SPECIES), dtype=np.int64)
    class_totals = np.bincount(y_true, minlength=len(SPECIES))
    present_classes = class_totals > 0
    if not np.all(present_classes):
        missing = [SPECIES[index] for index in labels[~present_classes]]
        raise ValueError(f"Validation predictions do not contain every configured class: {missing}")

    baseline_accuracy = float(accuracy_score(y_true, y_pred))
    baseline_macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    curve = _curve_from_ranked_predictions(y_true, y_pred, confidence, grid_size)
    for row in curve:
        row["accuracy_gain"] = row["accepted_accuracy"] - baseline_accuracy
        row["macro_f1_gain"] = row["accepted_macro_f1"] - baseline_macro_f1

    baseline = min(curve, key=lambda row: row["threshold"])
    eligible = [
        row
        for row in curve
        if row["coverage"] >= min_coverage
        and row["min_per_class_coverage"] >= min_per_class_coverage
    ]
    if target_accuracy is not None:
        eligible = [row for row in eligible if row["accepted_accuracy"] >= target_accuracy]
    eligible = [row for row in eligible if row["accuracy_gain"] >= min_accuracy_gain]

    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["coverage"],
                row["min_per_class_coverage"],
                row["accepted_macro_f1"],
                -row["threshold"],
            ),
        )
        threshold_enabled = True
        selection_status = "target_met"
        note = "Freeze this validation-selected operating point before test or external evaluation."
    else:
        selected = baseline
        threshold_enabled = False
        selection_status = "no_feasible_selective_operating_point"
        note = (
            "No threshold met the declared reliability, overall coverage, per-class coverage, and gain constraints. "
            "Abstention is disabled; test or external data must not be used to repair the threshold."
        )

    correct = y_true == y_pred
    summary = {
        "status": "complete",
        "selection_split": "validation",
        "selection_status": selection_status,
        "selection_rule": (
            "maximize validation coverage subject to target accepted accuracy, minimum overall coverage, "
            "minimum per-class coverage, and minimum accuracy gain"
        ),
        "threshold_enabled": threshold_enabled,
        "threshold": selected["threshold"] if threshold_enabled else 0.0,
        "minimum_coverage": min_coverage,
        "minimum_per_class_coverage": min_per_class_coverage,
        "minimum_accuracy_gain": min_accuracy_gain,
        "target_accuracy": target_accuracy,
        "baseline_metrics": baseline,
        "selected_metrics": selected,
        "confidence_diagnostics": {
            "correctness_auroc": _correctness_auroc(correct, confidence),
            "aurc": _aurc(correct, confidence),
            "interpretation": (
                "AUROC measures whether confidence ranks correct predictions above errors; "
                "AURC summarizes selective risk across coverages."
            ),
        },
        "note": note,
    }
    return summary, curve


def plot_calibration_curve(curve: list[dict], output: str | Path, summary: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "squiggle_species_mpl"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    minimum_coverage = float(summary["minimum_coverage"])
    visible = [row for row in curve if row["accepted_reads"] > 0 and row["coverage"] >= minimum_coverage]
    visible.sort(key=lambda row: row["coverage"])
    coverage = np.asarray([row["coverage"] for row in visible])
    accuracy = np.asarray([row["accepted_accuracy"] for row in visible])
    macro_f1 = np.asarray([row["accepted_macro_f1"] for row in visible])
    min_class_coverage = np.asarray([row["min_per_class_coverage"] for row in visible])

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.plot(coverage, accuracy, color="#276FBF", linewidth=2.4, label="Accepted accuracy")
    ax.plot(coverage, macro_f1, color="#2A9D6F", linewidth=2.4, label="Accepted macro-F1")
    ax.plot(
        coverage,
        min_class_coverage,
        color="#D97706",
        linewidth=2,
        linestyle="--",
        label="Minimum per-species coverage",
    )
    target = summary.get("target_accuracy")
    if target is not None:
        ax.axhline(target, color="#9F1239", linewidth=1.8, linestyle=":", label=f"Accuracy target={target:.2f}")

    selected = summary["selected_metrics"]
    if summary["threshold_enabled"]:
        ax.scatter(
            [selected["coverage"]],
            [selected["accepted_accuracy"]],
            s=80,
            color="#1F2937",
            zorder=5,
            label=f"Selected threshold={summary['threshold']:.3f}",
        )
        annotation = (
            f"threshold={summary['threshold']:.3f}\n"
            f"coverage={selected['coverage']:.3f}\n"
            f"accuracy={selected['accepted_accuracy']:.3f}\n"
            f"macro-F1={selected['accepted_macro_f1']:.3f}"
        )
        ax.annotate(
            annotation,
            (selected["coverage"], selected["accepted_accuracy"]),
            xytext=(-135, -76),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#94A3B8"},
            arrowprops={"arrowstyle": "->", "color": "#64748B"},
        )
    else:
        ax.text(
            0.02,
            0.05,
            "No feasible selective operating point; abstention disabled",
            transform=ax.transAxes,
            fontsize=10,
            color="#9F1239",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF1F2", "edgecolor": "#FDA4AF"},
        )

    ax.set_xlabel("Accepted coverage")
    ax.set_ylabel("Validation metric")
    ax.set_xlim(minimum_coverage, 1.005)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Validation selective operating-point calibration")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)
