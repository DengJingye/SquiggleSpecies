from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .constants import SPECIES


def classification_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidence: np.ndarray,
    species: tuple[str, ...] = SPECIES,
) -> dict:
    labels = list(range(len(species)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(species),
        output_dict=True,
        zero_division=0,
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(np.mean([report[name]["recall"] for name in species])),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "mean_confidence": float(np.mean(confidence)),
        "per_species": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in species
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
