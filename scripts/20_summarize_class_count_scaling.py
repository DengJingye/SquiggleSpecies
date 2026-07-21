#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from squiggle_species.utils import save_json, write_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize formally retrained 5-9 class models.")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--k9-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(args.protocol.read_text())

    rows = []
    for subset in protocol["subsets"]:
        class_count = int(subset["class_count"])
        summary_path = args.k9_summary if class_count == 9 else args.run_root / f"k{class_count}" / "pft_best" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text())
        val = summary["metrics"]["val"]
        test = summary["metrics"]["test"]
        rows.append(
            {
                "class_count": class_count,
                "species": ";".join(subset["species"]),
                "selection_protocol": protocol["protocol_name"],
                "val_accuracy": val["accuracy"],
                "val_macro_f1": val["macro_f1"],
                "test_accuracy": test["accuracy"],
                "test_macro_f1": test["macro_f1"],
                "summary": str(summary_path.resolve()),
            }
        )
    rows.sort(key=lambda row: row["class_count"])
    write_csv(args.output_dir / "class_count_metrics.csv", rows, list(rows[0]))

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    class_counts = [row["class_count"] for row in rows]
    ax.plot(class_counts, [row["test_macro_f1"] for row in rows], marker="o", linewidth=2, label="Test macro-F1")
    ax.plot(class_counts, [row["test_accuracy"] for row in rows], marker="s", linewidth=2, label="Test accuracy")
    ax.set_xticks(class_counts)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Number of classes")
    ax.set_ylabel("Score")
    ax.set_title("Signal classification scaling under a pre-registered nested protocol")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "class_count_scaling_curve.png", dpi=220)
    plt.close(fig)

    summary = {
        "status": "complete",
        "interpretation": (
            "Each point is a separately trained classifier. The curve measures task granularity under fixed "
            "data and split rules; it must not be described as post-hoc removal of difficult test classes."
        ),
        "rows": rows,
    }
    save_json(args.output_dir / "class_count_scaling_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
