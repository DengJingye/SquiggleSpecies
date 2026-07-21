from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path


REQUIRED_COLUMNS = {"read_id", "split", "barcode", "label"}


def audit_manifest(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")

    seen_rows = Counter(row["read_id"] for row in rows)
    duplicated_rows = sorted(read_id for read_id, count in seen_rows.items() if count > 1)
    read_splits: dict[str, set[str]] = defaultdict(set)
    file_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    split_species = Counter()
    for row in rows:
        read_splits[row["read_id"]].add(row["split"])
        split_species[(row["split"], row["barcode"])] += 1
        ccf_file = row.get("ccf_file") or row.get("source_ccf_file") or ""
        if ccf_file:
            file_splits[(row["barcode"], str(Path(ccf_file).name))].add(row["split"])

    read_leakage = sorted(read_id for read_id, splits in read_splits.items() if len(splits) > 1)
    file_leakage = sorted(f"{barcode}/{name}" for (barcode, name), splits in file_splits.items() if len(splits) > 1)
    return {
        "status": "pass" if not read_leakage and not file_leakage and not duplicated_rows else "fail",
        "manifest": str(path.resolve()),
        "rows": len(rows),
        "unique_reads": len(seen_rows),
        "columns": sorted(columns),
        "duplicate_read_rows": len(duplicated_rows),
        "read_id_split_leakage": len(read_leakage),
        "ccf_file_split_leakage": len(file_leakage),
        "examples": {
            "duplicate_read_ids": duplicated_rows[:10],
            "read_id_leakage": read_leakage[:10],
            "ccf_file_leakage": file_leakage[:10],
        },
        "counts": {
            split: {
                species: split_species[(split, species)]
                for species in sorted({row["barcode"] for row in rows})
            }
            for split in sorted({row["split"] for row in rows})
        },
    }
