from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .bonito_pft import BonitoPartialStudent, raw_mil_collate
from .ccf import discover_ccf5, preprocess_read, text_read_id
from .model_bundle import load_model_bundle, verify_bonito_weights
from .models import SignalStudent
from .reporting import build_report
from .utils import save_json


def _run_signature(bundle: dict, files: list[Path], max_reads: int) -> str:
    payload = {
        "bundle_sha256": bundle["_bundle_sha256"],
        "checkpoint_sha256": bundle.get("checkpoint", {}).get("sha256"),
        "files": [
            {
                "path": str(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in files
        ],
        "max_reads": int(max_reads),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_model(
    bundle: dict,
    bonito_model_dir: str | Path,
    device: str,
) -> tuple[BonitoPartialStudent, tuple[str, ...]]:
    verify_bonito_weights(bundle, bonito_model_dir)
    checkpoint = torch.load(bundle["_checkpoint_path"], map_location="cpu", weights_only=False)
    required = {"model_state_dict", "experiment", "trainable_lstm_blocks"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    class_names = tuple(bundle["class_names"])
    checkpoint_species = tuple(
        checkpoint.get("species", checkpoint["experiment"].get("species", ()))
    )
    if checkpoint_species != class_names:
        raise ValueError(
            "Model bundle class order does not match checkpoint: "
            f"bundle={class_names}, checkpoint={checkpoint_species}"
        )
    expected_blocks = bundle.get("adaptation", {}).get("unfreeze_last_n")
    if expected_blocks is not None and int(expected_blocks) != int(
        checkpoint["trainable_lstm_blocks"]
    ):
        raise ValueError(
            "Model bundle adaptation depth does not match checkpoint: "
            f"bundle={expected_blocks}, checkpoint={checkpoint['trainable_lstm_blocks']}"
        )

    experiment = checkpoint["experiment"]
    model_config = experiment["model"]
    student = SignalStudent(
        input_dim=768,
        hidden_dim=int(model_config["hidden_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        attention_dim=int(model_config["attention_dim"]),
        num_classes=len(class_names),
        dropout=float(model_config["dropout"]),
        transformer_layers=int(model_config["transformer_layers"]),
        transformer_heads=int(model_config["transformer_heads"]),
        transformer_ff_dim=int(model_config["transformer_ff_dim"]),
        aggregation=str(model_config.get("aggregation", "transformer")),
    )
    from bonito.util import load_model

    bonito_model = load_model(str(bonito_model_dir), device=device)
    chunk_microbatch = int(
        bundle.get("runtime", {}).get(
            "chunk_microbatch",
            experiment.get("training", {}).get("chunk_microbatch", 8),
        )
    )
    model = BonitoPartialStudent(
        bonito_model=bonito_model,
        student=student,
        trainable_lstm_blocks=int(checkpoint["trainable_lstm_blocks"]),
        chunk_microbatch=chunk_microbatch,
    ).to(device)
    del bonito_model
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_names


def _select_chunks(chunks: np.ndarray, max_chunks: int) -> np.ndarray:
    if len(chunks) <= max_chunks:
        return np.asarray(chunks, dtype=np.float32)
    indices = np.linspace(0, len(chunks) - 1, num=max_chunks, dtype=np.int64)
    return np.asarray(chunks[indices], dtype=np.float32)


def _predict_batch(
    model: BonitoPartialStudent,
    batch: list[tuple[np.ndarray, int, str]],
    class_names: tuple[str, ...],
    device: str,
) -> list[dict]:
    raw, mask, _, read_ids = raw_mil_collate(batch)
    with torch.no_grad():
        raw = raw.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.startswith("cuda"),
        ):
            logits, _, _ = model(raw, mask)
        probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    rows = []
    for index, read_id in enumerate(read_ids):
        predicted_label = int(predictions[index])
        row = {
            "read_id": read_id,
            "predicted_label": predicted_label,
            "predicted_species": class_names[predicted_label],
            "confidence": float(confidence[index]),
        }
        for label, name in enumerate(class_names):
            row[f"prob_{name}"] = float(probabilities[index, label])
        rows.append(row)
    return rows


def classify_ccf5(
    input_path: str | Path,
    model_bundle: str | Path,
    bonito_model_dir: str | Path,
    output_dir: str | Path,
    *,
    device: str = "cpu",
    batch_size: int | None = None,
    max_reads: int = 0,
    verify_hashes: bool = True,
) -> dict:
    try:
        import pyccf5 as slow5
    except ImportError as error:
        raise RuntimeError(
            "classify-ccf requires pyccf5; run it in the CCF5 extraction environment"
        ) from error
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    bundle = load_model_bundle(model_bundle, verify_hashes=verify_hashes)
    files = discover_ccf5(input_path)
    output_dir = Path(output_dir).resolve()
    per_file_dir = output_dir / "per_file"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    signature = _run_signature(bundle, files, max_reads)
    model, class_names = _load_model(bundle, bonito_model_dir, device)

    chunking = bundle["chunking"]
    preprocessing = bundle["preprocessing"]
    profile_id = str(preprocessing["profile_id"])
    discard_first = int(chunking["discard_first"])
    chunk_size = int(chunking["chunk_size"])
    overlap = int(chunking["overlap"])
    max_chunks = int(chunking["max_chunks"])
    batch_size = int(
        batch_size
        or bundle.get("runtime", {}).get("eval_batch_size", 6)
    )
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    all_rows: list[dict[str, str]] = []
    file_summaries = []
    remaining = max_reads
    seen_read_ids: set[str] = set()
    for file_index, ccf_path in enumerate(files):
        if max_reads > 0 and remaining <= 0:
            break
        stem = f"{file_index:04d}_{ccf_path.stem}"
        prediction_path = per_file_dir / f"{stem}_predictions.csv"
        summary_path = per_file_dir / f"{stem}_summary.json"
        if summary_path.exists() and prediction_path.exists():
            prior = json.loads(summary_path.read_text())
            if prior.get("status") == "complete" and prior.get("run_signature") == signature:
                with prediction_path.open(newline="") as handle:
                    prior_rows = list(csv.DictReader(handle))
                duplicates = seen_read_ids & {row["read_id"] for row in prior_rows}
                if duplicates:
                    raise ValueError(f"Duplicate read IDs across resumed files: {sorted(duplicates)[:3]}")
                seen_read_ids.update(row["read_id"] for row in prior_rows)
                all_rows.extend(prior_rows)
                file_summaries.append(prior)
                if max_reads > 0:
                    remaining -= int(prior["eligible_reads"])
                continue

        temporary = prediction_path.with_suffix(prediction_path.suffix + ".tmp")
        batch: list[tuple[np.ndarray, int, str]] = []
        batch_meta: list[tuple[int, str]] = []
        rows: list[dict] = []
        scanned = eligible = too_short = invalid = 0

        def flush() -> None:
            if not batch:
                return
            predicted = _predict_batch(model, batch, class_names, device)
            for row, (source_read_index, source_file) in zip(predicted, batch_meta):
                row["source_file"] = source_file
                row["source_read_index"] = source_read_index
                rows.append(row)
            batch.clear()
            batch_meta.clear()

        reader = slow5.Open(str(ccf_path), "r", DEBUG=0)
        try:
            for read_index, record in enumerate(reader.seq_reads(aux="all")):
                if max_reads > 0 and remaining <= 0:
                    break
                scanned += 1
                try:
                    chunks = preprocess_read(
                        record,
                        profile_id=profile_id,
                        discard_first=discard_first,
                        chunk_size=chunk_size,
                        overlap=overlap,
                    )
                except Exception:
                    invalid += 1
                    continue
                if not len(chunks):
                    too_short += 1
                    continue
                read_id = text_read_id(record["read_id"])
                if read_id in seen_read_ids:
                    raise ValueError(f"Duplicate read_id detected: {read_id}")
                seen_read_ids.add(read_id)
                batch.append((_select_chunks(chunks, max_chunks), -1, read_id))
                batch_meta.append((read_index, str(ccf_path)))
                eligible += 1
                if max_reads > 0:
                    remaining -= 1
                if len(batch) >= batch_size:
                    flush()
            flush()
        finally:
            reader.close()

        fields = list(rows[0]) if rows else [
            "read_id",
            "predicted_label",
            "predicted_species",
            "confidence",
            *[f"prob_{name}" for name in class_names],
            "source_file",
            "source_read_index",
        ]
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, prediction_path)
        file_summary = {
            "status": "complete",
            "run_signature": signature,
            "source_file": str(ccf_path),
            "scanned_reads": scanned,
            "eligible_reads": eligible,
            "too_short_reads": too_short,
            "invalid_reads": invalid,
            "predictions": str(prediction_path),
        }
        save_json(summary_path, file_summary)
        all_rows.extend(rows)
        file_summaries.append(file_summary)

    if not all_rows:
        raise ValueError("No eligible reads were found in the selected CCF5 files")
    merged_predictions = output_dir / "read_predictions.csv"
    temporary = merged_predictions.with_suffix(merged_predictions.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(temporary, merged_predictions)

    calibration = bundle.get("calibration") or {}
    threshold_enabled = bool(calibration.get("threshold_enabled", False))
    threshold = float(calibration.get("threshold", 0.0)) if threshold_enabled else 0.0
    report = build_report(
        all_rows,
        output_dir / "report",
        threshold=threshold,
        threshold_enabled=threshold_enabled,
        calibration=calibration or None,
        class_names=class_names,
    )
    summary = {
        "status": "complete",
        "mode": "streaming_ccf5_to_classification_report",
        "input": str(Path(input_path).resolve()),
        "model_bundle": bundle["_bundle_path"],
        "model_bundle_sha256": bundle["_bundle_sha256"],
        "checkpoint": bundle["_checkpoint_path"],
        "preprocessing_profile_id": profile_id,
        "chunking": chunking,
        "class_names": list(class_names),
        "run_signature": signature,
        "files": file_summaries,
        "total_predictions": len(all_rows),
        "read_predictions": str(merged_predictions),
        "report": report,
        "persistent_raw_cache_created": False,
    }
    save_json(output_dir / "classification_summary.json", summary)
    return summary
