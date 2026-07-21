from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import ChunkBagDataset, mil_collate, read_bag_manifest
from .models import SignalStudent


def predict_embedding_bags(
    manifest: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    split: str,
    model_config: dict,
    device: str = "cpu",
    batch_size: int = 128,
    max_chunks: int = 64,
) -> dict:
    records = read_bag_manifest(manifest).get(split, [])
    if not records:
        raise ValueError(f"No records for split={split} in {manifest}")
    dataset = ChunkBagDataset(records, max_chunks=max_chunks, training=False, seed=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=mil_collate)
    model = SignalStudent(**model_config).to(device)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model_state", payload.get("model", payload)) if isinstance(payload, dict) else payload
    model.load_state_dict(state)
    model.eval()
    rows = []
    with torch.no_grad():
        for chunks, mask, labels, read_ids in loader:
            logits, _, _ = model(chunks.to(device), mask.to(device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            confidence = probabilities.max(axis=1)
            for row_index, read_id in enumerate(read_ids):
                row = {
                    "read_id": read_id,
                    "true_label": int(labels[row_index]),
                    "predicted_label": int(predictions[row_index]),
                    "confidence": float(confidence[row_index]),
                }
                for label in range(probabilities.shape[1]):
                    row[f"prob_{label}"] = float(probabilities[row_index, label])
                rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"status": "complete", "split": split, "reads": len(rows), "output": str(output)}


def predict_raw_bags(
    manifest: str | Path,
    checkpoint: str | Path,
    bonito_model_dir: str | Path,
    output: str | Path,
    split: str = "test",
    device: str = "cpu",
    batch_size: int | None = None,
    max_chunks: int | None = None,
    chunk_microbatch: int | None = None,
    expected_preprocessing_profile: str | None = None,
) -> dict:
    """Run a partially fine-tuned Bonito checkpoint on standardized raw bags."""
    from .bonito_pft import BonitoPartialStudent, RawChunkBagDataset, raw_mil_collate, read_raw_manifest

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {"model_state_dict", "experiment", "trainable_lstm_blocks"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint is not a Bonito PFT checkpoint; missing keys: {sorted(missing)}")
    experiment = payload["experiment"]
    species = tuple(payload.get("species", experiment.get("species", ())))
    if not species:
        raise ValueError("Checkpoint does not declare species ordering")
    model_config = experiment["model"]
    training_config = experiment["training"]
    max_chunks = int(max_chunks or model_config["max_chunks"])
    batch_size = int(batch_size or training_config.get("eval_batch_size", 6))
    chunk_microbatch = int(chunk_microbatch or training_config.get("chunk_microbatch", 8))

    records = read_raw_manifest(manifest).get(split, [])
    if not records:
        raise ValueError(f"No records for split={split} in {manifest}")
    declared_profiles = {record.preprocessing_profile_id for record in records if record.preprocessing_profile_id}
    if len(declared_profiles) > 1:
        raise ValueError(f"Manifest mixes preprocessing profiles: {sorted(declared_profiles)}")
    declared_profile = next(iter(declared_profiles), None)
    if expected_preprocessing_profile and declared_profile != expected_preprocessing_profile:
        raise ValueError(
            "Preprocessing profile mismatch: "
            f"checkpoint/runner expects {expected_preprocessing_profile!r}, manifest declares {declared_profile!r}"
        )
    if any(record.label < 0 or record.label >= len(species) for record in records):
        raise ValueError("Manifest contains a label outside the checkpoint species range")
    dataset = RawChunkBagDataset(records, max_chunks=max_chunks, training=False, seed=0)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=raw_mil_collate,
        pin_memory=device.startswith("cuda"),
    )
    student = SignalStudent(
        input_dim=768,
        hidden_dim=int(model_config["hidden_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        attention_dim=int(model_config["attention_dim"]),
        num_classes=len(species),
        dropout=float(model_config["dropout"]),
        transformer_layers=int(model_config["transformer_layers"]),
        transformer_heads=int(model_config["transformer_heads"]),
        transformer_ff_dim=int(model_config["transformer_ff_dim"]),
        aggregation=str(model_config.get("aggregation", "transformer")),
    )
    from bonito.util import load_model

    bonito_model = load_model(str(bonito_model_dir), device=device)
    model = BonitoPartialStudent(
        bonito_model=bonito_model,
        student=student,
        trainable_lstm_blocks=int(payload["trainable_lstm_blocks"]),
        chunk_microbatch=chunk_microbatch,
    ).to(device)
    del bonito_model
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    rows = []
    with torch.no_grad():
        for raw, mask, labels, read_ids in loader:
            raw = raw.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits, _, _ = model(raw, mask)
            probabilities = torch.softmax(logits.float(), dim=1).cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            confidence = probabilities.max(axis=1)
            for row_index, read_id in enumerate(read_ids):
                true_label = int(labels[row_index])
                predicted_label = int(predictions[row_index])
                row = {
                    "read_id": read_id,
                    "true_label": true_label,
                    "true_species": species[true_label],
                    "predicted_label": predicted_label,
                    "predicted_species": species[predicted_label],
                    "confidence": float(confidence[row_index]),
                }
                for label, species_name in enumerate(species):
                    row[f"prob_{label}"] = float(probabilities[row_index, label])
                    row[f"prob_{species_name}"] = float(probabilities[row_index, label])
                rows.append(row)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "status": "complete",
        "mode": "bonito_partial_finetune_raw_cache",
        "split": split,
        "reads": len(rows),
        "species": list(species),
        "max_chunks": max_chunks,
        "preprocessing_profile_id": declared_profile,
        "output": str(output),
    }
