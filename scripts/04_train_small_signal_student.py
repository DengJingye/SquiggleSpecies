#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from squiggle_species.constants import SPECIES
from squiggle_species.data import ChunkBagDataset, mil_collate, read_bag_manifest
from squiggle_species.metrics import classification_summary
from squiggle_species.models import SignalStudent, cross_modal_distillation_loss
from squiggle_species.utils import file_sha256, read_json, save_json, set_seed, write_csv


class TeacherBank:
    def __init__(self, path: Path):
        data = np.load(path, allow_pickle=False)
        self.read_ids = [str(value) for value in data["read_ids"]]
        self.logits = np.asarray(data["logits"], dtype=np.float32)
        self.embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        self.index = {read_id: index for index, read_id in enumerate(self.read_ids)}

    def batch(self, read_ids, device):
        missing = [read_id for read_id in read_ids if read_id not in self.index]
        if missing:
            raise KeyError(f"Teacher output missing read IDs: {missing[:5]}")
        indices = [self.index[read_id] for read_id in read_ids]
        logits = torch.from_numpy(self.logits[indices]).to(device, non_blocking=True)
        embeddings = torch.from_numpy(self.embeddings[indices]).to(device, non_blocking=True)
        return logits, embeddings


def make_loader(records, max_chunks, training, seed, batch_size, device, num_workers):
    dataset = ChunkBagDataset(records, max_chunks=max_chunks, training=training, seed=seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        collate_fn=mil_collate,
        pin_memory=device.startswith("cuda"),
    )


def evaluate(model, loader, device, species):
    model.eval()
    y_true, y_pred, confidence, read_ids = [], [], [], []
    with torch.no_grad():
        for x, mask, labels, batch_read_ids in loader:
            logits, _, _ = model(x.to(device, non_blocking=True), mask.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            y_true.append(labels.numpy())
            y_pred.append(pred.cpu().numpy())
            confidence.append(conf.cpu().numpy())
            read_ids.extend(batch_read_ids)
    y_true_array = np.concatenate(y_true).astype(np.int64)
    y_pred_array = np.concatenate(y_pred).astype(np.int64)
    confidence_array = np.concatenate(confidence).astype(np.float32)
    return classification_summary(y_true_array, y_pred_array, confidence_array, species), read_ids, y_true_array, y_pred_array, confidence_array


def save_eval(output_dir, split, result, read_ids, y_true, y_pred, confidence, species):
    save_json(output_dir / f"{split}_metrics.json", result)
    prediction_rows = [
        {
            "read_id": read_id,
            "true_label": int(true_label),
            "true_species": species[int(true_label)],
            "pred_label": int(pred_label),
            "pred_species": species[int(pred_label)],
            "confidence": float(conf),
            "correct": int(true_label == pred_label),
        }
        for read_id, true_label, pred_label, conf in zip(read_ids, y_true, y_pred, confidence)
    ]
    write_csv(
        output_dir / f"{split}_predictions.csv",
        prediction_rows,
        ["read_id", "true_label", "true_species", "pred_label", "pred_species", "confidence", "correct"],
    )
    matrix_rows = []
    for true_index, values in enumerate(result["confusion_matrix"]):
        row = {"true_species": species[true_index]}
        row.update({f"pred_{name}": int(values[pred_index]) for pred_index, name in enumerate(species)})
        matrix_rows.append(row)
    write_csv(output_dir / f"{split}_confusion_matrix.csv", matrix_rows, ["true_species"] + [f"pred_{name}" for name in species])
    per_species_rows = [{"species": species, **metrics} for species, metrics in result["per_species"].items()]
    write_csv(output_dir / f"{split}_per_species_metrics.csv", per_species_rows, ["species", "precision", "recall", "f1", "support"])


def main():
    parser = argparse.ArgumentParser(description="Train a small group-held-out signal student with CE or cross-modal KD.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["ce", "kd"], required=True)
    parser.add_argument("--teacher-train", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--aggregation", choices=["mean", "attention", "transformer"], default=None)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    if args.mode == "kd" and args.teacher_train is None:
        raise ValueError("--teacher-train is required for mode=kd")
    experiment = read_json(args.experiment)
    species = tuple(experiment.get("species", SPECIES))
    model_config = dict(experiment["model"])
    if args.aggregation is not None:
        model_config["aggregation"] = args.aggregation
    if args.max_chunks:
        model_config["max_chunks"] = args.max_chunks
    train_config = experiment["training"]
    seed = int(experiment["seed"])
    set_seed(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split = read_bag_manifest(args.group_manifest)
    read_sets = {split: {record.read_id for record in records} for split, records in by_split.items()}
    overlap = (read_sets["train"] & read_sets["val"]) | (read_sets["train"] & read_sets["test"]) | (read_sets["val"] & read_sets["test"])
    group_sets = {split: {(record.barcode, record.ccf_file) for record in records} for split, records in by_split.items()}
    group_overlap = (group_sets["train"] & group_sets["val"]) | (group_sets["train"] & group_sets["test"]) | (group_sets["val"] & group_sets["test"])
    if overlap or group_overlap:
        raise ValueError(f"Split leakage: reads={len(overlap)} groups={len(group_overlap)}")

    max_chunks = int(model_config["max_chunks"])
    train_loader = make_loader(by_split["train"], max_chunks, True, seed, int(train_config["batch_size"]), args.device, args.num_workers)
    val_loader = make_loader(by_split["val"], max_chunks, False, seed, int(train_config["eval_batch_size"]), args.device, args.num_workers)
    test_loader = make_loader(by_split["test"], max_chunks, False, seed, int(train_config["eval_batch_size"]), args.device, args.num_workers)
    teacher = TeacherBank(args.teacher_train) if args.mode == "kd" else None

    model = SignalStudent(
        input_dim=int(model_config["input_dim"]),
        hidden_dim=int(model_config["hidden_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        attention_dim=int(model_config["attention_dim"]),
        num_classes=len(species),
        dropout=float(model_config["dropout"]),
        transformer_layers=int(model_config["transformer_layers"]),
        transformer_heads=int(model_config["transformer_heads"]),
        transformer_ff_dim=int(model_config["transformer_ff_dim"]),
        aggregation=str(model_config.get("aggregation", "transformer")),
    ).to(args.device)
    if args.eval_only_checkpoint is not None:
        checkpoint = torch.load(args.eval_only_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = {}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            result, read_ids, y_true, y_pred, confidence = evaluate(model, loader, args.device, species)
            save_eval(args.output_dir, split, result, read_ids, y_true, y_pred, confidence, species)
            metrics[split] = result
        summary = {
            "status": "complete",
            "mode": "eval_only",
            "source_checkpoint": str(args.eval_only_checkpoint),
            "aggregation": str(model_config.get("aggregation", "transformer")),
            "max_chunks": max_chunks,
            "metrics": metrics,
        }
        save_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config["learning_rate"]), weight_decay=float(train_config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(train_config["epochs"]))
    best_val, best_epoch, bad_epochs = -1.0, -1, 0
    checkpoint_path = args.output_dir / "model.pth"
    log_rows = []
    start_time = time()
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "kd": 0.0, "alignment": 0.0}
        batches = 0
        for x, mask, labels, read_ids in train_loader:
            x = x.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            logits, embeddings, _ = model(x, mask)
            if args.mode == "kd":
                teacher_logits, teacher_embeddings = teacher.batch(read_ids, args.device)
                loss, components = cross_modal_distillation_loss(
                    logits,
                    embeddings,
                    labels,
                    teacher_logits,
                    teacher_embeddings,
                    temperature=float(train_config["distill_temperature"]),
                    label_smoothing=float(train_config["label_smoothing"]),
                    distill_weight=float(train_config["distill_weight"]),
                    alignment_weight=float(train_config["embedding_alignment_weight"]),
                )
            else:
                loss = F.cross_entropy(logits, labels, label_smoothing=float(train_config["label_smoothing"]))
                components = {"ce": loss, "kd": loss.detach() * 0.0, "alignment": loss.detach() * 0.0}
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            totals["loss"] += float(loss.item())
            for key in ("ce", "kd", "alignment"):
                totals[key] += float(components[key].item())
            batches += 1
        scheduler.step()
        val_result, _, _, _, _ = evaluate(model, val_loader, args.device, species)
        row = {
            "epoch": epoch,
            "loss": totals["loss"] / max(1, batches),
            "ce": totals["ce"] / max(1, batches),
            "kd": totals["kd"] / max(1, batches),
            "alignment": totals["alignment"] / max(1, batches),
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "elapsed_sec": time() - start_time,
        }
        log_rows.append(row)
        print(f"{args.mode} epoch={epoch:03d} loss={row['loss']:.4f} val_f1={row['val_macro_f1']:.4f}", flush=True)
        if row["val_macro_f1"] > best_val:
            best_val = row["val_macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": args.mode,
                    "model_config": model_config,
                    "train_config": train_config,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val,
                    "group_manifest_sha256": file_sha256(args.group_manifest),
                    "species": species,
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= int(train_config["patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {}
    eval_loaders = [("val", val_loader)]
    if not args.skip_test:
        eval_loaders.append(("test", test_loader))
    for split, loader in eval_loaders:
        result, read_ids, y_true, y_pred, confidence = evaluate(model, loader, args.device, species)
        save_eval(args.output_dir, split, result, read_ids, y_true, y_pred, confidence, species)
        metrics[split] = result
    summary = {
        "status": "complete",
        "mode": args.mode,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "split_counts": {split: len(records) for split, records in by_split.items()},
        "read_overlap_count": len(overlap),
        "ccf_group_overlap_count": len(group_overlap),
        "runtime_sec": time() - start_time,
        "aggregation": str(model_config.get("aggregation", "transformer")),
        "max_chunks": max_chunks,
        "metrics": metrics,
    }
    save_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "train_log.csv", log_rows, list(log_rows[0].keys()))
    print(
        json.dumps(
            {
                "mode": args.mode,
                "aggregation": str(model_config.get("aggregation", "transformer")),
                "val_macro_f1": metrics["val"]["macro_f1"],
                "test_macro_f1": metrics.get("test", {}).get("macro_f1"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
