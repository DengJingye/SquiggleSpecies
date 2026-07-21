#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from bonito.util import load_model
from torch.utils.data import DataLoader

from squiggle_species.bonito_pft import (
    BonitoPartialStudent,
    RawChunkBagDataset,
    augment_signal,
    raw_mil_collate,
    read_raw_manifest,
    symmetric_consistency,
)
from squiggle_species.constants import SPECIES
from squiggle_species.metrics import classification_summary
from squiggle_species.models import SignalStudent
from squiggle_species.utils import file_sha256, read_json, save_json, set_seed, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["pft_a", "pft_b"], required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--chunk-microbatch", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    return parser.parse_args()


def make_loader(records, max_chunks, training, seed, batch_size, num_workers, pin_memory):
    dataset = RawChunkBagDataset(records, max_chunks=max_chunks, training=training, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=raw_mil_collate,
        pin_memory=pin_memory,
    )
    return dataset, loader


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, confidence, read_ids = [], [], [], []
    with torch.no_grad():
        for raw, mask, labels, batch_read_ids in loader:
            raw = raw.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits, _, _ = model(raw, mask)
            probs = torch.softmax(logits.float(), dim=1)
            conf, pred = probs.max(dim=1)
            y_true.append(labels.numpy())
            y_pred.append(pred.cpu().numpy())
            confidence.append(conf.cpu().numpy())
            read_ids.extend(batch_read_ids)
    y_true_array = np.concatenate(y_true).astype(np.int64)
    y_pred_array = np.concatenate(y_pred).astype(np.int64)
    confidence_array = np.concatenate(confidence).astype(np.float32)
    return classification_summary(y_true_array, y_pred_array, confidence_array), read_ids, y_true_array, y_pred_array, confidence_array


def save_eval(output_dir, split, result, read_ids, y_true, y_pred, confidence):
    save_json(output_dir / f"{split}_metrics.json", result)
    rows = [
        {
            "read_id": read_id,
            "true_label": int(true_label),
            "true_species": SPECIES[int(true_label)],
            "pred_label": int(pred_label),
            "pred_species": SPECIES[int(pred_label)],
            "confidence": float(conf),
            "correct": int(true_label == pred_label),
        }
        for read_id, true_label, pred_label, conf in zip(read_ids, y_true, y_pred, confidence)
    ]
    write_csv(
        output_dir / f"{split}_predictions.csv",
        rows,
        ["read_id", "true_label", "true_species", "pred_label", "pred_species", "confidence", "correct"],
    )
    matrix_rows = []
    for true_index, values in enumerate(result["confusion_matrix"]):
        row = {"true_species": SPECIES[true_index]}
        row.update({f"pred_{species}": int(values[index]) for index, species in enumerate(SPECIES)})
        matrix_rows.append(row)
    write_csv(
        output_dir / f"{split}_confusion_matrix.csv",
        matrix_rows,
        ["true_species"] + [f"pred_{species}" for species in SPECIES],
    )
    per_species = [{"species": species, **metrics} for species, metrics in result["per_species"].items()]
    write_csv(
        output_dir / f"{split}_per_species_metrics.csv",
        per_species,
        ["species", "precision", "recall", "f1", "support"],
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is required for Bonito partial fine-tuning")
    experiment = read_json(args.experiment)
    seed = int(experiment["seed"])
    set_seed(seed)
    model_config = experiment["model"]
    train_config = dict(experiment["training"])
    epochs = args.epochs or int(train_config["epochs"])
    batch_size = args.batch_size or int(train_config["batch_size"])
    eval_batch_size = args.eval_batch_size or int(train_config["eval_batch_size"])
    max_chunks = args.max_chunks or int(model_config["max_chunks"])
    chunk_microbatch = args.chunk_microbatch or int(train_config["chunk_microbatch"])
    num_workers = int(train_config["num_workers"]) if args.num_workers < 0 else args.num_workers
    trainable_blocks = 1 if args.mode == "pft_a" else 2
    consistency_weight = 0.0 if args.mode == "pft_a" else float(train_config["consistency_weight"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split = read_raw_manifest(args.raw_manifest)
    read_sets = {split: {record.read_id for record in records} for split, records in by_split.items()}
    overlap = (read_sets["train"] & read_sets["val"]) | (read_sets["train"] & read_sets["test"]) | (read_sets["val"] & read_sets["test"])
    group_sets = {split: {(record.barcode, record.ccf_file) for record in records} for split, records in by_split.items()}
    group_overlap = (group_sets["train"] & group_sets["val"]) | (group_sets["train"] & group_sets["test"]) | (group_sets["val"] & group_sets["test"])
    if overlap or group_overlap:
        raise ValueError(f"Split leakage: reads={len(overlap)} groups={len(group_overlap)}")

    train_dataset, train_loader = make_loader(
        by_split["train"], max_chunks, True, seed, batch_size, num_workers, args.device.startswith("cuda")
    )
    _, val_loader = make_loader(
        by_split["val"], max_chunks, False, seed, eval_batch_size, num_workers, args.device.startswith("cuda")
    )
    _, test_loader = make_loader(
        by_split["test"], max_chunks, False, seed, eval_batch_size, num_workers, args.device.startswith("cuda")
    )

    student = SignalStudent(
        input_dim=768,
        hidden_dim=int(model_config["hidden_dim"]),
        projection_dim=int(model_config["projection_dim"]),
        attention_dim=int(model_config["attention_dim"]),
        num_classes=len(SPECIES),
        dropout=float(model_config["dropout"]),
        transformer_layers=int(model_config["transformer_layers"]),
        transformer_heads=int(model_config["transformer_heads"]),
        transformer_ff_dim=int(model_config["transformer_ff_dim"]),
    )
    initial = torch.load(args.initial_student_checkpoint, map_location="cpu", weights_only=False)
    student.load_state_dict(initial["model_state_dict"])
    bonito_model = load_model(str(args.model_dir), device=args.device)
    model = BonitoPartialStudent(
        bonito_model=bonito_model,
        student=student,
        trainable_lstm_blocks=trainable_blocks,
        chunk_microbatch=chunk_microbatch,
    ).to(args.device)
    del bonito_model

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.trainable_backbone_parameters()),
                "lr": float(train_config["backbone_learning_rate"]),
            },
            {"params": model.student.parameters(), "lr": float(train_config["head_learning_rate"])},
        ],
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    best_val, best_epoch, bad_epochs = -1.0, -1, 0
    checkpoint_path = args.output_dir / "model.pth"
    log_rows = []
    started = time()

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        model.train()
        total_loss = total_ce = total_consistency = 0.0
        batches = 0
        for raw, mask, labels, _ in train_loader:
            raw = raw.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                if args.mode == "pft_b":
                    with torch.no_grad():
                        teacher_logits, _, _ = model(raw, mask)
                    model.train()
                    augmented = augment_signal(raw, mask, experiment["augmentation"])
                    logits, _, _ = model(augmented, mask)
                    consistency = symmetric_consistency(logits.float(), teacher_logits.float())
                else:
                    logits, _, _ = model(raw, mask)
                    consistency = logits.float().sum() * 0.0
                ce = F.cross_entropy(
                    logits.float(), labels, label_smoothing=float(train_config["label_smoothing"])
                )
                loss = ce + consistency_weight * consistency
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(train_config["gradient_clip"]),
            )
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item())
            total_ce += float(ce.item())
            total_consistency += float(consistency.item())
            batches += 1
        scheduler.step()
        val_result, _, _, _, _ = evaluate(model, val_loader, args.device)
        row = {
            "epoch": epoch,
            "loss": total_loss / max(1, batches),
            "ce": total_ce / max(1, batches),
            "consistency": total_consistency / max(1, batches),
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "head_lr": optimizer.param_groups[1]["lr"],
            "elapsed_sec": time() - started,
        }
        log_rows.append(row)
        print(
            f"{args.mode} epoch={epoch:03d} loss={row['loss']:.4f} val_f1={row['val_macro_f1']:.4f}",
            flush=True,
        )
        if row["val_macro_f1"] > best_val:
            best_val = row["val_macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": args.mode,
                    "trainable_lstm_blocks": trainable_blocks,
                    "experiment": experiment,
                    "best_epoch": best_epoch,
                    "best_val_macro_f1": best_val,
                    "raw_manifest_sha256": file_sha256(args.raw_manifest),
                    "initial_student_checkpoint": str(args.initial_student_checkpoint),
                    "bonito_model_dir": str(args.model_dir),
                    "species": SPECIES,
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
    for split, loader in (("val", val_loader), ("test", test_loader)):
        result, read_ids, y_true, y_pred, confidence = evaluate(model, loader, args.device)
        save_eval(args.output_dir, split, result, read_ids, y_true, y_pred, confidence)
        metrics[split] = result
    summary = {
        "status": "complete",
        "mode": args.mode,
        "trainable_lstm_blocks": trainable_blocks,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "split_counts": {split: len(records) for split, records in by_split.items()},
        "read_overlap_count": len(overlap),
        "ccf_group_overlap_count": len(group_overlap),
        "max_chunks": max_chunks,
        "batch_size": batch_size,
        "runtime_sec": time() - started,
        "metrics": metrics,
    }
    save_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "train_log.csv", log_rows, list(log_rows[0]))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
