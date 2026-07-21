#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from squiggle_species.constants import SPECIES, SPLITS
from squiggle_species.metrics import classification_summary
from squiggle_species.models import KmerTeacher
from squiggle_species.utils import read_json, save_json, set_seed


class KmerDataset(Dataset):
    def __init__(self, root: Path, split: str):
        self.x = np.load(root / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_y.npy", mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return torch.from_numpy(np.asarray(self.x[index], dtype=np.float32)), int(self.y[index]), index


def evaluate(model, loader, device):
    model.eval()
    labels, predictions, confidence = [], [], []
    with torch.no_grad():
        for x, y, _ in loader:
            logits, _ = model(x.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            labels.append(y.numpy())
            predictions.append(pred.cpu().numpy())
            confidence.append(conf.cpu().numpy())
    return classification_summary(np.concatenate(labels), np.concatenate(predictions), np.concatenate(confidence))


def extract_matched_cache(resources, group_manifest: Path, output_dir: Path):
    source_root = Path(resources["sequence_kmer_cache"])
    cache_summary = read_json(source_root / "sequence_kmer_cache_summary.json")
    input_dim = int(cache_summary["kmer_dim"])
    records = {split: [] for split in SPLITS}
    wanted = {}
    with group_manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            new_index = len(records[split])
            record = {
                "new_index": new_index,
                "read_id": row["read_id"],
                "label": int(row["label"]),
                "barcode": row["barcode"],
                "ccf_file": row["ccf_file"],
            }
            records[split].append(record)
            wanted[row["read_id"]] = (split, new_index, int(row["label"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    filled = {}
    for split in SPLITS:
        arrays[split] = np.lib.format.open_memmap(
            output_dir / f"{split}_x.npy", mode="w+", dtype=np.float16, shape=(len(records[split]), input_dim)
        )
        filled[split] = np.zeros(len(records[split]), dtype=bool)
        np.save(output_dir / f"{split}_y.npy", np.asarray([row["label"] for row in records[split]], dtype=np.int64))
        np.save(output_dir / f"{split}_read_ids.npy", np.asarray([row["read_id"] for row in records[split]], dtype="U96"))

    found = set()
    for source_split in ("train", "atlas", "val", "test"):
        source_manifest = source_root / "manifests" / f"{source_split}_sequence_manifest.csv"
        matched = []
        with source_manifest.open(newline="") as handle:
            for source_index, row in enumerate(csv.DictReader(handle)):
                target = wanted.get(row["read_id"])
                if target is not None:
                    matched.append((source_index, row["read_id"], target))
        if not matched:
            continue
        source_x = np.load(source_root / f"{source_split}_x.npy", mmap_mode="r")
        source_y = np.load(source_root / f"{source_split}_y.npy", mmap_mode="r")
        for start in range(0, len(matched), 1024):
            block = matched[start : start + 1024]
            source_indices = np.asarray([row[0] for row in block], dtype=np.int64)
            block_x = np.asarray(source_x[source_indices], dtype=np.float16)
            for local_index, (source_index, read_id, (new_split, new_index, label)) in enumerate(block):
                if int(source_y[source_index]) != label:
                    raise ValueError(f"Sequence/signal label mismatch for {read_id}")
                arrays[new_split][new_index] = block_x[local_index]
                filled[new_split][new_index] = True
                found.add(read_id)

    missing = sorted(set(wanted) - found)
    for array in arrays.values():
        array.flush()
    if missing:
        raise ValueError(f"{len(missing)} selected signal reads lack sequence k-mer cache; examples={missing[:10]}")
    if any(not mask.all() for mask in filled.values()):
        raise ValueError("At least one matched k-mer output row was not filled")
    return input_dim, {split: len(rows) for split, rows in records.items()}


def export_teacher_outputs(model, data_root, output_dir, device, batch_size, num_workers):
    result = {}
    for split in SPLITS:
        dataset = KmerDataset(data_root, split)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.startswith("cuda"))
        logits_rows, embedding_rows, labels = [], [], []
        model.eval()
        with torch.no_grad():
            for x, y, _ in loader:
                logits, embedding = model(x.to(device, non_blocking=True))
                logits_rows.append(logits.cpu().numpy().astype(np.float32))
                embedding_rows.append(embedding.cpu().numpy().astype(np.float16))
                labels.append(y.numpy())
        logits = np.vstack(logits_rows)
        embeddings = np.vstack(embedding_rows)
        y_true = np.concatenate(labels).astype(np.int64)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        metrics = classification_summary(y_true, np.argmax(probs, axis=1), np.max(probs, axis=1))
        np.savez_compressed(
            output_dir / f"teacher_{split}.npz",
            read_ids=np.load(data_root / f"{split}_read_ids.npy"),
            labels=y_true,
            logits=logits,
            embeddings=embeddings,
        )
        result[split] = metrics
    return result


def main():
    parser = argparse.ArgumentParser(description="Train a leakage-safe sequence teacher on the new CCF-group split.")
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--group-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    resources = read_json(args.resources)
    experiment = read_json(args.experiment)
    seed = int(experiment["seed"])
    set_seed(seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.output_dir / "matched_kmer_cache"
    input_dim, split_counts = extract_matched_cache(resources, args.group_manifest, cache_dir)

    model = KmerTeacher(input_dim=input_dim, num_classes=len(SPECIES)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_loader = DataLoader(KmerDataset(cache_dir, "train"), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
    val_loader = DataLoader(KmerDataset(cache_dir, "val"), batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"))
    best_val, best_epoch, bad_epochs = -1.0, -1, 0
    train_log = []
    start_time = time()
    checkpoint_path = args.output_dir / "sequence_teacher.pth"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for x, y, _ in train_loader:
            x = x.to(args.device, non_blocking=True)
            y = y.to(args.device, non_blocking=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=0.03)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        val_metrics = evaluate(model, val_loader, args.device)
        row = {"epoch": epoch, "train_loss": total_loss / max(1, batches), "val_macro_f1": val_metrics["macro_f1"], "elapsed_sec": time() - start_time}
        train_log.append(row)
        print(f"teacher epoch={epoch:03d} loss={row['train_loss']:.4f} val_f1={row['val_macro_f1']:.4f}", flush=True)
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "input_dim": input_dim, "best_epoch": best_epoch, "best_val_macro_f1": best_val}, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = export_teacher_outputs(model, cache_dir, args.output_dir, args.device, args.eval_batch_size, args.num_workers)
    save_json(args.output_dir / "teacher_metrics.json", {"best_epoch": best_epoch, "best_val_macro_f1": best_val, "split_counts": split_counts, "metrics": metrics})
    save_json(args.output_dir / "train_log.json", train_log)
    print(json.dumps({"best_epoch": best_epoch, "best_val_macro_f1": best_val, "test_macro_f1": metrics["test"]["macro_f1"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
