#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from squiggle_species.bonito_pft import read_raw_manifest
from squiggle_species.constants import SPECIES
from squiggle_species.data import mil_collate
from squiggle_species.metrics import classification_summary
from squiggle_species.models import SignalStudent
from squiggle_species.utils import read_json, save_json


class LegacyMatchedDataset(Dataset):
    def __init__(self, records, max_chunks):
        self.records = records
        self.max_chunks = int(max_chunks)
        self.arrays = {}

    def __len__(self):
        return len(self.records)

    def array(self, path):
        if path not in self.arrays:
            self.arrays[path] = np.load(path, mmap_mode="r")
        return self.arrays[path]

    def __getitem__(self, index):
        record = self.records[index]
        if record.raw_n_chunks <= self.max_chunks:
            local = np.arange(record.raw_n_chunks, dtype=np.int64)
        else:
            local = np.linspace(0, record.raw_n_chunks - 1, num=self.max_chunks, dtype=np.int64)
        chunks = np.asarray(
            self.array(record.legacy_chunk_path)[record.legacy_chunk_start + local], dtype=np.float32
        )
        return chunks, record.label, record.read_id


def evaluate(model, loader, device):
    model.eval()
    true, pred, confidence = [], [], []
    with torch.no_grad():
        for x, mask, labels, _ in loader:
            logits, _, _ = model(x.to(device), mask.to(device))
            probs = torch.softmax(logits, dim=1)
            conf, labels_pred = probs.max(dim=1)
            true.append(labels.numpy())
            pred.append(labels_pred.cpu().numpy())
            confidence.append(conf.cpu().numpy())
    return classification_summary(
        np.concatenate(true).astype(np.int64),
        np.concatenate(pred).astype(np.int64),
        np.concatenate(confidence).astype(np.float32),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    experiment = read_json(args.experiment)
    config = experiment["model"]
    records = read_raw_manifest(args.raw_manifest)
    model = SignalStudent(
        input_dim=768,
        hidden_dim=int(config["hidden_dim"]),
        projection_dim=int(config["projection_dim"]),
        attention_dim=int(config["attention_dim"]),
        num_classes=len(SPECIES),
        dropout=float(config["dropout"]),
        transformer_layers=int(config["transformer_layers"]),
        transformer_heads=int(config["transformer_heads"]),
        transformer_ff_dim=int(config["transformer_ff_dim"]),
    ).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {}
    for split in ("val", "test"):
        loader = DataLoader(
            LegacyMatchedDataset(records[split], args.max_chunks),
            batch_size=256,
            shuffle=False,
            num_workers=0,
            collate_fn=mil_collate,
        )
        metrics[split] = evaluate(model, loader, args.device)
    result = {
        "status": "complete",
        "mode": "frozen_ce_matched_chunks",
        "max_chunks": args.max_chunks,
        "split_counts": {split: len(values) for split, values in records.items()},
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
