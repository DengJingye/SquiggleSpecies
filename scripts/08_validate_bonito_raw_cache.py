#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from bonito.util import load_model

from squiggle_species.bonito_pft import read_raw_manifest


def normalize_output(output: torch.Tensor, batch: int) -> torch.Tensor:
    if output.shape[1] == batch and output.shape[2] == 768:
        return output.permute(1, 0, 2)
    if output.shape[0] == batch and output.shape[2] == 768:
        return output
    raise ValueError(f"Unexpected Bonito output: {tuple(output.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunks-per-species", type=int, default=8)
    args = parser.parse_args()

    model = load_model(str(args.model_dir), device=args.device)
    encoder = nn.Sequential(*list(model.encoder.children())[:9]).to(args.device).half().eval()
    by_split = read_raw_manifest(args.raw_manifest)
    selected = []
    seen = set()
    for split in ("train", "val", "test"):
        for record in by_split[split]:
            if record.barcode not in seen:
                selected.append(record)
                seen.add(record.barcode)
    cosines, max_errors = [], []
    details = []
    with torch.no_grad():
        for record in selected:
            raw_array = np.load(record.raw_chunk_path, mmap_mode="r")
            legacy_array = np.load(record.legacy_chunk_path, mmap_mode="r")
            n = min(record.raw_n_chunks, args.chunks_per_species)
            raw = torch.as_tensor(
                np.asarray(raw_array[record.raw_chunk_start : record.raw_chunk_start + n], dtype=np.float32),
                device=args.device,
            ).half()
            output = encoder(raw.unsqueeze(1))
            current = normalize_output(output, n).mean(dim=1).float()
            legacy = torch.as_tensor(
                np.asarray(legacy_array[record.legacy_chunk_start : record.legacy_chunk_start + n], dtype=np.float32),
                device=args.device,
            )
            cosine = F.cosine_similarity(current, legacy, dim=1)
            max_error = (current - legacy).abs().amax(dim=1)
            cosines.extend(cosine.cpu().tolist())
            max_errors.extend(max_error.cpu().tolist())
            details.append(
                {
                    "barcode": record.barcode,
                    "read_id": record.read_id,
                    "n_chunks": n,
                    "mean_cosine": float(cosine.mean().item()),
                    "max_abs_error": float(max_error.max().item()),
                }
            )
    result = {
        "status": "pass" if min(cosines) >= 0.999 else "fail",
        "n_chunks": len(cosines),
        "minimum_cosine": min(cosines),
        "mean_cosine": float(np.mean(cosines)),
        "maximum_abs_error": max(max_errors),
        "details": details,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
