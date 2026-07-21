#!/usr/bin/env python3
"""Create a fresh Bonito 768D cache from a provenance-tracked raw chunk cache."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from bonito.util import load_model
from numpy.lib.format import open_memmap


SPECIES = ["LB01", "LB06", "LB07", "LB08", "LB09", "LB12", "LB18", "LB11", "LB02"]
BAG_FIELDS = [
    "split",
    "split_order",
    "barcode",
    "label",
    "read_id",
    "part_id",
    "part_base",
    "part_read_index",
    "chunk_start",
    "n_chunks",
    "chunk_path",
    "ccf_file",
    "ccf_read_index",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--repair-provenance", action="store_true")
    return parser.parse_args()


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def source_provenance(raw_manifest):
    summary_path = raw_manifest.parent / "cache_summary.json"
    if not summary_path.exists():
        return {"raw_manifest": str(raw_manifest.resolve()), "description": "unknown raw-cache provenance"}
    summary = json.loads(summary_path.read_text())
    species = summary.get("species", [])
    first = species[0] if species else {}
    strategy = first.get("signal_strategy", "unknown")
    module = first.get("signal_module_file", "unknown")
    return {
        "raw_manifest": str(raw_manifest.resolve()),
        "signal_strategy": strategy,
        "signal_module_file": module,
        "description": f"{strategy} via {module}",
    }


def normalize_output(output, batch):
    if output.ndim != 3:
        raise ValueError(f"Expected 3D Bonito output, got {tuple(output.shape)}")
    if output.shape[1] == batch and output.shape[2] == 768:
        return output.permute(1, 0, 2)
    if output.shape[0] == batch and output.shape[2] == 768:
        return output
    raise ValueError(f"Unexpected Bonito output shape: {tuple(output.shape)}")


def merge(args):
    all_rows = []
    metas = []
    for barcode in SPECIES:
        species_dir = args.output_dir / barcode
        meta_path = species_dir / "cache_meta.json"
        manifest_path = species_dir / f"{barcode}_bag_manifest.csv"
        if not meta_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"Incomplete fresh Bonito cache for {barcode}")
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "complete":
            raise ValueError(f"Incomplete fresh Bonito cache meta: {meta_path}")
        metas.append(meta)
        all_rows.extend(read_rows(manifest_path))
    all_rows.sort(key=lambda row: (SPECIES.index(row["barcode"]), row["split"], int(row["split_order"])))
    merged = args.output_dir / "bag_manifest.csv"
    write_csv(merged, all_rows, BAG_FIELDS)
    summary = {
        "status": "complete",
        "raw_manifest": str(args.raw_manifest.resolve()),
        "bag_manifest": str(merged),
        "n_reads": len(all_rows),
        "n_chunks": sum(int(row["n_chunks"]) for row in all_rows),
        "validation": "shape, row-count and finite-value audit; no classification threshold is selected here",
        "source_standardization": source_provenance(args.raw_manifest),
        "species": metas,
    }
    atomic_json(args.output_dir / "cache_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance(args.raw_manifest)
    if args.repair_provenance:
        for barcode in SPECIES:
            meta_path = args.output_dir / barcode / "cache_meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(meta_path)
            meta = json.loads(meta_path.read_text())
            meta["source_standardization"] = provenance
            atomic_json(meta_path, meta)
        merge(args)
        return
    if args.merge_only:
        merge(args)
        return
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("Invalid shard-index")
    rows = read_rows(args.raw_manifest)
    by_species = defaultdict(list)
    for row in rows:
        by_species[row["barcode"]].append(row)
    selected_species = [barcode for index, barcode in enumerate(SPECIES) if index % args.num_shards == args.shard_index]

    model = load_model(str(args.model_dir), device=args.device)
    encoder = nn.Sequential(*list(model.encoder.children())[:9]).to(args.device).half().eval()
    with torch.no_grad():
        for barcode in selected_species:
            species_rows = by_species[barcode]
            species_dir = args.output_dir / barcode
            species_dir.mkdir(parents=True, exist_ok=True)
            output_path = species_dir / f"{barcode}_chunk_embed.npy"
            manifest_path = species_dir / f"{barcode}_bag_manifest.csv"
            meta_path = species_dir / "cache_meta.json"
            if meta_path.exists() and manifest_path.exists() and output_path.exists():
                meta = json.loads(meta_path.read_text())
                if meta.get("status") == "complete" and int(meta.get("n_reads", -1)) == len(species_rows):
                    print(f"[{barcode}] valid existing fresh cache", flush=True)
                    continue
            raw_paths = {row["raw_chunk_path"] for row in species_rows}
            if len(raw_paths) != 1:
                raise ValueError(f"Expected one raw memmap for {barcode}, got {raw_paths}")
            raw = np.load(next(iter(raw_paths)), mmap_mode="r")
            total_chunks = raw.shape[0]
            output = open_memmap(output_path, mode="w+", dtype=np.float16, shape=(total_chunks, 768))
            for start in range(0, total_chunks, args.batch_size):
                end = min(start + args.batch_size, total_chunks)
                batch = torch.as_tensor(np.asarray(raw[start:end], dtype=np.float32), device=args.device).half()
                encoded = normalize_output(encoder(batch.unsqueeze(1)), end - start).mean(dim=1).float()
                if not torch.isfinite(encoded).all():
                    raise FloatingPointError(f"Non-finite Bonito embedding in {barcode} chunks {start}:{end}")
                output[start:end] = encoded.cpu().numpy().astype(np.float16, copy=False)
                if end == total_chunks or end % (args.batch_size * 50) == 0:
                    print(f"[{barcode}] embedded {end}/{total_chunks}", flush=True)
            output.flush()
            bag_rows = []
            for row in species_rows:
                item = {field: row.get(field, "") for field in BAG_FIELDS}
                item["chunk_path"] = str(output_path)
                item["chunk_start"] = row["raw_chunk_start"]
                item["n_chunks"] = row["raw_n_chunks"]
                bag_rows.append(item)
            write_csv(manifest_path, bag_rows, BAG_FIELDS)
            meta = {
                "status": "complete",
                "barcode": barcode,
                "n_reads": len(species_rows),
                "n_chunks": total_chunks,
                "dtype": "float16",
                "shape": [total_chunks, 768],
                "chunk_path": str(output_path),
                "manifest_path": str(manifest_path),
                "source_standardization": provenance,
            }
            atomic_json(meta_path, meta)
            print(f"[{barcode}] fresh Bonito cache complete ({provenance['description']})", flush=True)


if __name__ == "__main__":
    main()
