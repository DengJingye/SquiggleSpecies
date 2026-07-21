#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
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
)
from squiggle_species.constants import SPECIES
from squiggle_species.group_robust import (
    CrossFilePairBatchSampler,
    ccf_group_accuracy_summary,
    cross_file_supcon_loss,
    group_dro_loss,
)
from squiggle_species.metrics import classification_summary
from squiggle_species.models import SignalStudent
from squiggle_species.utils import file_sha256, read_json, save_json, set_seed, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stone Bonito PFT and DNABERT-S-style objective ablation.")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--initial-student-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=["ce", "supcon", "mixup", "c2lr_mimix", "crossfile_groupdro"],
        required=True,
    )
    parser.add_argument("--trainable-blocks", type=int, required=True, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=0)
    parser.add_argument("--chunk-microbatch", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--hard-pairs", default="")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--eval-only-checkpoint", type=Path, default=None)
    parser.add_argument("--aggregation", choices=["mean", "attention", "transformer"], default=None)
    parser.add_argument("--seed-override", type=int, default=None)
    return parser.parse_args()


def parse_hard_pairs(value: str, species: tuple[str, ...]) -> set[tuple[int, int]]:
    species_to_label = {name: index for index, name in enumerate(species)}
    pairs: set[tuple[int, int]] = set()
    for item in [part.strip() for part in value.split(",") if part.strip()]:
        left, right = [part.strip() for part in item.split(":", 1)]
        a, b = species_to_label[left], species_to_label[right]
        pairs.add((min(a, b), max(a, b)))
    return pairs


def make_loader(records, max_chunks, training, seed, batch_size, num_workers, pin_memory):
    dataset = RawChunkBagDataset(records, max_chunks=max_chunks, training=training, seed=seed)
    return dataset, DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        drop_last=training,
        num_workers=num_workers,
        collate_fn=raw_mil_collate,
        pin_memory=pin_memory,
    )


def weighted_supcon(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    hard_pairs: set[tuple[int, int]],
    hard_negative_weight: float,
) -> torch.Tensor:
    features = F.normalize(features, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    n = features.shape[0]
    non_self = ~torch.eye(n, dtype=torch.bool, device=features.device)
    positive = labels[:, None].eq(labels[None, :]) & non_self
    weights = torch.ones_like(logits)
    if hard_negative_weight > 1.0:
        for left, right in hard_pairs:
            pair_mask = ((labels[:, None] == left) & (labels[None, :] == right)) | (
                (labels[:, None] == right) & (labels[None, :] == left)
            )
            weights = torch.where(pair_mask, torch.full_like(weights, hard_negative_weight), weights)
    denominator = (torch.exp(logits) * weights * non_self).sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_prob = logits - denominator.log()
    count = positive.sum(dim=1)
    valid = count > 0
    if not torch.any(valid):
        return features.sum() * 0.0
    return -((log_prob * positive).sum(dim=1)[valid] / count[valid]).mean()


def imix_contrastive(
    mixed_anchor: torch.Tensor,
    reference: torch.Tensor,
    permutation: torch.Tensor,
    lam: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    mixed_anchor = F.normalize(mixed_anchor, dim=1)
    reference = F.normalize(reference, dim=1)
    log_prob = F.log_softmax(mixed_anchor @ reference.T / temperature, dim=1)
    rows = torch.arange(mixed_anchor.shape[0], device=mixed_anchor.device)
    return -(lam * log_prob[rows, rows] + (1.0 - lam) * log_prob[rows, permutation]).mean()


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


def evaluate(model, loader, device, species):
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
    y_true = np.concatenate(y_true).astype(np.int64)
    y_pred = np.concatenate(y_pred).astype(np.int64)
    confidence = np.concatenate(confidence).astype(np.float32)
    return classification_summary(y_true, y_pred, confidence, species), read_ids, y_true, y_pred, confidence


def save_eval(output_dir, split, result, read_ids, y_true, y_pred, confidence, species, group_by_read=None):
    save_json(output_dir / f"{split}_metrics.json", result)
    rows = []
    for read_id, true_label, pred_label, conf in zip(read_ids, y_true, y_pred, confidence):
        row = {
                "read_id": read_id,
                "true_label": int(true_label),
                "true_species": species[int(true_label)],
                "pred_label": int(pred_label),
                "pred_species": species[int(pred_label)],
                "confidence": float(conf),
                "correct": int(true_label == pred_label),
            }
        if group_by_read is not None:
            row["ccf_file"] = group_by_read[str(read_id)]
        rows.append(row)
    prediction_fields = [
        "read_id", "true_label", "true_species", "pred_label", "pred_species", "confidence", "correct"
    ]
    if group_by_read is not None:
        prediction_fields.append("ccf_file")
    write_csv(
        output_dir / f"{split}_predictions.csv",
        rows,
        prediction_fields,
    )
    matrix_rows = []
    for true_index, values in enumerate(result["confusion_matrix"]):
        row = {"true_species": species[true_index]}
        row.update({f"pred_{name}": int(values[index]) for index, name in enumerate(species)})
        matrix_rows.append(row)
    write_csv(
        output_dir / f"{split}_confusion_matrix.csv",
        matrix_rows,
        ["true_species"] + [f"pred_{name}" for name in species],
    )
    per_species = [{"species": species, **metrics} for species, metrics in result["per_species"].items()]
    write_csv(
        output_dir / f"{split}_per_species_metrics.csv",
        per_species,
        ["species", "precision", "recall", "f1", "support"],
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    experiment = read_json(args.experiment)
    species = tuple(experiment.get("species", SPECIES))
    seed = int(args.seed_override if args.seed_override is not None else experiment["seed"])
    set_seed(seed)
    random.seed(seed)
    model_config = dict(experiment["model"])
    if args.aggregation is not None:
        model_config["aggregation"] = args.aggregation
    train_config = experiment["training"]
    objective_config = experiment["objectives"]
    epochs = args.epochs or int(train_config["epochs"])
    max_chunks = args.max_chunks or int(model_config["max_chunks"])
    batch_size = args.batch_size or int(train_config["batch_size"])
    eval_batch_size = args.eval_batch_size or int(train_config["eval_batch_size"])
    chunk_microbatch = args.chunk_microbatch or int(train_config["chunk_microbatch"])
    num_workers = int(train_config["num_workers"]) if args.num_workers < 0 else args.num_workers
    hard_pairs = parse_hard_pairs(args.hard_pairs, species)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split = read_raw_manifest(args.raw_manifest)
    manifest_species = {record.barcode for records in by_split.values() for record in records}
    if manifest_species != set(species):
        raise ValueError(
            f"Manifest species {sorted(manifest_species)} do not match experiment species {sorted(species)}"
        )
    read_sets = {split: {record.read_id for record in records} for split, records in by_split.items()}
    overlap = (read_sets["train"] & read_sets["val"]) | (read_sets["train"] & read_sets["test"]) | (
        read_sets["val"] & read_sets["test"]
    )
    group_sets = {split: {(record.barcode, record.ccf_file) for record in records} for split, records in by_split.items()}
    group_overlap = (group_sets["train"] & group_sets["val"]) | (group_sets["train"] & group_sets["test"]) | (
        group_sets["val"] & group_sets["test"]
    )
    if overlap or group_overlap:
        raise ValueError(f"Split leakage: reads={len(overlap)} groups={len(group_overlap)}")

    group_by_read = {
        record.read_id: record.ccf_file for records in by_split.values() for record in records
    }
    train_groups = sorted({record.ccf_file for record in by_split["train"]})
    train_group_to_index = {group: index for index, group in enumerate(train_groups)}

    train_sampler = None
    if args.objective == "crossfile_groupdro":
        train_dataset = RawChunkBagDataset(by_split["train"], max_chunks=max_chunks, training=True, seed=seed)
        train_sampler = CrossFilePairBatchSampler(by_split["train"], batch_size=batch_size, seed=seed)
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=num_workers,
            collate_fn=raw_mil_collate,
            pin_memory=args.device.startswith("cuda"),
        )
    else:
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
        num_classes=len(species),
        dropout=float(model_config["dropout"]),
        transformer_layers=int(model_config["transformer_layers"]),
        transformer_heads=int(model_config["transformer_heads"]),
        transformer_ff_dim=int(model_config["transformer_ff_dim"]),
        aggregation=str(model_config["aggregation"]),
    )
    initial = torch.load(args.initial_student_checkpoint, map_location="cpu", weights_only=False)
    initial_state = initial["model_state_dict"]
    if tuple(initial.get("species", SPECIES)) == species:
        student.load_state_dict(initial_state)
    else:
        shared_state = {
            name: value for name, value in initial_state.items() if not name.startswith("classifier.")
        }
        missing, unexpected = student.load_state_dict(shared_state, strict=False)
        if set(missing) != {"classifier.weight", "classifier.bias"} or unexpected:
            raise ValueError(
                f"Cannot reuse initial student for subset classes; missing={missing}, unexpected={unexpected}"
            )
    bonito_model = load_model(str(args.model_dir), device=args.device)
    model = BonitoPartialStudent(
        bonito_model=bonito_model,
        student=student,
        trainable_lstm_blocks=args.trainable_blocks,
        chunk_microbatch=chunk_microbatch,
    ).to(args.device)
    del bonito_model

    if args.eval_only_checkpoint is not None:
        checkpoint = torch.load(args.eval_only_checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        metrics = {}
        for split, loader in (("val", val_loader), ("test", test_loader)):
            result, read_ids, y_true, y_pred, confidence = evaluate(model, loader, args.device, species)
            save_eval(args.output_dir, split, result, read_ids, y_true, y_pred, confidence, species, group_by_read)
            group_summary, group_rows = ccf_group_accuracy_summary(
                read_ids, y_true, y_pred, group_by_read
            )
            save_json(args.output_dir / f"{split}_ccf_group_summary.json", group_summary)
            write_csv(
                args.output_dir / f"{split}_ccf_group_metrics.csv",
                group_rows,
                ["ccf_file", "label", "n_reads", "accuracy"],
            )
            metrics[split] = result
        summary = {
            "status": "complete",
            "mode": "eval_only",
            "source_checkpoint": str(args.eval_only_checkpoint),
            "objective": args.objective,
            "trainable_lstm_blocks": args.trainable_blocks,
            "aggregation": str(model_config["aggregation"]),
            "max_chunks": max_chunks,
            "seed": seed,
            "metrics": metrics,
        }
        save_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return

    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.trainable_backbone_parameters()), "lr": float(train_config["backbone_learning_rate"])},
            {"params": model.student.parameters(), "lr": float(train_config["head_learning_rate"])},
        ],
        weight_decay=float(train_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.device.startswith("cuda"))
    checkpoint_path = args.output_dir / "model.pth"
    best_val, best_epoch, best_stage, bad_epochs = -1.0, -1, "", 0
    log_rows = []
    started = time()
    group_weights = torch.full(
        (len(train_groups),),
        1.0 / max(1, len(train_groups)),
        device=args.device,
        dtype=torch.float32,
    )

    for epoch in range(1, epochs + 1):
        train_dataset.set_epoch(epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "contrast": 0.0, "mix": 0.0, "group_dro": 0.0}
        batches = 0
        stage_two = args.objective == "c2lr_mimix" and epoch > int(objective_config["curriculum_warmup_epochs"])
        curriculum_fraction = min(1.0, epoch / max(1, int(objective_config["curriculum_warmup_epochs"])))
        hard_weight = 1.0 + curriculum_fraction * (float(objective_config["hard_negative_weight"]) - 1.0)
        for raw, mask, labels, batch_read_ids in train_loader:
            raw = raw.to(args.device, non_blocking=True)
            mask = mask.to(args.device, non_blocking=True)
            labels = labels.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.device.startswith("cuda")):
                group_dro_component = raw.sum() * 0.0
                if stage_two:
                    reference_raw = augment_signal(raw, mask, experiment["augmentation"])
                    reference_logits, reference_embedding, _ = model(reference_raw, mask)
                    permutation = torch.randperm(labels.shape[0], device=args.device)
                    beta = torch.distributions.Beta(
                        float(objective_config["mixup_alpha"]), float(objective_config["mixup_alpha"])
                    )
                    lam = beta.sample((labels.shape[0],)).to(args.device)
                    lam = torch.maximum(lam, 1.0 - lam)
                    mix_layer = random.randint(model.train_start, 8)
                    mixed_logits, mixed_embedding, _, _ = model.forward_manifold_mix(
                        raw, mask, permutation, lam, mix_layer
                    )
                    contrast = imix_contrastive(
                        mixed_embedding,
                        reference_embedding,
                        permutation,
                        lam,
                        float(objective_config["temperature"]),
                    )
                    onehot = F.one_hot(labels, num_classes=len(species)).float()
                    mixed_targets = lam[:, None] * onehot + (1.0 - lam[:, None]) * onehot[permutation]
                    mix_loss = soft_cross_entropy(mixed_logits.float(), mixed_targets)
                    ce = F.cross_entropy(
                        reference_logits.float(), labels, label_smoothing=float(train_config["label_smoothing"])
                    )
                    loss = (
                        float(objective_config["ce_weight"]) * ce
                        + float(objective_config["mimix_weight"]) * contrast
                        + float(objective_config["mixup_weight"]) * mix_loss
                    )
                else:
                    logits, embedding, _ = model(raw, mask)
                    ce = F.cross_entropy(logits.float(), labels, label_smoothing=float(train_config["label_smoothing"]))
                    contrast = ce.detach() * 0.0
                    mix_loss = ce.detach() * 0.0
                    if args.objective == "crossfile_groupdro":
                        augmented = augment_signal(raw, mask, experiment["augmentation"])
                        logits_two, embedding_two, _ = model(augmented, mask)
                        per_sample_ce = 0.5 * (
                            F.cross_entropy(
                                logits.float(),
                                labels,
                                label_smoothing=float(train_config["label_smoothing"]),
                                reduction="none",
                            )
                            + F.cross_entropy(
                                logits_two.float(),
                                labels,
                                label_smoothing=float(train_config["label_smoothing"]),
                                reduction="none",
                            )
                        )
                        ce = per_sample_ce.mean()
                        batch_groups = torch.tensor(
                            [train_group_to_index[group_by_read[read_id]] for read_id in batch_read_ids],
                            dtype=torch.long,
                            device=args.device,
                        )
                        group_dro_component, _ = group_dro_loss(
                            per_sample_ce,
                            batch_groups,
                            group_weights,
                            float(objective_config["group_dro_step_size"]),
                        )
                        features = torch.cat([embedding, embedding_two], dim=0)
                        doubled_labels = torch.cat([labels, labels], dim=0)
                        doubled_groups = torch.cat([batch_groups, batch_groups], dim=0)
                        contrast = cross_file_supcon_loss(
                            features,
                            doubled_labels,
                            doubled_groups,
                            float(objective_config["temperature"]),
                        )
                        dro_weight = float(objective_config["group_dro_weight"])
                        loss = (
                            (1.0 - dro_weight) * ce
                            + dro_weight * group_dro_component
                            + float(objective_config["cross_file_supcon_weight"]) * contrast
                        )
                    elif args.objective in {"supcon", "c2lr_mimix"}:
                        augmented = augment_signal(raw, mask, experiment["augmentation"])
                        logits_two, embedding_two, _ = model(augmented, mask)
                        features = torch.cat([embedding, embedding_two], dim=0)
                        doubled_labels = torch.cat([labels, labels], dim=0)
                        contrast = weighted_supcon(
                            features,
                            doubled_labels,
                            float(objective_config["temperature"]),
                            hard_pairs,
                            hard_weight,
                        )
                        ce = 0.5 * (
                            ce
                            + F.cross_entropy(
                                logits_two.float(), labels, label_smoothing=float(train_config["label_smoothing"])
                            )
                        )
                        loss = ce + float(objective_config["supcon_weight"]) * contrast
                    elif args.objective == "mixup":
                        permutation = torch.randperm(labels.shape[0], device=args.device)
                        beta = torch.distributions.Beta(
                            float(objective_config["mixup_alpha"]), float(objective_config["mixup_alpha"])
                        )
                        lam = beta.sample((labels.shape[0], 1)).to(args.device)
                        lam = torch.maximum(lam, 1.0 - lam)
                        mixed_embedding = F.normalize(
                            lam * embedding + (1.0 - lam) * embedding[permutation], dim=1
                        )
                        onehot = F.one_hot(labels, num_classes=len(species)).float()
                        targets = lam * onehot + (1.0 - lam) * onehot[permutation]
                        mix_loss = soft_cross_entropy(model.student.classifier(mixed_embedding).float(), targets)
                        loss = ce + float(objective_config["mixup_weight"]) * mix_loss
                    else:
                        loss = ce
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(train_config["gradient_clip"]),
            )
            scaler.step(optimizer)
            scaler.update()
            totals["loss"] += float(loss.item())
            totals["ce"] += float(ce.item())
            totals["contrast"] += float(contrast.item())
            totals["mix"] += float(mix_loss.item())
            totals["group_dro"] += float(group_dro_component.item())
            batches += 1
        scheduler.step()
        val_result, val_read_ids, val_true, val_pred, _ = evaluate(model, val_loader, args.device, species)
        val_group_summary, _ = ccf_group_accuracy_summary(
            val_read_ids, val_true, val_pred, group_by_read
        )
        row = {
            "epoch": epoch,
            "stage": "mimix" if stage_two else "base",
            "loss": totals["loss"] / max(1, batches),
            "ce": totals["ce"] / max(1, batches),
            "contrast": totals["contrast"] / max(1, batches),
            "mix": totals["mix"] / max(1, batches),
            "group_dro": totals["group_dro"] / max(1, batches),
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "val_worst_group_accuracy": val_group_summary["worst_group_accuracy"],
            "val_p10_group_accuracy": val_group_summary["p10_group_accuracy"],
            "elapsed_sec": time() - started,
        }
        log_rows.append(row)
        print(
            f"blocks={args.trainable_blocks} objective={args.objective} epoch={epoch:03d} "
            f"stage={row['stage']} loss={row['loss']:.4f} val_f1={row['val_macro_f1']:.4f} "
            f"val_p10_group={row['val_p10_group_accuracy']:.4f}",
            flush=True,
        )
        if row["val_macro_f1"] > best_val:
            best_val, best_epoch, best_stage, bad_epochs = row["val_macro_f1"], epoch, row["stage"], 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "objective": args.objective,
                    "trainable_lstm_blocks": args.trainable_blocks,
                    "experiment": experiment,
                    "best_epoch": best_epoch,
                    "best_stage": best_stage,
                    "best_val_macro_f1": best_val,
                    "seed": seed,
                    "raw_manifest_sha256": file_sha256(args.raw_manifest),
                    "initial_student_checkpoint": str(args.initial_student_checkpoint),
                    "bonito_model_dir": str(args.model_dir),
                    "hard_pairs": args.hard_pairs,
                    "species": species,
                    "group_dro_weights": group_weights.detach().cpu(),
                },
                checkpoint_path,
            )
        else:
            bad_epochs += 1
            minimum_curriculum_epoch = int(objective_config["curriculum_warmup_epochs"]) + int(train_config["patience"])
            if bad_epochs >= int(train_config["patience"]) and (
                args.objective != "c2lr_mimix" or epoch >= minimum_curriculum_epoch
            ):
                break

    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = {}
    eval_loaders = [("val", val_loader)]
    if args.evaluate_test:
        eval_loaders.append(("test", test_loader))
    for split, loader in eval_loaders:
        result, read_ids, y_true, y_pred, confidence = evaluate(model, loader, args.device, species)
        save_eval(args.output_dir, split, result, read_ids, y_true, y_pred, confidence, species, group_by_read)
        group_summary, group_rows = ccf_group_accuracy_summary(read_ids, y_true, y_pred, group_by_read)
        save_json(args.output_dir / f"{split}_ccf_group_summary.json", group_summary)
        write_csv(
            args.output_dir / f"{split}_ccf_group_metrics.csv",
            group_rows,
            ["ccf_file", "label", "n_reads", "accuracy"],
        )
        metrics[split] = result
        metrics[f"{split}_ccf_groups"] = group_summary
    summary = {
        "status": "complete",
        "objective": args.objective,
        "trainable_lstm_blocks": args.trainable_blocks,
        "aggregation": str(model_config["aggregation"]),
        "best_epoch": best_epoch,
        "best_stage": best_stage,
        "best_val_macro_f1": best_val,
        "seed": seed,
        "split_counts": {split: len(records) for split, records in by_split.items()},
        "read_overlap_count": len(overlap),
        "ccf_group_overlap_count": len(group_overlap),
        "max_chunks": max_chunks,
        "batch_size": batch_size,
        "hard_pairs": args.hard_pairs,
        "group_robust_config": objective_config if args.objective == "crossfile_groupdro" else None,
        "runtime_sec": time() - started,
        "metrics": metrics,
    }
    save_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "train_log.csv", log_rows, list(log_rows[0]))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
