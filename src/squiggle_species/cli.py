from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from . import __version__
from .utils import read_json, save_json, write_csv


HELP = """Squiggle Species command line interface.

Common workflows:
  squiggle-species inventory-ccf /data/signals -o inventory.json
  squiggle-species cache-ccf /data/signals --profile legacy-stone-v1 --output-dir raw_cache
  squiggle-species classify-ccf /data/signals --model-bundle model_bundle.json --bonito-model-dir models --output-dir results --device cuda:0
  squiggle-species validate-manifest split_manifest.csv -o audit.json
  squiggle-species predict --manifest bag_manifest.csv --checkpoint model.pth --config experiment.json --output predictions.csv
  squiggle-species predict-raw-cache --manifest raw_manifest.csv --checkpoint model.pth --bonito-model-dir models --output predictions.csv
  squiggle-species calibrate --predictions val_predictions.csv --output-dir calibration
  squiggle-species report --predictions test_predictions.csv --threshold-json calibration/calibration.json --output-dir report

Threshold policy:
  A reliability target and coverage constraints are declared before calibration.
  The threshold is selected on validation predictions and then frozen for test
  or external samples. Test and real mixed samples are never used for calibration.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="squiggle-species", description="Nanopore signal microbial classification toolkit.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect-config", help="Validate and print a resource configuration.")
    inspect_parser.add_argument("config", type=Path)

    inventory_parser = subparsers.add_parser("inventory-ccf", help="Inventory CCF5/BLOW5/SLOW5 input files.")
    inventory_parser.add_argument("input", type=Path)
    inventory_parser.add_argument("-o", "--output", type=Path)

    cache_parser = subparsers.add_parser(
        "cache-ccf",
        help="Build a resumable standardized raw-chunk cache from CCF5 files.",
    )
    cache_parser.add_argument("input", type=Path)
    cache_parser.add_argument("--output-dir", type=Path, required=True)
    cache_parser.add_argument(
        "--profile",
        choices=["legacy-stone-v1", "apple-sclamp-v1"],
        required=True,
    )
    cache_parser.add_argument("--discard-first", type=int, default=5000)
    cache_parser.add_argument("--chunk-size", type=int, default=6000)
    cache_parser.add_argument("--overlap", type=int, default=3000)
    cache_parser.add_argument("--reads-per-part", type=int, default=1000)
    cache_parser.add_argument("--max-reads", type=int, default=0)

    classify_parser = subparsers.add_parser(
        "classify-ccf",
        help="Stream CCF5 reads through a frozen model bundle and build a report.",
    )
    classify_parser.add_argument("input", type=Path)
    classify_parser.add_argument("--model-bundle", type=Path, required=True)
    classify_parser.add_argument("--bonito-model-dir", type=Path, required=True)
    classify_parser.add_argument("--output-dir", type=Path, required=True)
    classify_parser.add_argument("--device", default="cpu")
    classify_parser.add_argument("--batch-size", type=int)
    classify_parser.add_argument("--max-reads", type=int, default=0)
    classify_parser.add_argument(
        "--skip-hash-verification",
        action="store_true",
        help="Development only; do not use for formal benchmark or external inference.",
    )

    bundle_parser = subparsers.add_parser(
        "inspect-model-bundle",
        help="Validate model metadata, preprocessing compatibility and file hashes.",
    )
    bundle_parser.add_argument("model_bundle", type=Path)
    bundle_parser.add_argument("--bonito-model-dir", type=Path)

    manifest_parser = subparsers.add_parser("validate-manifest", help="Audit read and CCF-file split leakage.")
    manifest_parser.add_argument("manifest", type=Path)
    manifest_parser.add_argument("-o", "--output", type=Path)

    predict_parser = subparsers.add_parser("predict", help="Predict from a prepared Bonito chunk-embedding cache.")
    predict_parser.add_argument("--manifest", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--config", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    predict_parser.add_argument("--device", default="cpu")
    predict_parser.add_argument("--batch-size", type=int, default=128)
    predict_parser.add_argument("--max-chunks", type=int)

    raw_predict_parser = subparsers.add_parser(
        "predict-raw-cache", help="Predict standardized raw chunk bags with a Bonito partial-finetune checkpoint."
    )
    raw_predict_parser.add_argument("--manifest", type=Path, required=True)
    raw_predict_parser.add_argument("--checkpoint", type=Path, required=True)
    raw_predict_parser.add_argument("--bonito-model-dir", type=Path, required=True)
    raw_predict_parser.add_argument("--output", type=Path, required=True)
    raw_predict_parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    raw_predict_parser.add_argument("--device", default="cpu")
    raw_predict_parser.add_argument("--batch-size", type=int)
    raw_predict_parser.add_argument("--max-chunks", type=int)
    raw_predict_parser.add_argument("--chunk-microbatch", type=int)
    raw_predict_parser.add_argument(
        "--expected-preprocessing-profile",
        help="Reject a raw cache whose declared preprocessing profile does not match this value.",
    )

    calibrate_parser = subparsers.add_parser("calibrate", help="Select a confidence threshold on validation predictions.")
    calibrate_parser.add_argument("--predictions", type=Path, required=True)
    calibrate_parser.add_argument("--output-dir", type=Path, required=True)
    calibrate_parser.add_argument("--min-coverage", type=float, default=0.5)
    calibrate_parser.add_argument("--target-accuracy", type=float, default=0.90)
    calibrate_parser.add_argument("--min-per-class-coverage", type=float, default=0.5)
    calibrate_parser.add_argument("--min-accuracy-gain", type=float, default=0.01)
    calibrate_parser.add_argument("--grid-size", type=int, default=1001)

    report_parser = subparsers.add_parser("report", help="Generate metrics, abundance tables and figures.")
    report_parser.add_argument("--predictions", type=Path, required=True)
    report_parser.add_argument("--output-dir", type=Path, required=True)
    threshold = report_parser.add_mutually_exclusive_group()
    threshold.add_argument("--threshold", type=float, default=0.0)
    threshold.add_argument("--threshold-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if not args_list or args_list[0] in {"-h", "--help"}:
        print(HELP.rstrip())
        return 0
    parser = build_parser()
    args = parser.parse_args(args_list)

    if args.command == "inspect-config":
        resources = read_json(args.config)
        result = {key: {"path": value, "exists": Path(value).exists()} for key, value in resources.items()}
    elif args.command == "inventory-ccf":
        from .inventory import inventory_ccf5

        result = inventory_ccf5(args.input)
        if args.output:
            save_json(args.output, result)
    elif args.command == "cache-ccf":
        from .ccf import cache_ccf5

        result = cache_ccf5(
            args.input,
            args.output_dir,
            profile_id=args.profile,
            discard_first=args.discard_first,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            reads_per_part=args.reads_per_part,
            max_reads=args.max_reads,
        )
    elif args.command == "classify-ccf":
        from .pipeline import classify_ccf5

        result = classify_ccf5(
            args.input,
            args.model_bundle,
            args.bonito_model_dir,
            args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            max_reads=args.max_reads,
            verify_hashes=not args.skip_hash_verification,
        )
    elif args.command == "inspect-model-bundle":
        from .model_bundle import load_model_bundle, verify_bonito_weights

        bundle = load_model_bundle(args.model_bundle)
        result = {
            "status": "valid",
            "schema_version": bundle["schema_version"],
            "model_family": bundle["model_family"],
            "class_names": bundle["class_names"],
            "preprocessing": bundle["preprocessing"],
            "chunking": bundle["chunking"],
            "checkpoint": bundle["_checkpoint_path"],
            "bundle_sha256": bundle["_bundle_sha256"],
        }
        if args.bonito_model_dir:
            result["bonito_weights"] = verify_bonito_weights(
                bundle,
                args.bonito_model_dir,
            )
    elif args.command == "validate-manifest":
        from .manifest import audit_manifest

        result = audit_manifest(args.manifest)
        if args.output:
            save_json(args.output, result)
    elif args.command == "predict":
        from .inference import predict_embedding_bags

        experiment = read_json(args.config)
        model = experiment["model"]
        model_config = {
            "input_dim": int(experiment.get("signal", {}).get("feature_dim", 768)),
            "hidden_dim": int(model["hidden_dim"]),
            "projection_dim": int(model["projection_dim"]),
            "attention_dim": int(model["attention_dim"]),
            "num_classes": len(experiment["species"]),
            "dropout": float(model["dropout"]),
            "transformer_layers": int(model["transformer_layers"]),
            "transformer_heads": int(model["transformer_heads"]),
            "transformer_ff_dim": int(model["transformer_ff_dim"]),
        }
        result = predict_embedding_bags(
            args.manifest,
            args.checkpoint,
            args.output,
            args.split,
            model_config,
            args.device,
            args.batch_size,
            args.max_chunks or int(model["max_chunks"]),
        )
    elif args.command == "calibrate":
        from .calibration import calibrate_threshold, plot_calibration_curve, read_prediction_rows

        summary, curve = calibrate_threshold(
            read_prediction_rows(args.predictions),
            min_coverage=args.min_coverage,
            target_accuracy=args.target_accuracy,
            min_per_class_coverage=args.min_per_class_coverage,
            min_accuracy_gain=args.min_accuracy_gain,
            grid_size=args.grid_size,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(args.output_dir / "calibration.json", summary)
        write_csv(args.output_dir / "calibration_curve.csv", curve, list(curve[0]))
        plot_calibration_curve(curve, args.output_dir / "calibration_curve.png", summary)
        result = summary
    elif args.command == "predict-raw-cache":
        from .inference import predict_raw_bags

        result = predict_raw_bags(
            manifest=args.manifest,
            checkpoint=args.checkpoint,
            bonito_model_dir=args.bonito_model_dir,
            output=args.output,
            split=args.split,
            device=args.device,
            batch_size=args.batch_size,
            max_chunks=args.max_chunks,
            chunk_microbatch=args.chunk_microbatch,
            expected_preprocessing_profile=args.expected_preprocessing_profile,
        )
    elif args.command == "report":
        from .reporting import build_report, load_predictions

        threshold = args.threshold
        threshold_enabled = threshold > 0
        calibration = None
        if args.threshold_json:
            calibration = read_json(args.threshold_json)
            threshold_enabled = bool(calibration.get("threshold_enabled", True))
            threshold = float(calibration["threshold"]) if threshold_enabled else 0.0
        result = build_report(
            load_predictions(args.predictions),
            args.output_dir,
            threshold,
            threshold_enabled=threshold_enabled,
            calibration=calibration,
        )
    else:
        parser.error("a command is required")
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
