from __future__ import annotations

from pathlib import Path

from .preprocessing import PROFILE_APPLE_SCLAMP_V1, PROFILE_LEGACY_STONE_V1
from .utils import file_sha256, read_json


SUPPORTED_SCHEMA = "1.0"
SUPPORTED_MODEL_FAMILIES = {"bonito_pft"}
SUPPORTED_PROFILES = {PROFILE_LEGACY_STONE_V1, PROFILE_APPLE_SCLAMP_V1}


def _resolve_file(bundle_path: Path, value: str, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = bundle_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Model bundle {field} does not exist: {path}")
    return path


def load_model_bundle(path: str | Path, verify_hashes: bool = True) -> dict:
    bundle_path = Path(path)
    if bundle_path.is_dir():
        bundle_path = bundle_path / "model_bundle.json"
    if not bundle_path.is_file():
        raise FileNotFoundError(f"Model bundle does not exist: {bundle_path}")
    bundle_path = bundle_path.resolve()
    bundle = read_json(bundle_path)

    if str(bundle.get("schema_version")) != SUPPORTED_SCHEMA:
        raise ValueError(
            f"Unsupported model bundle schema: {bundle.get('schema_version')!r}; "
            f"expected {SUPPORTED_SCHEMA!r}"
        )
    family = str(bundle.get("model_family", ""))
    if family not in SUPPORTED_MODEL_FAMILIES:
        raise ValueError(f"Unsupported model_family: {family!r}")

    class_names = bundle.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("Model bundle class_names must contain at least two classes")
    if any(not isinstance(name, str) or not name.strip() for name in class_names):
        raise ValueError("Model bundle class_names must contain non-empty strings")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Model bundle class_names contains duplicates")

    preprocessing = bundle.get("preprocessing", {})
    profile_id = preprocessing.get("profile_id")
    if profile_id not in SUPPORTED_PROFILES:
        raise ValueError(f"Unsupported preprocessing profile in model bundle: {profile_id!r}")
    chunking = bundle.get("chunking", {})
    discard_first = int(chunking.get("discard_first", -1))
    chunk_size = int(chunking.get("chunk_size", 0))
    overlap = int(chunking.get("overlap", -1))
    max_chunks = int(chunking.get("max_chunks", 0))
    if discard_first < 0 or chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"Invalid chunking declaration: {chunking}")
    if max_chunks < 1:
        raise ValueError("Model bundle chunking.max_chunks must be positive")

    checkpoint = bundle.get("checkpoint", {})
    checkpoint_path = _resolve_file(bundle_path, str(checkpoint.get("path", "")), "checkpoint.path")
    expected_sha = str(checkpoint.get("sha256", "")).strip()
    if verify_hashes and expected_sha:
        observed_sha = file_sha256(checkpoint_path)
        if observed_sha != expected_sha:
            raise ValueError(
                f"Checkpoint SHA256 mismatch: expected={expected_sha}, observed={observed_sha}"
            )

    calibration = bundle.get("calibration", {})
    if calibration:
        threshold_enabled = bool(calibration.get("threshold_enabled", False))
        threshold = float(calibration.get("threshold", 0.0))
        if threshold_enabled and not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Invalid calibrated threshold: {threshold}")
        if calibration.get("selected_on") not in {None, "validation"}:
            raise ValueError("Calibration selected_on must be 'validation'")

    resolved = dict(bundle)
    resolved["_bundle_path"] = str(bundle_path)
    resolved["_bundle_sha256"] = file_sha256(bundle_path)
    resolved["_checkpoint_path"] = str(checkpoint_path)
    return resolved


def verify_bonito_weights(bundle: dict, bonito_model_dir: str | Path) -> dict:
    model_dir = Path(bonito_model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Bonito model directory does not exist: {model_dir}")
    declaration = bundle.get("backbone", {}).get("weights", {})
    filename = str(declaration.get("filename", "weights_0.tar"))
    weights = model_dir / filename
    if not weights.is_file():
        raise FileNotFoundError(f"Bonito weights do not exist: {weights}")
    expected_sha = str(declaration.get("sha256", "")).strip()
    observed_sha = file_sha256(weights) if expected_sha else ""
    if expected_sha and observed_sha != expected_sha:
        raise ValueError(
            f"Bonito weights SHA256 mismatch: expected={expected_sha}, observed={observed_sha}"
        )
    return {
        "model_dir": str(model_dir),
        "weights": str(weights),
        "expected_sha256": expected_sha or None,
        "observed_sha256": observed_sha or None,
    }
