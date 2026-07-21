from __future__ import annotations

from typing import Mapping

import numpy as np


PROFILE_LEGACY_STONE_V1 = "legacy-stone-v1"
PROFILE_APPLE_SCLAMP_V1 = "apple-sclamp-v1"


def restore_physical_signal(read: Mapping) -> np.ndarray:
    """Restore picoampere-like signal values from a pyccf5 read record."""
    raw = np.asarray(read["signal"])
    if "lvdsmid" in read:
        return ((raw.astype(np.float32) - float(read["lvdsmid"])) * float(read["unit"])).astype(np.float32)
    signal = float(read["K"]) * float(read["scale"]) * (
        raw.astype(np.uint16).astype(np.float32) + float(read["offset"])
    ) + float(read["B"])
    return signal.astype(np.float32, copy=False)


def legacy_stone_v1(signal: np.ndarray) -> np.ndarray:
    """Exact Median/MAD profile used to train the current Zymo9 PFT model.

    This intentionally has no smooth clamp. Adding the newer shared sclamp
    changes the model input distribution and therefore defines a different
    preprocessing profile.
    """
    signal = np.asarray(signal)
    if signal.size == 0:
        return np.asarray([], dtype=np.float32)
    median = np.median(signal)
    mad = max(float(1.4826 * np.median(np.abs(signal - median))), 1.0)
    return ((signal - median) / mad).astype(np.float32)


def _repair_range(signal: np.ndarray, minimum: float = 1.0, maximum: float = 220.0) -> np.ndarray:
    cleaned = np.asarray(signal, dtype=np.float32)
    if cleaned.size == 0:
        return cleaned
    invalid = np.flatnonzero((cleaned < minimum) | (cleaned > maximum))
    if invalid.size == 0:
        return cleaned
    cleaned = cleaned.copy()
    for index in invalid:
        if index == 0:
            cleaned[index] = maximum if cleaned[index] > maximum else minimum
        else:
            cleaned[index] = cleaned[index - 1]
    return cleaned


def _remove_spikes(signal: np.ndarray, window_size: int = 6000, threshold: float = 5.0) -> np.ndarray:
    from scipy.ndimage import median_filter

    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    local_median = median_filter(signal, size=window_size, mode="reflect")
    residual = signal - local_median
    scale = max(float(1.4826 * np.median(np.abs(residual))), 1.0)
    spikes = np.flatnonzero(np.abs(residual) > threshold * scale)
    if spikes.size == 0:
        return signal.copy()
    cleaned = signal.copy()
    for index in spikes:
        cleaned[index] = local_median[index] if index == 0 else cleaned[index - 1]
    return cleaned


def _central_mad_normalize(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size == 0:
        return signal
    median = np.median(signal)
    residual = signal - median
    lower, upper = np.quantile(residual, [0.01, 0.99])
    central = residual[(residual >= lower) & (residual <= upper)]
    scale = max(float(1.4826 * np.median(np.abs(central))), 1.0)
    return (residual / scale).astype(np.float32)


def _smooth_clamp(signal: np.ndarray, linear_bound: float = 5.0, target_bound: float = 6.0) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    delta = target_bound - linear_bound
    if signal.size == 0 or linear_bound <= 0 or delta <= 0:
        return signal
    output = signal.copy()
    upper = output > linear_bound
    lower = output < -linear_bound
    output[upper] = linear_bound + delta * np.tanh((output[upper] - linear_bound) / delta)
    output[lower] = -linear_bound + delta * np.tanh((output[lower] + linear_bound) / delta)
    return output.astype(np.float32, copy=False)


def apple_sclamp_v1(signal: np.ndarray) -> np.ndarray:
    """Latest Apple pipeline used by the foundation-model experiments."""
    from scipy.signal import medfilt

    repaired = _repair_range(signal)
    despiked = _remove_spikes(repaired, window_size=6000, threshold=5.0)
    normalized = _central_mad_normalize(despiked)
    filtered = medfilt(normalized, kernel_size=5).astype(np.float32)
    return _smooth_clamp(filtered, linear_bound=5.0, target_bound=6.0)


def process_signal(signal: np.ndarray, profile_id: str) -> np.ndarray:
    profiles = {
        PROFILE_LEGACY_STONE_V1: legacy_stone_v1,
        PROFILE_APPLE_SCLAMP_V1: apple_sclamp_v1,
    }
    try:
        function = profiles[profile_id]
    except KeyError as error:
        raise ValueError(f"Unsupported preprocessing profile: {profile_id}") from error
    return function(signal)
