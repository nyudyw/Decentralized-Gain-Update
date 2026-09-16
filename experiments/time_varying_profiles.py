"""Convert real power profiles to IEEE-33 bus injections with fixed settings."""

from __future__ import annotations

import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
LOCAL_TIMEZONE = "America/Los_Angeles"
START_HOUR = 9
END_HOUR = 21

PROFILE_GAIN = 21.0
RAMP_AMPLITUDE = 1.8
SHAPE_CLIP = 0.22
SMOOTHING_WINDOW = 15
WARMUP_MINUTES = 60


def normalize_and_aggregate_profiles(
    raw_p_profiles: np.ndarray,
    profile_mix_weights: np.ndarray,
    normalization: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Clip and normalize source profiles, then combine them for 32 buses."""

    raw = np.asarray(raw_p_profiles, dtype=float)
    weights = np.asarray(profile_mix_weights, dtype=float)
    lower = np.asarray(normalization["winsor_lower"], dtype=float)
    upper = np.asarray(normalization["winsor_upper"], dtype=float)
    centers = np.asarray(normalization["centers"], dtype=float)
    ranges = np.asarray(normalization["robust_ranges"], dtype=float)
    if np.any(ranges <= 0.0):
        raise ValueError("robust_ranges must be positive.")

    clipped = np.clip(raw, lower[:, None], upper[:, None])
    normalized = (clipped - centers[:, None]) / ranges[:, None]
    return normalized, weights @ normalized


def _local_window_mask(timestamps: np.ndarray, *, date: str) -> np.ndarray:
    """Select UTC timestamps from 9am inclusive to 9pm exclusive in Los Angeles."""

    zone = ZoneInfo(LOCAL_TIMEZONE)
    day = datetime.fromisoformat(date).date()
    start = datetime.combine(day, time(hour=START_HOUR), tzinfo=zone)
    end = datetime.combine(day, time(hour=END_HOUR), tzinfo=zone)
    start_utc = np.datetime64(
        start.astimezone(timezone.utc).replace(tzinfo=None), "ns"
    )
    end_utc = np.datetime64(
        end.astimezone(timezone.utc).replace(tzinfo=None), "ns"
    )
    values = np.asarray(timestamps).astype("datetime64[ns]")
    return (values >= start_utc) & (values < end_utc)


def load_profile_data(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    date: str | None = None,
) -> dict[str, Any]:
    """Load the 9am-9pm profiles, inferring the local date unless one is supplied."""

    source_dir = Path(data_root).resolve() / "inputs"
    npz_path = source_dir / "real_data_feeder_inputs.npz"
    manifest_path = source_dir / "real_data_feeder_inputs_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(npz_path, allow_pickle=False) as source:
        timestamps_all = source["timestamps"].astype("datetime64[ns]")
        raw_all = np.asarray(source["raw_p_profiles"], dtype=float)
        weights = np.asarray(source["profile_mix_weights"], dtype=float)
        profile_ids = np.asarray(source["profile_ids"]).astype(str)

    if date is None:
        zone = ZoneInfo(LOCAL_TIMEZONE)
        local_dates = {
            datetime.fromtimestamp(int(value), tz=timezone.utc).astimezone(zone).date()
            for value in timestamps_all.astype("datetime64[s]").astype(np.int64)
        }
        if len(local_dates) != 1:
            raise ValueError("Input spans multiple local dates; select one with --date.")
        date = next(iter(local_dates)).isoformat()

    normalized_all, aggregated_all = normalize_and_aggregate_profiles(
        raw_all,
        weights,
        manifest["p_profile_normalization"],
    )
    mask = _local_window_mask(timestamps_all, date=date)
    expected_samples = (END_HOUR - START_HOUR) * 60
    selected_samples = np.count_nonzero(mask)
    if selected_samples != expected_samples:
        raise ValueError(
            f"{date} contains {selected_samples} selected samples; "
            f"expected {expected_samples}."
        )
    aggregated = aggregated_all[:, mask]
    centered = aggregated - np.mean(aggregated, axis=1, keepdims=True)
    return {
        "date": date,
        "timestamps": timestamps_all[mask],
        "raw_p_profiles": raw_all[:, mask],
        "normalized_p_profiles": normalized_all[:, mask],
        "profile_mix_weights": weights,
        "profile_ids": profile_ids,
        "aggregated_p_shape": aggregated,
        "centered_p_shape": centered,
        "source_npz": npz_path,
        "source_manifest": manifest_path,
    }


def smoothstep(num_samples: int) -> np.ndarray:
    """Create a smooth ramp from zero to one, or zero for a single sample."""

    if num_samples == 1:
        return np.zeros(1, dtype=float)
    phase = np.arange(num_samples, dtype=float) / (num_samples - 1)
    return phase**2 * (3.0 - 2.0 * phase)


def smooth_centered_shape(
    centered_shape: np.ndarray,
    *,
    window: int,
) -> np.ndarray:
    """Smooth each bus profile with a centered moving average and remove its mean."""

    shape = np.asarray(centered_shape, dtype=float)
    radius = window // 2
    padded = np.pad(shape, ((0, 0), (radius, radius)), mode="edge")
    kernel = np.full(window, 1.0 / window)
    smoothed = np.vstack(
        [np.convolve(row, kernel, mode="valid") for row in padded]
    )
    return smoothed - np.mean(smoothed, axis=1, keepdims=True)


def stress_amplitude_envelope(
    num_samples: int,
    *,
    warmup_minutes: int,
) -> np.ndarray:
    """Ramp profile strength during warmup, using one sample per minute."""

    envelope = np.ones(num_samples, dtype=float)
    warmup_samples = min(warmup_minutes, num_samples)
    if warmup_samples:
        envelope[:warmup_samples] = smoothstep(warmup_samples)
    return envelope


def map_profiles_to_ieee33(
    p_ieee_injection: np.ndarray,
    centered_shape: np.ndarray,
    *,
    profile_gain: float = PROFILE_GAIN,
    ramp_amplitude: float = RAMP_AMPLITUDE,
    shape_clip: float = SHAPE_CLIP,
    smoothing_window: int = SMOOTHING_WINDOW,
    warmup_minutes: int = WARMUP_MINUTES,
) -> dict[str, np.ndarray]:
    """Smooth, clip, and scale profiles into IEEE-33 power loads and injections."""

    p_ieee_load = -np.asarray(p_ieee_injection, dtype=float).reshape(-1)
    shape = np.asarray(centered_shape, dtype=float)
    smoothed_shape = smooth_centered_shape(shape, window=smoothing_window)
    bounded_shape = np.clip(smoothed_shape, -shape_clip, shape_clip)
    amplitude_envelope = stress_amplitude_envelope(
        shape.shape[1],
        warmup_minutes=warmup_minutes,
    )
    effective_shape = bounded_shape * amplitude_envelope[None, :]
    envelope = smoothstep(shape.shape[1])
    multiplier = (
        profile_gain * effective_shape
        + ramp_amplitude * envelope[None, :]
    )
    p_load = p_ieee_load[:, None] * multiplier
    return {
        "p_ieee_load": p_ieee_load,
        "temporally_smoothed_p_shape": smoothed_shape,
        "bounded_p_shape": bounded_shape,
        "stress_amplitude_envelope": amplitude_envelope,
        "effective_p_shape": effective_shape,
        "smoothstep_envelope": envelope,
        "p_multiplier": multiplier,
        "p_load": p_load,
        "p_injection": -p_load,
    }
