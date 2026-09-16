"""Shared plot styles and helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from utils.paths import PROJECT_ROOT, project_relative_path


DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "results" / "time_varying_results.npz"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS.parent / "figures"


def load_result_arrays(results: str | Path | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Read the experiment's NPZ arrays or use an in-memory result."""
    if isinstance(results, dict):
        return results
    with np.load(Path(results).resolve(), allow_pickle=False) as source:
        return {name: source[name] for name in source.files}


TIME_VARYING_PLOT_STYLE = {
    "font.size": 36,
    "axes.labelsize": 46,
    "xtick.labelsize": 39,
    "ytick.labelsize": 39,
    "legend.fontsize": 34,
    "axes.linewidth": 2.0,
}

TIME_VARYING_FIGSIZE = (21.5, 9.0)
TIME_VARYING_SUBPLOT_MARGINS = {
    "left": 0.13,
    "right": 0.96,
    "bottom": 0.20,
    "top": 0.975,
}


def padded_limits(
    values: np.ndarray,
    *,
    pad_fraction: float,
    include_zero: bool = False,
) -> tuple[float, float]:
    """Return data limits with extra space around the range."""
    values = np.asarray(values, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)
    span = upper - lower
    pad = max(span * float(pad_fraction), 1e-5)
    return lower - pad, upper + pad


def save_pdf(
    fig: plt.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    crop_to_content: bool = True,
) -> str:
    """Save a figure as PDF, close it, and return its project-relative path."""
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    path = output_path / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight" if crop_to_content else None)
    plt.close(fig)
    return project_relative_path(path)


def clock_axis(
    axis: plt.Axes,
    timestamps: np.ndarray,
    *,
    timezone_name: str = "America/Los_Angeles",
) -> None:
    """Add local-time ticks through the end of the final sample minute."""
    zone = ZoneInfo(timezone_name)
    local_times = local_datetimes(timestamps, timezone_name=timezone_name)
    axis.set_xlim(local_times[0], local_times[-1] + timedelta(minutes=1))
    axis.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=zone))
    axis.xaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _position: mdates.num2date(value, tz=zone)
            .strftime("%I%p")
            .lstrip("0")
            .lower()
        )
    )
    axis.set_xlabel("Time", labelpad=17)


def local_datetimes(
    timestamps: np.ndarray,
    *,
    timezone_name: str = "America/Los_Angeles",
) -> list[datetime]:
    """Convert UTC timestamps to datetimes in the chosen time zone."""
    zone = ZoneInfo(timezone_name)
    seconds = np.asarray(timestamps).astype("datetime64[s]").astype(np.int64)
    return [datetime.fromtimestamp(int(value), tz=timezone.utc).astimezone(zone) for value in seconds]
