"""Plot adaptive controller gains."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.plotting import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RESULTS,
    TIME_VARYING_FIGSIZE,
    TIME_VARYING_PLOT_STYLE,
    TIME_VARYING_SUBPLOT_MARGINS,
    clock_axis,
    load_result_arrays,
    local_datetimes,
    save_pdf,
)
from experiments.time_varying_profiles import LOCAL_TIMEZONE
from utils.paths import project_relative_path


def plot_time_varying_gain_dql_fig(
    results: str | Path | dict[str, np.ndarray] = DEFAULT_RESULTS,
    *,
    output_dir: str | Path | None = None,
    timezone_name: str = LOCAL_TIMEZONE,
    plot_buses: np.ndarray | None = None,
) -> dict[str, Any]:
    """Save a PDF of gain changes for ten selected buses."""
    arrays = load_result_arrays(results)
    output_path = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    mode_names = [str(value) for value in arrays["mode_names"].tolist()]
    block_index = mode_names.index("block")
    timestamps = np.asarray(arrays["timestamps"]).astype("datetime64[ns]")
    local_times = local_datetimes(timestamps, timezone_name=timezone_name)
    gains = np.asarray(arrays["gains"], dtype=float)[block_index, :-1]
    plot_buses = np.asarray(
        arrays["plot_ieee_buses"] if plot_buses is None else plot_buses, dtype=int
    )
    selected = gains[:, plot_buses - 2]
    minimum = float(np.min(selected))
    maximum = float(np.max(selected))
    span = max(maximum - minimum, maximum * 0.05, 1e-6)
    y_limits = (
        max(minimum * 0.65, minimum - 0.15 * span, 1e-6),
        maximum + 0.55 * span,
    )
    colors = plt.colormaps.get_cmap("tab10").resampled(plot_buses.size)

    with plt.rc_context(TIME_VARYING_PLOT_STYLE):
        fig, axis = plt.subplots(figsize=TIME_VARYING_FIGSIZE, dpi=140)
        for line_index, bus in enumerate(plot_buses):
            axis.step(
                local_times,
                selected[:, line_index],
                where="post",
                color=colors(line_index),
                linewidth=2.8,
                alpha=0.96,
                label=f"node{int(bus)}",
            )
        axis.set_ylabel(r"$K$", labelpad=23)
        axis.set_ylim(*y_limits)
        axis.tick_params(axis="x", pad=13)
        axis.tick_params(axis="y", pad=10)
        axis.grid(True, linestyle="--", linewidth=0.95, alpha=0.60)
        clock_axis(axis, timestamps, timezone_name=timezone_name)
        axis.set_xlabel("Time", labelpad=22)
        axis.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=5,
            frameon=True,
            framealpha=0.92,
            columnspacing=1.6,
            handlelength=1.3,
            handletextpad=0.35,
            borderaxespad=0.1,
            borderpad=0.3,
            labelspacing=0.25,
        )
        fig.subplots_adjust(**TIME_VARYING_SUBPLOT_MARGINS)
        artifact = save_pdf(
            fig,
            output_path,
            f"block_gain_Varying_{plot_buses.size}nodes",
            crop_to_content=False,
        )

    return {
        "output_dir": project_relative_path(output_path),
        "plot_buses": plot_buses.tolist(),
        "gain_minimum": minimum,
        "gain_maximum": maximum,
        "y_limits": list(y_limits),
        "artifact": artifact,
    }


def main() -> None:
    """Generate the gain plot from command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timezone", default=LOCAL_TIMEZONE)
    parser.add_argument("--plot-buses", nargs="+", type=int)
    args = parser.parse_args()
    print(
        json.dumps(
            plot_time_varying_gain_dql_fig(
                args.results,
                output_dir=args.output_dir,
                timezone_name=args.timezone,
                plot_buses=(
                    None
                    if args.plot_buses is None
                    else np.asarray(args.plot_buses, dtype=int)
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
