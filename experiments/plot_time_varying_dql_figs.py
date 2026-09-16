"""Plot controller voltages and reactive-power actions."""

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
    padded_limits,
    save_pdf,
)
from experiments.time_varying_profiles import LOCAL_TIMEZONE
from utils.paths import project_relative_path


PLOT_MODES = ("nolearning", "block")
VOLTAGE_LIMITS = (0.95, 1.05)
VOLTAGE_Y_MAX = 1.099
VOLTAGE_TICKS = np.arange(0.94, 1.081, 0.02)


def plot_time_varying_dql_figs(
    results: str | Path | dict[str, np.ndarray] = DEFAULT_RESULTS,
    *,
    output_dir: str | Path | None = None,
    timezone_name: str = LOCAL_TIMEZONE,
) -> dict[str, Any]:
    """Save voltage and action PDFs for the fixed-gain and adaptive controllers."""
    arrays = load_result_arrays(results)
    output_path = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    mode_names = [str(value) for value in arrays["mode_names"].tolist()]
    mode_indices = [mode_names.index(mode) for mode in PLOT_MODES]
    timestamps = np.asarray(arrays["timestamps"]).astype("datetime64[ns]")
    local_times = local_datetimes(timestamps, timezone_name=timezone_name)
    plot_buses = np.asarray(arrays["plot_ieee_buses"], dtype=int)
    voltage = np.asarray(arrays["voltage"], dtype=float)[mode_indices][
        :, :, plot_buses - 1
    ]
    action = np.asarray(arrays["control"], dtype=float)[mode_indices][
        :, :, plot_buses - 2
    ]
    voltage_auto_limits = padded_limits(voltage, pad_fraction=0.075)
    voltage_limits = (
        min(voltage_auto_limits[0], VOLTAGE_LIMITS[0] - 0.01),
        max(voltage_auto_limits[1], VOLTAGE_Y_MAX),
    )
    action_limits = padded_limits(action, pad_fraction=0.06, include_zero=True)
    colors = plt.colormaps.get_cmap("tab10").resampled(plot_buses.size)
    artifacts: dict[str, str] = {}

    for mode_index, mode in enumerate(PLOT_MODES):
        with plt.rc_context(TIME_VARYING_PLOT_STYLE):
            fig, axis = plt.subplots(figsize=TIME_VARYING_FIGSIZE, dpi=140)
            for line_index, bus in enumerate(plot_buses):
                axis.plot(
                    local_times,
                    voltage[mode_index, :, line_index],
                    color=colors(line_index),
                    linewidth=2.6,
                    alpha=0.96,
                    label=f"node{int(bus)}",
                )
            for limit in VOLTAGE_LIMITS:
                axis.axhline(
                    limit,
                    color="#707070",
                    linestyle="--",
                    linewidth=1.45,
                    alpha=0.75,
                    zorder=1.5,
                )
            axis.set_ylabel("v (p.u.)", labelpad=23)
            axis.set_ylim(*voltage_limits)
            axis.set_yticks(VOLTAGE_TICKS)
            axis.grid(True, linestyle="--", linewidth=0.95, alpha=0.60)
            clock_axis(axis, timestamps, timezone_name=timezone_name)
            axis.set_xlabel("Time", labelpad=22)
            axis.tick_params(axis="x", pad=12)
            axis.tick_params(axis="y", pad=10)
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
            artifacts[f"{mode}_voltage"] = save_pdf(
                fig,
                output_path,
                f"{mode}_voltage_Varying_{plot_buses.size}nodes",
                crop_to_content=False,
            )

        with plt.rc_context(TIME_VARYING_PLOT_STYLE):
            fig, axis = plt.subplots(figsize=TIME_VARYING_FIGSIZE, dpi=140)
            for line_index in range(plot_buses.size):
                axis.plot(
                    local_times,
                    action[mode_index, :, line_index],
                    color=colors(line_index),
                    linewidth=2.6,
                    alpha=0.96,
                )
            axis.set_ylabel("q (p.u.)", labelpad=23)
            axis.set_ylim(*action_limits)
            axis.grid(True, linestyle="--", linewidth=0.95, alpha=0.60)
            clock_axis(axis, timestamps, timezone_name=timezone_name)
            axis.set_xlabel("Time", labelpad=22)
            axis.tick_params(axis="x", pad=12)
            axis.tick_params(axis="y", pad=10)
            fig.subplots_adjust(**TIME_VARYING_SUBPLOT_MARGINS)
            artifacts[f"{mode}_action"] = save_pdf(
                fig,
                output_path,
                f"{mode}_action_Varying_{plot_buses.size}nodes",
                crop_to_content=False,
            )

    return {
        "output_dir": project_relative_path(output_path),
        "plot_buses": plot_buses.tolist(),
        "shared_y_limits": {
            "voltage": list(voltage_limits),
            "action": list(action_limits),
        },
        "artifacts": artifacts,
    }


def main() -> None:
    """Generate voltage and action plots from command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--timezone", default=LOCAL_TIMEZONE)
    args = parser.parse_args()
    print(
        json.dumps(
            plot_time_varying_dql_figs(
                args.results,
                output_dir=args.output_dir,
                timezone_name=args.timezone,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
