"""Summarize voltage violations, deviations, and actions for two controllers."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.paths import PROJECT_ROOT, project_relative_path


DEFAULT_RESULTS = PROJECT_ROOT / "outputs" / "results" / "time_varying_results.npz"
COMPARED_MODES = ("nolearning", "block")
MODE_LABELS = {
    "nolearning": "Linear control",
    "block": "Decentralized gain adaptation",
}


def compute_summary(
    results_path: str | Path,
    *,
    voltage_reference: float = 1.0,
    lower_limit: float = 0.95,
    upper_limit: float = 1.05,
) -> dict[str, Any]:
    """Summarize non-slack voltage/action (p.u.) and violations (node-minutes and affected minutes)."""

    results_path = Path(results_path).resolve()
    if not np.isfinite([voltage_reference, lower_limit, upper_limit]).all():
        raise ValueError("Voltage reference and limits must be finite.")
    if not lower_limit < upper_limit:
        raise ValueError("lower_limit must be less than upper_limit.")

    with np.load(results_path, allow_pickle=False) as source:
        mode_names = [str(value) for value in source["mode_names"].tolist()]
        voltage = np.asarray(source["voltage"], dtype=float)
        control = np.asarray(source["control"], dtype=float)

    records: list[dict[str, Any]] = []
    for mode in COMPARED_MODES:
        mode_index = mode_names.index(mode)
        non_slack_voltage = voltage[mode_index, :, 1:]
        mode_control = control[mode_index]
        absolute_deviation = np.abs(non_slack_voltage - voltage_reference)
        undervoltage = non_slack_voltage < lower_limit
        overvoltage = non_slack_voltage > upper_limit
        violation = undervoltage | overvoltage
        records.append(
            {
                "mode": mode,
                "controller": MODE_LABELS[mode],
                "undervoltage_node_minutes": int(np.count_nonzero(undervoltage)),
                "overvoltage_node_minutes": int(np.count_nonzero(overvoltage)),
                "violation_node_minutes": int(np.count_nonzero(violation)),
                "violation_minutes": int(
                    np.count_nonzero(np.any(violation, axis=1))
                ),
                "mean_voltage_deviation_pu": float(np.mean(absolute_deviation)),
                "mean_absolute_action_pu": float(np.mean(np.abs(mode_control))),
            }
        )

    linear_deviation = records[0]["mean_voltage_deviation_pu"]
    adaptive_deviation = records[1]["mean_voltage_deviation_pu"]
    if linear_deviation <= 0.0:
        raise ValueError("Linear-control mean voltage deviation must be positive.")
    reduction_percent = 100.0 * (
        linear_deviation - adaptive_deviation
    ) / linear_deviation
    linear_action = records[0]["mean_absolute_action_pu"]
    adaptive_action = records[1]["mean_absolute_action_pu"]
    if linear_action <= 0.0:
        raise ValueError("Linear-control mean absolute action must be positive.")
    action_increase_percent = 100.0 * (
        adaptive_action - linear_action
    ) / linear_action
    return {
        "results_path": project_relative_path(results_path),
        "voltage_reference": float(voltage_reference),
        "lower_limit": float(lower_limit),
        "upper_limit": float(upper_limit),
        "samples": int(voltage.shape[1]),
        "non_slack_buses": 32,
        "records": records,
        "adaptive_voltage_deviation_reduction_percent": float(
            reduction_percent
        ),
        "adaptive_action_magnitude_increase_percent": float(
            action_increase_percent
        ),
    }


def save_csv(
    summary: dict[str, Any],
    output_dir: str | Path,
) -> str:
    """Save the compact controller comparison as one CSV table."""

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "controller_metrics.csv"
    fieldnames = [
        "controller",
        "undervoltage_node_minutes",
        "overvoltage_node_minutes",
        "violation_node_minutes",
        "violation_minutes",
        "mean_voltage_deviation_pu",
        "mean_absolute_action_pu",
        "adaptive_voltage_deviation_reduction_percent",
        "adaptive_action_magnitude_increase_percent",
    ]
    csv_records = []
    for record in summary["records"]:
        csv_record = dict(record)
        if record["mode"] == "block":
            csv_record["adaptive_voltage_deviation_reduction_percent"] = (
                summary["adaptive_voltage_deviation_reduction_percent"]
            )
            csv_record["adaptive_action_magnitude_increase_percent"] = (
                summary["adaptive_action_magnitude_increase_percent"]
            )
        csv_records.append(csv_record)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_records)
    return project_relative_path(csv_path)


def main() -> None:
    """Parse arguments, summarize controller metrics, and write the comparison CSV."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--voltage-reference", type=float, default=1.0)
    parser.add_argument("--lower-limit", type=float, default=0.95)
    parser.add_argument("--upper-limit", type=float, default=1.05)
    args = parser.parse_args()

    summary = compute_summary(
        args.results,
        voltage_reference=args.voltage_reference,
        lower_limit=args.lower_limit,
        upper_limit=args.upper_limit,
    )
    output_dir = args.output_dir or args.results.resolve().parent
    csv_path = save_csv(summary, output_dir)
    print(
        "Adaptive voltage-deviation reduction: "
        f"{summary['adaptive_voltage_deviation_reduction_percent']:.2f}%"
    )
    print(
        "Adaptive action-magnitude increase: "
        f"{summary['adaptive_action_magnitude_increase_percent']:.2f}%"
    )
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
