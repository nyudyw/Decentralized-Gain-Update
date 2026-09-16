"""Compare voltage controllers using time-varying input data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_common import (
    GAIN_MAX,
    GAIN_MIN,
    MODE_NAMES,
    build_context,
    generate_diagonal_gains,
    make_controller,
    voltage_action_metrics,
)
from utils.paths import project_relative_path
from experiments.time_varying_profiles import (
    DEFAULT_DATA_ROOT,
    LOCAL_TIMEZONE,
    PROFILE_GAIN,
    RAMP_AMPLITUDE,
    SHAPE_CLIP,
    SMOOTHING_WINDOW,
    WARMUP_MINUTES,
    load_profile_data,
    map_profiles_to_ieee33,
)


DEFAULT_GAIN_SEED = 588
DEFAULT_GAIN_LOW = 0.006
DEFAULT_GAIN_HIGH = 0.018
DEFAULT_T = 5
DEFAULT_ETA = 0.3
DEFAULT_LAMBDA = 0.05
PLOT_NODE_SEED = 5
PLOT_NODE_COUNT = 10


def sample_plot_buses(
    *,
    seed: int = PLOT_NODE_SEED,
    node_count: int = PLOT_NODE_COUNT,
) -> np.ndarray:
    """Randomly select distinct non-slack buses for plotting."""

    candidates = np.arange(2, 34, dtype=int)
    generator = np.random.default_rng(seed)
    return np.sort(generator.choice(candidates, size=node_count, replace=False))


def _run_mode(
    mode: str,
    *,
    p_injection: np.ndarray,
    q_base: np.ndarray,
    initial_gains: np.ndarray,
    T: int,
    eta: float,
    gain_regularization: float,
) -> dict[str, Any]:
    """Simulate one controller mode and record its voltage, action, and gains."""
    context = build_context()
    controller = make_controller(
        mode=mode,
        gains=initial_gains,
    )
    p_time = np.asarray(p_injection, dtype=float)
    q_base = np.asarray(q_base, dtype=float).reshape(-1)
    n_time, n_controlled = p_time.shape

    voltage = np.empty((n_time, context.model.n_bus), dtype=float)
    control = np.zeros((n_time, n_controlled), dtype=float)
    next_control = np.zeros_like(control)
    q_total = np.empty((n_time, n_controlled), dtype=float)
    gains = np.empty((n_time + 1, n_controlled), dtype=float)
    gradient = np.full((n_time, n_controlled), np.nan, dtype=float)
    did_gain_update = np.zeros(n_time, dtype=bool)
    control_action_mask = np.zeros(n_time, dtype=bool)
    max_abs_voltage_error = np.empty(n_time, dtype=float)
    voltage_error_l2 = np.empty(n_time, dtype=float)
    control_l2 = np.empty(n_time, dtype=float)
    gains[0] = controller.gains

    error_buffer: list[np.ndarray] = []
    gain_windows: list[tuple[int, int]] = []

    for minute in range(n_time):
        applied_action = controller.u_ctrl.copy()
        if mode == "nocontrol":
            q_at_minute = q_base.copy()
        else:
            q_at_minute = controller.construct_q_total(
                q_base.reshape(-1, 1), applied_action
            ).reshape(-1)
        solution = context.model.runpf(
            p_time[minute].reshape(-1, 1),
            q_at_minute.reshape(-1, 1),
        )
        current_voltage = np.asarray(solution["v"], dtype=float).reshape(-1)
        if not np.all(np.isfinite(current_voltage)):
            raise RuntimeError(f"{mode} produced non-finite voltage at minute {minute}.")
        voltage_error = controller.measure_voltage_error(current_voltage)

        voltage[minute] = current_voltage
        control[minute] = applied_action
        q_total[minute] = q_at_minute
        max_abs_voltage_error[minute] = float(
            np.max(np.abs(current_voltage[1:] - 1.0))
        )
        voltage_error_l2[minute] = float(np.linalg.norm(voltage_error))
        control_l2[minute] = float(np.linalg.norm(applied_action))

        if mode == "block":
            error_buffer.append(voltage_error.copy())
            if len(error_buffer) == 2 * T:
                gradient[minute] = controller.update_gains(
                    np.asarray(error_buffer),
                    eta=eta,
                    lam=gain_regularization,
                    T=T,
                    k_min=GAIN_MIN,
                    k_max=GAIN_MAX,
                )
                did_gain_update[minute] = True
                gain_windows.append((minute - 2 * T + 1, minute))
                error_buffer.clear()

        if mode != "nocontrol":
            controller.step_voltage_control(current_voltage)
            control_action_mask[minute] = True
        next_control[minute] = controller.u_ctrl
        gains[minute + 1] = controller.gains

    return {
        "mode": mode,
        "voltage": voltage,
        "control": control,
        "next_control": next_control,
        "q_total": q_total,
        "gains": gains,
        "gradient": gradient,
        "did_gain_update": did_gain_update,
        "control_action_mask": control_action_mask,
        "max_abs_voltage_error": max_abs_voltage_error,
        "voltage_error_l2": voltage_error_l2,
        "control_l2": control_l2,
        "gain_windows": gain_windows,
    }


def run_time_varying_controller_comparison(
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    output_dir: str | Path | None = None,
    date: str | None = None,
    gain_seed: int = DEFAULT_GAIN_SEED,
    gain_low: float = DEFAULT_GAIN_LOW,
    gain_high: float = DEFAULT_GAIN_HIGH,
    profile_gain: float = PROFILE_GAIN,
    ramp_amplitude: float = RAMP_AMPLITUDE,
    shape_clip: float = SHAPE_CLIP,
    smoothing_window: int = SMOOTHING_WINDOW,
    warmup_minutes: int = WARMUP_MINUTES,
    T: int = DEFAULT_T,
    eta: float = DEFAULT_ETA,
    gain_regularization: float = DEFAULT_LAMBDA,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Run all controller modes and save their results, metrics, and optional plots."""

    if T < 1:
        raise ValueError("T must be at least 1.")
    if eta < 0.0 or gain_regularization < 0.0:
        raise ValueError("eta and gain_regularization must be nonnegative.")
    output_path = Path(
        output_dir or PROJECT_ROOT / "outputs" / "results"
    ).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    context = build_context()
    profile = load_profile_data(data_root=data_root, date=date)
    mapped = map_profiles_to_ieee33(
        context.p_ieee_injection,
        profile["centered_p_shape"],
        profile_gain=profile_gain,
        ramp_amplitude=ramp_amplitude,
        shape_clip=shape_clip,
        smoothing_window=smoothing_window,
        warmup_minutes=warmup_minutes,
    )
    p_time = np.asarray(mapped["p_injection"], dtype=float).T
    q_base = np.asarray(context.q_ieee_injection, dtype=float)
    initial_gains = generate_diagonal_gains(
        32,
        seed=gain_seed,
        low=gain_low,
        high=gain_high,
    )
    mode_results = [
        _run_mode(
            mode,
            p_injection=p_time,
            q_base=q_base,
            initial_gains=initial_gains,
            T=T,
            eta=eta,
            gain_regularization=gain_regularization,
        )
        for mode in MODE_NAMES
    ]
    n_time = p_time.shape[0]
    block_windows = mode_results[MODE_NAMES.index("block")]["gain_windows"]
    voltage_stack = np.stack([result["voltage"] for result in mode_results])
    plot_buses = sample_plot_buses()
    arrays: dict[str, np.ndarray] = {
        "mode_names": np.asarray(MODE_NAMES),
        "timestamps": np.asarray(profile["timestamps"]),
        "minute_index": np.arange(n_time, dtype=int),
        "raw_p_profiles": np.asarray(profile["raw_p_profiles"]),
        "normalized_p_profiles": np.asarray(profile["normalized_p_profiles"]),
        "profile_mix_weights": np.asarray(profile["profile_mix_weights"]),
        "profile_ids": np.asarray(profile["profile_ids"]),
        "aggregated_p_shape": np.asarray(profile["aggregated_p_shape"]),
        "centered_p_shape": np.asarray(profile["centered_p_shape"]),
        "temporally_smoothed_p_shape": np.asarray(
            mapped["temporally_smoothed_p_shape"]
        ),
        "bounded_p_shape": np.asarray(mapped["bounded_p_shape"]),
        "stress_amplitude_envelope": np.asarray(
            mapped["stress_amplitude_envelope"]
        ),
        "effective_p_shape": np.asarray(mapped["effective_p_shape"]),
        "smoothstep_envelope": np.asarray(mapped["smoothstep_envelope"]),
        "p_multiplier": np.asarray(mapped["p_multiplier"]),
        "p_load": np.asarray(mapped["p_load"]),
        "p_injection": np.asarray(mapped["p_injection"]),
        "p_ieee_load": np.asarray(mapped["p_ieee_load"]),
        "q_base": q_base,
        "voltage": voltage_stack,
        "control": np.stack([result["control"] for result in mode_results]),
        "next_control": np.stack(
            [result["next_control"] for result in mode_results]
        ),
        "q_total": np.stack([result["q_total"] for result in mode_results]),
        "gains": np.stack([result["gains"] for result in mode_results]),
        "gradient": np.stack([result["gradient"] for result in mode_results]),
        "did_gain_update": np.stack(
            [result["did_gain_update"] for result in mode_results]
        ),
        "control_action_mask": np.stack(
            [result["control_action_mask"] for result in mode_results]
        ),
        "max_abs_voltage_error": np.stack(
            [result["max_abs_voltage_error"] for result in mode_results]
        ),
        "voltage_error_l2": np.stack(
            [result["voltage_error_l2"] for result in mode_results]
        ),
        "control_l2": np.stack([result["control_l2"] for result in mode_results]),
        "initial_gain_vector": initial_gains,
        "initial_gain_matrix": np.diag(initial_gains),
        "plot_ieee_buses": plot_buses,
        "plot_node_seed": np.asarray(PLOT_NODE_SEED),
        "profile_gain": np.asarray(float(profile_gain)),
        "ramp_amplitude": np.asarray(float(ramp_amplitude)),
        "shape_clip": np.asarray(float(shape_clip)),
        "smoothing_window": np.asarray(int(smoothing_window)),
        "warmup_minutes": np.asarray(int(warmup_minutes)),
        "T": np.asarray(T),
        "eta": np.asarray(float(eta)),
        "gain_regularization": np.asarray(float(gain_regularization)),
        "gain_seed": np.asarray(int(gain_seed)),
        "gain_low": np.asarray(float(gain_low)),
        "gain_high": np.asarray(float(gain_high)),
        "block_window_start": np.asarray(
            [window[0] for window in block_windows], dtype=int
        ),
        "block_window_end": np.asarray(
            [window[1] for window in block_windows], dtype=int
        ),
    }
    npz_path = output_path / "time_varying_results.npz"
    np.savez_compressed(npz_path, **arrays)

    modes = {}
    for result in mode_results:
        metrics = voltage_action_metrics(result["voltage"], result["control"])
        metrics.update(
            {
                "num_gain_updates": int(np.count_nonzero(result["did_gain_update"])),
                "final_gain_min": float(np.min(result["gains"][-1])),
                "final_gain_max": float(np.max(result["gains"][-1])),
            }
        )
        modes[str(result["mode"])] = metrics

    summary: dict[str, Any] = {
        "experiment": "time_varying_controller_comparison",
        "date": profile["date"],
        "samples": int(n_time),
        "source": {
            "npz": project_relative_path(profile["source_npz"]),
            "manifest": project_relative_path(profile["source_manifest"]),
            "arrays_used": ["raw_p_profiles", "profile_mix_weights"],
        },
        "p_construction": {
            "method": "fixed_daily_mapping",
            "profile_gain": float(profile_gain),
            "ramp_amplitude": float(ramp_amplitude),
            "shape_clip": float(shape_clip),
            "smoothing_window": int(smoothing_window),
            "warmup_minutes": int(warmup_minutes),
            "timezone": LOCAL_TIMEZONE,
            "local_window": "09:00-21:00",
        },
        "controller": {
            "control_action_period_minutes": 1,
            "gain_sample_period_minutes": 1,
            "T": int(T),
            "block_window_samples": int(2 * T),
            "eta": float(eta),
            "gain_regularization": float(gain_regularization),
            "gain_seed": int(gain_seed),
            "gain_low": float(gain_low),
            "gain_high": float(gain_high),
            "reactive_action_limit_pu": None,
        },
        "plot_nodes": {
            "method": "uniform_random_without_replacement",
            "seed": PLOT_NODE_SEED,
            "population_ieee_buses": "2-33",
            "node_count": PLOT_NODE_COUNT,
            "selected_ieee_buses": plot_buses.tolist(),
        },
        "modes": modes,
        "outputs": {"npz": project_relative_path(npz_path)},
    }
    if make_plots:
        from experiments.plot_time_varying_dql_figs import (
            plot_time_varying_dql_figs,
        )
        from experiments.plot_time_varying_gain_dql_figs import (
            plot_time_varying_gain_dql_fig,
        )

        figures_path = output_path / "figures"
        summary["outputs"]["figures"] = plot_time_varying_dql_figs(
            npz_path, output_dir=figures_path
        )
        summary["outputs"]["gain_figure"] = plot_time_varying_gain_dql_fig(
            npz_path, output_dir=figures_path
        )
    summary_path = output_path / "summary.json"
    summary["outputs"]["summary_json"] = project_relative_path(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"summary": summary, "arrays": arrays, "mode_results": mode_results}


def _build_parser() -> argparse.ArgumentParser:
    """Define the experiment's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--date",
        help="Local date (YYYY-MM-DD); inferred from input timestamps when omitted.",
    )
    parser.add_argument("--gain-seed", type=int, default=DEFAULT_GAIN_SEED)
    parser.add_argument("--gain-low", type=float, default=DEFAULT_GAIN_LOW)
    parser.add_argument("--gain-high", type=float, default=DEFAULT_GAIN_HIGH)
    parser.add_argument("--profile-gain", type=float, default=PROFILE_GAIN)
    parser.add_argument("--ramp-amplitude", type=float, default=RAMP_AMPLITUDE)
    parser.add_argument("--shape-clip", type=float, default=SHAPE_CLIP)
    parser.add_argument(
        "--smoothing-window", type=int, default=SMOOTHING_WINDOW
    )
    parser.add_argument("--warmup-minutes", type=int, default=WARMUP_MINUTES)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--gain-regularization", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--no-plots", action="store_true")
    return parser


def main() -> None:
    """Run the experiment with command-line options."""
    args = _build_parser().parse_args()
    result = run_time_varying_controller_comparison(
        data_root=args.data_root,
        output_dir=args.output_dir,
        date=args.date,
        gain_seed=args.gain_seed,
        gain_low=args.gain_low,
        gain_high=args.gain_high,
        profile_gain=args.profile_gain,
        ramp_amplitude=args.ramp_amplitude,
        shape_clip=args.shape_clip,
        smoothing_window=args.smoothing_window,
        warmup_minutes=args.warmup_minutes,
        T=args.T,
        eta=args.eta,
        gain_regularization=args.gain_regularization,
        make_plots=not args.no_plots,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
