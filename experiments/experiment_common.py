"""Share IEEE-33 setup, controller defaults, and experiment metrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from controllers.decentralized_linear import (
    DecentralizedGainAdaptationLinearController,
    DecentralizedLinearController,
)
from models.PF_models import DistFlow
from utils.case_loader import load_case_from_files


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = PROJECT_ROOT / "cases" / "pypower"
CASE_NAME = "case33"

MODE_NAMES = ("nocontrol", "nolearning", "block")
CONTROLLED_BUSES = np.arange(2, 34, dtype=int)
VOLTAGE_REFERENCE = 1.0
GAIN_MIN = 0.001
GAIN_MAX = 0.06


@dataclass(frozen=True)
class ExperimentContext:
    """Store IEEE-33 data, per-unit injections, and the DistFlow simulator."""

    model: DistFlow
    p_ieee_injection: np.ndarray
    q_ieee_injection: np.ndarray


def build_context() -> ExperimentContext:
    """Load IEEE-33 data, convert injections to per unit, and create a simulator."""

    case = load_case_from_files(CASE_ROOT, CASE_NAME)
    model = DistFlow(case)
    base_mva = float(case["baseMVA"])
    return ExperimentContext(
        model=model,
        p_ieee_injection=-np.asarray(case["bus"][1:, 2], dtype=float) / base_mva,
        q_ieee_injection=-np.asarray(case["bus"][1:, 3], dtype=float) / base_mva,
    )


def make_controller(
    *,
    mode: str,
    gains: np.ndarray,
) -> DecentralizedLinearController:
    """Create the controller for the selected mode using the supplied gains."""

    controller_type = (
        DecentralizedGainAdaptationLinearController
        if mode == "block"
        else DecentralizedLinearController
    )
    return controller_type(
        controlled_buses=CONTROLLED_BUSES,
        gains=np.asarray(gains, dtype=float).reshape(-1),
        v_ref=VOLTAGE_REFERENCE,
    )


def generate_diagonal_gains(
    n_controlled: int,
    *,
    seed: int,
    low: float,
    high: float,
) -> np.ndarray:
    """Sample diagonal gains uniformly within the bounds using a fixed seed."""

    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=n_controlled)


def voltage_action_metrics(
    voltage: np.ndarray,
    control: np.ndarray,
) -> dict[str, float | int | bool]:
    """Summarize voltage, 0.95-1.05 per-unit limit violations, and control magnitude."""

    voltage = np.asarray(voltage, dtype=float)[..., 1:]
    control = np.asarray(control, dtype=float)
    deviation = np.abs(voltage - VOLTAGE_REFERENCE)
    violation = deviation > 0.05
    control_l2 = np.linalg.norm(control, axis=-1)
    return {
        "voltage_minimum": float(np.min(voltage)),
        "voltage_maximum": float(np.max(voltage)),
        "voltage_deviation_maximum": float(np.max(deviation)),
        "voltage_deviation_mean": float(np.mean(deviation)),
        "voltage_deviation_rmse": float(np.sqrt(np.mean(deviation**2))),
        "voltage_deviation_p95": float(np.quantile(deviation, 0.95)),
        "voltage_violation_minutes": int(
            np.count_nonzero(np.any(violation, axis=-1))
        ),
        "voltage_violation_node_minutes": int(np.count_nonzero(violation)),
        "all_voltages_within_0p95_1p05": bool(not np.any(violation)),
        "mean_absolute_action": float(np.mean(np.abs(control))),
        "mean_action_l2": float(np.mean(control_l2)),
        "maximum_absolute_action": float(np.max(np.abs(control))),
    }
