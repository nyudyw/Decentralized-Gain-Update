"""Incremental voltage control and local gain adaptation."""

import numpy as np


def compute_local_gain_gradient(
    voltage_error_history: np.ndarray,
    gains: np.ndarray,
    T: int,
    lam: float,
) -> np.ndarray:
    """Compute -2 sum_t sum_s e_s e_(2t-1-s) + lam * gains locally."""
    window = voltage_error_history[-2 * T :]
    correlation = np.zeros_like(gains)
    for t in range(1, T + 1):
        for s in range(t):
            correlation += window[s] * window[2 * t - 1 - s]
    return -2.0 * correlation + lam * gains


class DecentralizedLinearController:
    """Apply diagonal incremental control at the selected non-slack buses."""

    def __init__(
        self,
        *,
        controlled_buses: np.ndarray,
        gains: np.ndarray,
        v_ref: float = 1.0,
    ):
        # Bus labels start at 1; bus 1 is the slack bus.
        self.controlled_indices = controlled_buses - 2
        self.full_voltage_indices = controlled_buses - 1
        self.gains = gains.copy()
        self.v_ref = v_ref
        self.u_ctrl = np.zeros_like(self.gains)

    def construct_q_total(
        self,
        q_base: np.ndarray,
        u_ctrl: np.ndarray,
    ) -> np.ndarray:
        """Add control injections to the non-slack reactive-power vector."""
        q_total = q_base.copy()
        q_total[self.controlled_indices, 0] += u_ctrl
        return q_total

    def measure_voltage_error(self, v_full: np.ndarray) -> np.ndarray:
        """Return controlled-bus voltage errors in per unit."""
        return v_full[self.full_voltage_indices] - self.v_ref

    def step_voltage_control(self, v_full: np.ndarray) -> None:
        """Apply u_next = u - gains * (v - v_ref)."""
        error = self.measure_voltage_error(v_full)
        self.u_ctrl = self.u_ctrl - self.gains * error


class DecentralizedGainAdaptationLinearController(DecentralizedLinearController):
    """Decentralized linear controller with projected gain adaptation."""

    def update_gains(
        self,
        voltage_error_history: np.ndarray,
        *,
        eta: float,
        lam: float,
        T: int,
        k_min: float,
        k_max: float,
    ) -> np.ndarray:
        """Take a local gradient step and project gains onto [k_min, k_max]."""
        grad = compute_local_gain_gradient(
            voltage_error_history, self.gains, T, lam
        )
        self.gains = np.clip(self.gains - eta * grad, k_min, k_max)
        return grad
