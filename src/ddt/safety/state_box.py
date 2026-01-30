from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..interfaces import Estimate, SafetyFilter


@dataclass
class StateBoxSafetyFilter(SafetyFilter):
    """Discrete-time 'CBF-like' safety filter for a scalar state x with box constraints.

    This is a *minimal* placeholder. For real systems implement a proper CBF-QP.

    We enforce:
        x_{k+1} in [x_min, x_max]
    using the estimated linear dynamics from theta_hat = [a, b]:
        x_{k+1} = a x_k + b u_k

    The filter minimally modifies u_nom by clamping it to satisfy the bounds.
    """

    x_min: float
    x_max: float
    alpha: float = 0.05  # unused in this simple clamp
    slack_weight: float = 1000.0  # unused

    def reset(self) -> None:
        return

    def filter(self, u_nom: np.ndarray, est: Estimate) -> np.ndarray:
        u_nom_val = float(np.asarray(u_nom).reshape(()))
        x = float(est.x_hat.reshape(()))
        a, b = [float(v) for v in est.theta_hat.reshape(-1)[:2]]

        # If b is too small, we can't correct via u. Just return u_nom.
        if abs(b) < 1e-8:
            return np.array([u_nom_val], dtype=float)

        # Compute u interval that keeps x_next within bounds:
        # x_next = a*x + b*u
        u_low = (self.x_min - a * x) / b
        u_high = (self.x_max - a * x) / b
        if u_low > u_high:
            u_low, u_high = u_high, u_low

        u_safe = float(np.clip(u_nom_val, u_low, u_high))
        return np.array([u_safe], dtype=float)
