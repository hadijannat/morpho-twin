"""Persistence of Excitation (PE) monitoring."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PEStatus:
    """Persistence of Excitation status."""

    lambda_min: float  # Minimum eigenvalue of rolling FIM
    condition_number: float  # Condition number of FIM
    is_pe_satisfied: bool  # Whether PE condition is met
    recommended_probe_weight: float  # Suggested lambda_info for dual-control


@dataclass
class PEMonitor:
    """Monitor for Persistence of Excitation.

    Tracks a rolling Fisher Information Matrix to detect
    when the system is not being sufficiently excited for
    parameter identifiability.

    The PE condition requires:
        λ_min(F̃) ≥ λ_threshold

    where F̃ = Σ φ(k) @ φ(k)^T is the rolling FIM.
    """

    window: int = 100  # Rolling window size
    lambda_threshold: float = 0.1  # PE threshold
    condition_threshold: float = 100.0  # Maximum condition number
    ntheta: int = 2  # Number of parameters

    # Internal state
    _regressor_buffer: deque = field(default_factory=deque, init=False)
    _rolling_fim: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._regressor_buffer = deque(maxlen=self.window)
        self._rolling_fim = np.eye(self.ntheta) * 1e-6  # Small initial FIM

    def reset(self) -> None:
        """Reset the PE monitor."""
        self._regressor_buffer.clear()
        self._rolling_fim = np.eye(self.ntheta) * 1e-6

    def update(
        self,
        regressor: np.ndarray,
    ) -> PEStatus:
        """Update PE monitor with new regressor.

        For linear regression y = φ^T θ, the regressor φ should be
        provided at each time step. For MHE/NMPC, this is typically
        the sensitivity of the output to parameters.

        Args:
            regressor: Regressor vector φ (ntheta,) or matrix (n, ntheta)

        Returns:
            Current PE status
        """
        regressor = np.atleast_2d(np.asarray(regressor, dtype=np.float64))

        # Add to buffer
        for row in regressor:
            self._regressor_buffer.append(row.copy())

        # Compute rolling FIM
        self._rolling_fim = self._compute_rolling_fim()

        # Compute PE metrics
        return self._compute_pe_status()

    def _compute_rolling_fim(self) -> np.ndarray:
        """Compute rolling Fisher Information Matrix."""
        if len(self._regressor_buffer) == 0:
            return np.eye(self.ntheta) * 1e-6

        # F̃ = Σ φ(k) @ φ(k)^T
        fim = np.zeros((self.ntheta, self.ntheta))

        for phi in self._regressor_buffer:
            phi = phi.reshape(-1, 1)
            fim += phi @ phi.T

        # Add small regularization
        fim += 1e-8 * np.eye(self.ntheta)

        return fim

    def _compute_pe_status(self) -> PEStatus:
        """Compute current PE status from rolling FIM."""
        fim = self._rolling_fim

        # Eigenvalue analysis
        eigvals = np.linalg.eigvalsh(fim)
        lambda_min = float(np.min(eigvals))
        lambda_max = float(np.max(eigvals))

        # Condition number
        if lambda_min > 1e-10:
            cond = lambda_max / lambda_min
        else:
            cond = np.inf

        # PE satisfied?
        is_pe = lambda_min >= self.lambda_threshold and cond <= self.condition_threshold

        # Recommended probing weight
        if lambda_min < self.lambda_threshold:
            # Increase probing proportionally to deficiency
            deficiency = self.lambda_threshold / max(lambda_min, 1e-10)
            probe_weight = 0.01 * min(deficiency, 10.0)  # Cap at 0.1
        else:
            probe_weight = 0.0  # No extra probing needed

        return PEStatus(
            lambda_min=lambda_min,
            condition_number=float(cond),
            is_pe_satisfied=is_pe,
            recommended_probe_weight=probe_weight,
        )

    def get_fim(self) -> np.ndarray:
        """Get current rolling FIM."""
        result: np.ndarray = self._rolling_fim.copy()
        return result

    def get_status(self) -> PEStatus:
        """Get current PE status without update."""
        return self._compute_pe_status()


def compute_regressor_from_estimate(
    x: np.ndarray,
    u: np.ndarray,
    theta: np.ndarray,
    sensitivity_func: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Compute regressor for PE monitoring.

    For linear systems x+ = a*x + b*u, the regressor is φ = [x, u].
    For nonlinear systems, use the sensitivity function.

    Args:
        x: Current state
        u: Current input
        theta: Parameter estimate (unused for linear)
        sensitivity_func: Optional function computing dh/dθ

    Returns:
        Regressor vector (ntheta,)
    """
    x = np.atleast_1d(x)
    u = np.atleast_1d(u)

    if sensitivity_func is not None:
        # Use sensitivity for nonlinear systems
        return np.array(sensitivity_func(x, u, theta)).flatten()
    else:
        # Linear regressor: [x, u]
        return np.concatenate([x, u])


def detect_pe_violation_trend(
    pe_history: list[PEStatus],
    lookback: int = 10,
) -> bool:
    """Detect if PE is trending worse.

    Args:
        pe_history: List of recent PE statuses
        lookback: Number of steps to analyze

    Returns:
        True if PE is getting worse
    """
    if len(pe_history) < lookback:
        return False

    recent = pe_history[-lookback:]
    lambda_mins = [s.lambda_min for s in recent]

    # Check if λ_min is consistently decreasing
    decreasing_count = sum(
        1 for i in range(1, len(lambda_mins)) if lambda_mins[i] < lambda_mins[i - 1]
    )

    return decreasing_count > lookback * 0.7  # >70% decreasing
