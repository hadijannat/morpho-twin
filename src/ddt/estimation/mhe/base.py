"""Abstract base class for MHE implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...interfaces import Estimate, Estimator

if TYPE_CHECKING:
    from .config import MHEConfig


@dataclass
class MHEBase(ABC, Estimator):
    """Abstract base class for Moving Horizon Estimation.

    Solves:
        min  ||x_{k-N} - x_prior||²_{P₀⁻¹}
           + Σ||w_i||²_{Q⁻¹}
           + Σ||v_i||²_{R⁻¹}
           + η||θ - θ_prior||²

        s.t. x_{i+1} = f(x_i, u_applied_i, θ) + w_i
             y_i = h(x_i) + v_i

    CRITICAL: Uses u_applied (safe input), NOT u_nom.
    """

    cfg: MHEConfig
    dt: float
    nx: int = 1
    nu: int = 1
    ny: int = 1
    ntheta: int = 2

    # Internal buffers (initialized in __post_init__)
    _y_buffer: list[np.ndarray] = field(default_factory=list, init=False)
    _u_buffer: list[np.ndarray] = field(default_factory=list, init=False)
    _x_traj: np.ndarray = field(init=False)
    _theta_hat: np.ndarray = field(init=False)
    _theta_cov: np.ndarray = field(init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._y_buffer = []
        self._u_buffer = []
        self._x_traj = np.zeros((self.cfg.horizon + 1, self.nx))
        self._theta_hat = np.array(self.cfg.parameters.theta_init, dtype=np.float64)
        self._theta_cov = np.diag(self.cfg.parameters.P_theta_diag).astype(np.float64)
        self._initialized = False

    def reset(self) -> None:
        """Reset estimator state."""
        self._y_buffer.clear()
        self._u_buffer.clear()
        self._x_traj = np.zeros((self.cfg.horizon + 1, self.nx))
        self._theta_hat = np.array(self.cfg.parameters.theta_init, dtype=np.float64)
        self._theta_cov = np.diag(self.cfg.parameters.P_theta_diag).astype(np.float64)
        self._initialized = False

    def update(self, y: np.ndarray, u_applied: np.ndarray) -> Estimate:
        """Update estimate with new measurement and applied input.

        Args:
            y: Current measurement (ny,)
            u_applied: Applied (safe) input from previous step (nu,)

        Returns:
            Estimate with x_hat, theta_hat, theta_cov
        """
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        u_applied = np.atleast_1d(np.asarray(u_applied, dtype=np.float64))

        # Store in buffers
        self._y_buffer.append(y.copy())
        self._u_buffer.append(u_applied.copy())

        # Trim to horizon
        N = self.cfg.horizon
        if len(self._y_buffer) > N + 1:
            self._y_buffer.pop(0)
        if len(self._u_buffer) > N + 1:
            self._u_buffer.pop(0)

        # Need at least 2 measurements to estimate
        if len(self._y_buffer) < 2:
            x_hat = y.copy()
            return Estimate(
                x_hat=x_hat,
                theta_hat=self._theta_hat.copy(),
                theta_cov=self._theta_cov.copy(),
            )

        # Solve MHE problem
        x_opt, theta_opt, cov = self._solve_mhe()

        self._x_traj = x_opt
        self._theta_hat = theta_opt
        self._theta_cov = cov
        self._initialized = True

        # Current state estimate is last element
        x_hat = x_opt[-1].copy()

        return Estimate(
            x_hat=x_hat,
            theta_hat=self._theta_hat.copy(),
            theta_cov=self._theta_cov.copy(),
        )

    @abstractmethod
    def _solve_mhe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the MHE optimization problem.

        Returns:
            x_opt: Optimal state trajectory (N+1, nx)
            theta_opt: Optimal parameter estimate (ntheta,)
            cov: Parameter covariance estimate (ntheta, ntheta)
        """
        ...

    def get_trajectory(self) -> np.ndarray:
        """Return the current state trajectory estimate."""
        result: np.ndarray = self._x_traj.copy()
        return result

    @property
    def horizon_filled(self) -> bool:
        """Check if buffer has reached full horizon."""
        return len(self._y_buffer) >= self.cfg.horizon + 1
