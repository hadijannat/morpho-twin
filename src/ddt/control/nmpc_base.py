"""Abstract base class for NMPC implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..interfaces import Controller, Estimate

if TYPE_CHECKING:
    pass


def freeze_theta(theta: np.ndarray) -> np.ndarray:
    """Stop-gradient boundary for parameter estimates.

    CRITICAL: This prevents control loss from flowing back to parameter
    estimates during gradient-based optimization. This ensures causal
    correctness - control actions should not retroactively influence
    the parameter estimates used to compute them.

    Args:
        theta: Parameter estimate array

    Returns:
        Detached parameter array (no gradient flow)
    """
    try:
        import jax

        if isinstance(theta, jax.Array):
            theta = jax.lax.stop_gradient(theta)
    except ImportError:
        pass

    return np.asarray(theta, dtype=np.float64)


@dataclass
class NMPCBase(ABC, Controller):
    """Abstract base class for Nonlinear Model Predictive Control.

    Solves at each step:
        min  Σ ||x_i - x_ref||²_Q + ||u_i||²_R + λ_info * J_info

        s.t. x_{i+1} = f(x_i, u_i, θ_frozen)
             x_min ≤ x_i ≤ x_max
             u_min ≤ u_i ≤ u_max

    where θ_frozen = freeze_theta(θ_hat) prevents gradient flow.
    """

    dt: float
    horizon: int
    nx: int = 1
    nu: int = 1
    ntheta: int = 2

    # Cost weights
    Q: np.ndarray = field(default_factory=lambda: np.array([[10.0]]))
    R_u: np.ndarray = field(default_factory=lambda: np.array([[0.1]]))
    lambda_info: float = 0.01  # FIM probing weight
    fim_criterion: str = "d_optimal"  # FIM criterion for dual-control

    # Constraints
    x_min: np.ndarray = field(default_factory=lambda: np.array([-np.inf]))
    x_max: np.ndarray = field(default_factory=lambda: np.array([np.inf]))
    u_min: np.ndarray = field(default_factory=lambda: np.array([-1.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    # Internal state
    _u_traj: np.ndarray = field(init=False)
    _x_traj: np.ndarray = field(init=False)
    _ref: np.ndarray = field(init=False)
    _theta_frozen: np.ndarray = field(init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._u_traj = np.zeros((self.horizon, self.nu))
        self._x_traj = np.zeros((self.horizon + 1, self.nx))
        self._ref = np.zeros(self.nx)
        self._theta_frozen = np.ones(self.ntheta)
        self._initialized = False

    def reset(self) -> None:
        """Reset controller state."""
        self._u_traj = np.zeros((self.horizon, self.nu))
        self._x_traj = np.zeros((self.horizon + 1, self.nx))
        self._ref = np.zeros(self.nx)
        self._theta_frozen = np.ones(self.ntheta)
        self._initialized = False

    def compute_u(self, ref: np.ndarray, est: Estimate) -> np.ndarray:
        """Compute control input using NMPC.

        Args:
            ref: Reference/setpoint (nx,)
            est: Current state and parameter estimate

        Returns:
            Control input u (nu,)
        """
        ref = np.atleast_1d(np.asarray(ref, dtype=np.float64))
        x_hat = np.atleast_1d(np.asarray(est.x_hat, dtype=np.float64))

        # CRITICAL: Freeze theta to prevent gradient flow
        theta_frozen = freeze_theta(est.theta_hat)
        theta_cov = est.theta_cov.copy()

        self._ref = ref
        self._theta_frozen = theta_frozen

        # Solve NMPC
        u_opt, x_opt, fim = self._solve_nmpc(x_hat, ref, theta_frozen, theta_cov)

        # Store for warm-start
        self._u_traj = u_opt
        self._x_traj = x_opt
        self._initialized = True

        # Return first control action
        result: np.ndarray = u_opt[0].copy()
        return result

    @abstractmethod
    def _solve_nmpc(
        self,
        x0: np.ndarray,
        ref: np.ndarray,
        theta: np.ndarray,
        theta_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the NMPC optimization problem.

        Args:
            x0: Current state (nx,)
            ref: Reference trajectory or setpoint (nx,)
            theta: Frozen parameter estimate (ntheta,)
            theta_cov: Parameter covariance for FIM weighting

        Returns:
            u_opt: Optimal input trajectory (horizon, nu)
            x_opt: Optimal state trajectory (horizon+1, nx)
            fim: Fisher Information Matrix for dual-control
        """
        ...

    def get_predicted_trajectory(self) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted state and input trajectories."""
        return self._x_traj.copy(), self._u_traj.copy()
