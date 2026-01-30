"""EKF-based arrival cost updater for MHE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np


@dataclass
class EKFArrivalCostUpdater:
    """Updates MHE arrival cost prior between solves.

    Runs a lightweight EKF to propagate (x_prior, P0) forward between
    MHE solve times, incorporating measurements that arrive before
    the next MHE optimization.

    This prevents "amnesia" about data before the MHE window boundary.

    The EKF uses Joseph form for numerical stability:
        P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    """

    nx: int
    ntheta: int
    Q: np.ndarray  # Process noise covariance (nx, nx)
    R: np.ndarray  # Measurement noise covariance (ny, ny)

    # Internal state
    _x_prior: np.ndarray = field(init=False)
    _P: np.ndarray = field(init=False)
    _theta: np.ndarray = field(init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Initialize EKF state."""
        self._x_prior = np.zeros(self.nx)
        self._P = np.eye(self.nx) * 10.0  # Start with loose prior
        self._theta = np.ones(self.ntheta)
        self._initialized = False

    def initialize(
        self,
        x0: np.ndarray,
        P0: np.ndarray,
        theta: np.ndarray,
    ) -> None:
        """Initialize EKF with starting state.

        Args:
            x0: Initial state estimate (nx,)
            P0: Initial state covariance (nx, nx)
            theta: Parameter estimate for dynamics (ntheta,)
        """
        self._x_prior = np.atleast_1d(np.asarray(x0, dtype=np.float64)).copy()
        self._P = np.atleast_2d(np.asarray(P0, dtype=np.float64)).copy()
        self._theta = np.atleast_1d(np.asarray(theta, dtype=np.float64)).copy()
        self._initialized = True

    def predict(
        self,
        u_applied: np.ndarray,
        f_discrete: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        A_func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    ) -> None:
        """EKF prediction step: propagate state and covariance forward.

        x_pred = f(x, u, theta)
        P_pred = A @ P @ A.T + Q

        Args:
            u_applied: Applied (safe) control input (nu,)
            f_discrete: Discrete dynamics function f(x, u, theta) -> x_next
            A_func: Jacobian of dynamics w.r.t. state: df/dx(x, u, theta) -> (nx, nx)
        """
        if not self._initialized:
            return

        u_applied = np.atleast_1d(np.asarray(u_applied, dtype=np.float64))

        # Predict state
        x_pred = f_discrete(self._x_prior, u_applied, self._theta)
        self._x_prior = np.atleast_1d(x_pred)

        # Linearize dynamics
        A = A_func(self._x_prior, u_applied, self._theta)
        A = np.atleast_2d(A)

        # Predict covariance
        self._P = A @ self._P @ A.T + self.Q

    def update(
        self,
        y: np.ndarray,
        h_func: Callable[[np.ndarray], np.ndarray],
        H_func: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        """EKF measurement update with Joseph form for numerical stability.

        K = P @ H.T @ (H @ P @ H.T + R)^{-1}
        x = x + K @ (y - h(x))
        P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T  [Joseph form]

        Args:
            y: Measurement (ny,)
            h_func: Measurement function h(x) -> y_pred
            H_func: Measurement Jacobian dh/dx(x) -> (ny, nx)
        """
        if not self._initialized:
            return

        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        # Predicted measurement
        y_pred = h_func(self._x_prior)
        y_pred = np.atleast_1d(y_pred)

        # Measurement Jacobian
        H = H_func(self._x_prior)
        H = np.atleast_2d(H)

        # Innovation covariance
        S = H @ self._P @ H.T + self.R
        S = np.atleast_2d(S)

        # Kalman gain
        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback: skip update if singular
            return

        # State update
        innovation = y - y_pred
        self._x_prior = self._x_prior + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.nx) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ self.R @ K.T

        # Ensure symmetry
        self._P = 0.5 * (self._P + self._P.T)

    def update_from_mhe_solution(
        self,
        x_mhe_start: np.ndarray,
        P_mhe: np.ndarray,
        theta: np.ndarray,
    ) -> None:
        """Sync EKF with MHE solution at window boundary.

        After MHE solves, the beginning of the window provides a refined
        estimate. Use this to reset the EKF prior for the next interval.

        Args:
            x_mhe_start: MHE estimate at window start (nx,)
            P_mhe: Covariance at window start (nx, nx) or (nx,) diagonal
            theta: Updated parameter estimate (ntheta,)
        """
        self._x_prior = np.atleast_1d(np.asarray(x_mhe_start, dtype=np.float64)).copy()

        P_mhe = np.asarray(P_mhe, dtype=np.float64)
        if P_mhe.ndim == 1:
            self._P = np.diag(P_mhe)
        else:
            self._P = P_mhe.copy()

        self._theta = np.atleast_1d(np.asarray(theta, dtype=np.float64)).copy()
        self._initialized = True

    def get_arrival_cost_prior(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_prior, P0_diag) for MHE arrival cost.

        Returns:
            x_prior: Prior state estimate (nx,)
            P0_diag: Diagonal of prior covariance (nx,)
        """
        if not self._initialized:
            # Return loose prior if not initialized
            return np.zeros(self.nx), np.ones(self.nx) * 10.0

        return self._x_prior.copy(), np.diag(self._P).copy()

    def reset(self) -> None:
        """Reset EKF state."""
        self._x_prior = np.zeros(self.nx)
        self._P = np.eye(self.nx) * 10.0
        self._theta = np.ones(self.ntheta)
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if EKF has been initialized."""
        return self._initialized
