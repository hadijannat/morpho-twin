"""Warm-start utilities for MHE trajectory shifting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def shift_trajectory(
    x_traj: np.ndarray,
    u_traj: np.ndarray | None = None,
    x_new: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Shift trajectory forward by one step for warm-starting.

    MHE uses a sliding window. When we get a new measurement, we shift
    the window forward. To warm-start the optimizer, we shift the previous
    solution forward and extrapolate the final element.

    Args:
        x_traj: State trajectory (N+1, nx)
        u_traj: Optional input trajectory (N, nu)
        x_new: Optional new state to append (if None, extrapolate)

    Returns:
        Shifted x_traj, shifted u_traj (or None)
    """
    N = len(x_traj) - 1
    x_traj.shape[1] if x_traj.ndim > 1 else 1

    # Shift states: drop first, append new/extrapolated last
    x_shifted = np.zeros_like(x_traj)
    x_shifted[:-1] = x_traj[1:]

    if x_new is not None:
        x_shifted[-1] = x_new
    else:
        # Linear extrapolation from last two points
        if N >= 1:
            x_shifted[-1] = 2 * x_traj[-1] - x_traj[-2]
        else:
            x_shifted[-1] = x_traj[-1]

    # Shift inputs if provided
    u_shifted = None
    if u_traj is not None:
        u_shifted = np.zeros_like(u_traj)
        if len(u_traj) > 1:
            u_shifted[:-1] = u_traj[1:]
            u_shifted[-1] = u_traj[-1]  # Repeat last input
        elif len(u_traj) == 1:
            u_shifted[0] = u_traj[0]

    return x_shifted, u_shifted


def shift_parameters(
    theta: np.ndarray,
    mode: str = "static",
    random_walk_std: np.ndarray | None = None,
) -> np.ndarray:
    """Shift parameter estimate for warm-starting.

    Args:
        theta: Current parameter estimate (ntheta,)
        mode: 'static' (no change) or 'random_walk' (add drift)
        random_walk_std: Std for random walk (if mode='random_walk')

    Returns:
        Shifted parameter estimate
    """
    if mode == "static":
        result: np.ndarray = theta.copy()
        return result
    elif mode == "random_walk":
        if random_walk_std is None:
            result = theta.copy()
            return result
        # Don't actually add noise (that would be wrong), just keep the estimate
        result = theta.copy()
        return result
    else:
        raise ValueError(f"Unknown parameter mode: {mode}")


def compute_arrival_cost_prior(
    x_old: np.ndarray,
    theta_old: np.ndarray,
    P_old: np.ndarray,
    x_new_meas: np.ndarray,
    u_old: np.ndarray,
    model_func: Callable[..., Any],
    Q: np.ndarray,
    R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate arrival cost update using EKF-like propagation.

    Updates the arrival cost prior (x_prior, theta_prior, P_prior) using
    a single EKF prediction step.

    Args:
        x_old: Previous arrival cost state prior
        theta_old: Previous parameter prior
        P_old: Previous covariance (nx+ntheta, nx+ntheta)
        x_new_meas: New measurement (used for correction)
        u_old: Input applied
        model_func: f(x, u, theta) -> x_next
        Q: Process noise covariance
        R: Measurement noise covariance

    Returns:
        Updated (x_prior, theta_prior, P_prior)
    """
    # For simplicity in this implementation, we use a basic approach:
    # predict the state, keep parameters, inflate covariance slightly

    nx = len(x_old)
    len(theta_old)

    # Predict state
    x_pred = np.array(model_func(x_old, u_old, theta_old)).flatten()

    # Parameters stay the same (for static mode)
    theta_pred = theta_old.copy()

    # Simple covariance update: add process noise
    P_pred = P_old.copy()
    P_pred[:nx, :nx] += Q

    return x_pred, theta_pred, P_pred
