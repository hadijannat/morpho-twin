"""Fisher Information Matrix computation for dual-control."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def propagate_sensitivity(
    S_prev: np.ndarray,
    A: np.ndarray,
    C_theta: np.ndarray,
) -> np.ndarray:
    """Propagate parameter sensitivity forward in time.

    S_{j+1} = A_j @ S_j + C_j

    where:
        S_j = dx_j/dθ (sensitivity of state to parameters)
        A_j = df/dx at (x_j, u_j, θ)
        C_j = df/dθ at (x_j, u_j, θ)

    Args:
        S_prev: Previous sensitivity matrix (nx, ntheta)
        A: State Jacobian (nx, nx)
        C_theta: Parameter Jacobian (nx, ntheta)

    Returns:
        Updated sensitivity matrix S_next (nx, ntheta)
    """
    result: np.ndarray = A @ S_prev + C_theta
    return result


def compute_fim_prediction(
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    theta: np.ndarray,
    A_func: Callable[..., Any],
    C_func: Callable[..., Any],
    R_inv: np.ndarray,
    S0: np.ndarray | None = None,
) -> np.ndarray:
    """Compute predicted Fisher Information Matrix along trajectory.

    FIM = Σ S_j^T @ R^{-1} @ S_j

    where S_j is the sensitivity of the output to parameters at step j.

    This is used in dual-control to quantify the information content
    of a control trajectory for parameter estimation.

    Args:
        x_traj: State trajectory (N+1, nx)
        u_traj: Input trajectory (N, nu)
        theta: Parameter estimate (ntheta,)
        A_func: Function computing df/dx
        C_func: Function computing df/dtheta
        R_inv: Inverse measurement noise covariance (ny, ny)
        S0: Initial sensitivity (nx, ntheta), defaults to zero

    Returns:
        Fisher Information Matrix (ntheta, ntheta)
    """
    N = len(u_traj)
    nx = x_traj.shape[1] if x_traj.ndim > 1 else 1
    ntheta = len(theta)

    if S0 is None:
        S = np.zeros((nx, ntheta))
    else:
        S = S0.copy()

    fim = np.zeros((ntheta, ntheta))

    for j in range(N):
        x_j = x_traj[j] if x_traj.ndim > 1 else np.array([x_traj[j]])
        u_j = u_traj[j] if u_traj.ndim > 1 else np.array([u_traj[j]])

        # Get Jacobians
        A_j = np.array(A_func(x_j, u_j, theta)).reshape(nx, nx)
        C_j = np.array(C_func(x_j, u_j, theta)).reshape(nx, ntheta)

        # Propagate sensitivity
        S = propagate_sensitivity(S, A_j, C_j)

        # Accumulate FIM (assuming output = state for simplicity)
        fim += S.T @ R_inv @ S

    return fim


def compute_fim_increment(
    S: np.ndarray,
    R_inv: np.ndarray,
) -> np.ndarray:
    """Compute FIM increment from current sensitivity.

    ΔF = S^T @ R^{-1} @ S

    Args:
        S: Sensitivity matrix (ny, ntheta) or (nx, ntheta) if y = x
        R_inv: Inverse measurement noise covariance

    Returns:
        FIM increment (ntheta, ntheta)
    """
    result: np.ndarray = S.T @ R_inv @ S
    return result


def fim_from_trajectory_linearization(
    x_ref: np.ndarray,
    u_ref: np.ndarray,
    theta: np.ndarray,
    model_A: Callable[..., Any],
    model_C: Callable[..., Any],
    R_inv: np.ndarray,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Compute FIM and sensitivities by linearizing along reference trajectory.

    This is useful for RTI-NMPC where we linearize around the previous solution.

    Args:
        x_ref: Reference state trajectory (N+1, nx)
        u_ref: Reference input trajectory (N, nu)
        theta: Parameter estimate
        model_A: df/dx function
        model_C: df/dtheta function
        R_inv: Inverse measurement covariance

    Returns:
        fim: Fisher Information Matrix
        sensitivities: List of sensitivity matrices S_j
    """
    N = len(u_ref)
    nx = x_ref.shape[1] if x_ref.ndim > 1 else 1
    ntheta = len(theta)

    S = np.zeros((nx, ntheta))
    sensitivities = [S.copy()]
    fim = np.zeros((ntheta, ntheta))

    for j in range(N):
        x_j = x_ref[j] if x_ref.ndim > 1 else np.array([x_ref[j]])
        u_j = u_ref[j] if u_ref.ndim > 1 else np.array([u_ref[j]])

        A_j = np.array(model_A(x_j, u_j, theta)).reshape(nx, nx)
        C_j = np.array(model_C(x_j, u_j, theta)).reshape(nx, ntheta)

        S = A_j @ S + C_j
        sensitivities.append(S.copy())
        fim += S.T @ R_inv @ S

    return fim, sensitivities
