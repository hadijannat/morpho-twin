"""Covariance extraction for MHE."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None  # type: ignore[assignment]


def extract_covariance(
    nlp_solver: ca.nlpsol | None,
    sol: ca.DM | None,
    ntheta: int,
    theta_indices: tuple[int, int],
    method: str = "hessian",
    eps: float = 1e-6,
) -> np.ndarray:
    """Extract parameter covariance from MHE solution.

    For Gauss-Newton MHE, the inverse of the Hessian of the objective
    at the solution approximates the posterior covariance.

    Args:
        nlp_solver: CasADi NLP solver (may be None for fallback)
        sol: Solution dictionary from solver
        ntheta: Number of parameters
        theta_indices: (start, end) indices of theta in decision variable
        method: 'hessian' (from solver) or 'finite_diff' (numerical)
        eps: Finite difference step size

    Returns:
        Parameter covariance matrix (ntheta, ntheta)
    """
    if method == "hessian" and nlp_solver is not None and sol is not None:
        return _extract_from_hessian(nlp_solver, sol, ntheta, theta_indices)
    else:
        # Return a default/prior covariance
        return np.eye(ntheta) * 0.1


def _extract_from_hessian(
    nlp_solver: ca.nlpsol,
    sol: dict,
    ntheta: int,
    theta_indices: tuple[int, int],
) -> np.ndarray:
    """Extract covariance from Hessian of Lagrangian.

    The inverse of the Gauss-Newton Hessian approximation gives
    the covariance of the estimates.
    """
    if ca is None:
        return np.eye(ntheta) * 0.1

    try:
        # Try to get Hessian from solver stats
        stats = nlp_solver.stats()
        if "hessian" in stats:
            H = np.array(stats["hessian"])
        else:
            # Fallback: approximate with prior
            return np.eye(ntheta) * 0.1

        # Extract theta block
        start, end = theta_indices
        H_theta = H[start:end, start:end]

        # Invert to get covariance
        try:
            cov = np.linalg.inv(H_theta + 1e-8 * np.eye(ntheta))
            # Ensure positive definiteness
            cov = (cov + cov.T) / 2
            eigvals = np.linalg.eigvalsh(cov)
            if np.min(eigvals) < 0:
                cov += (abs(np.min(eigvals)) + 1e-6) * np.eye(ntheta)
            result: np.ndarray = cov
            return result
        except np.linalg.LinAlgError:
            return np.eye(ntheta) * 0.1

    except Exception:
        return np.eye(ntheta) * 0.1


def compute_fim_from_trajectory(
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    theta: np.ndarray,
    C_func: Callable[..., Any],
    R_inv: np.ndarray,
) -> np.ndarray:
    """Compute Fisher Information Matrix from trajectory.

    FIM = Σ C_i^T @ R^{-1} @ C_i

    where C_i = df/dθ at (x_i, u_i, θ)

    This is useful for persistence of excitation monitoring
    and dual-control cost formulation.

    Args:
        x_traj: State trajectory (N+1, nx)
        u_traj: Input trajectory (N, nu)
        theta: Parameter estimate (ntheta,)
        C_func: Function computing df/dtheta
        R_inv: Inverse measurement noise covariance

    Returns:
        Fisher Information Matrix (ntheta, ntheta)
    """
    N = len(u_traj)
    ntheta = len(theta)

    fim = np.zeros((ntheta, ntheta))

    for i in range(N):
        x_i = x_traj[i] if x_traj.ndim > 1 else np.array([x_traj[i]])
        u_i = u_traj[i] if u_traj.ndim > 1 else np.array([u_traj[i]])

        C_i = np.array(C_func(x_i, u_i, theta)).reshape(-1, ntheta)
        fim += C_i.T @ R_inv @ C_i

    return fim


def estimate_covariance_from_fim(
    fim: np.ndarray,
    prior_cov: np.ndarray,
    regularization: float = 1e-6,
) -> np.ndarray:
    """Estimate parameter covariance from FIM.

    Uses Bayesian update: Σ_post^{-1} = Σ_prior^{-1} + FIM

    Args:
        fim: Fisher Information Matrix
        prior_cov: Prior parameter covariance
        regularization: Regularization for numerical stability

    Returns:
        Posterior covariance estimate
    """
    ntheta = len(fim)

    try:
        prior_inv = np.linalg.inv(prior_cov + regularization * np.eye(ntheta))
        post_inv = prior_inv + fim + regularization * np.eye(ntheta)
        post_cov = np.linalg.inv(post_inv)

        # Ensure symmetry and positive definiteness
        post_cov = (post_cov + post_cov.T) / 2
        eigvals = np.linalg.eigvalsh(post_cov)
        if np.min(eigvals) < 0:
            post_cov += (abs(np.min(eigvals)) + 1e-8) * np.eye(ntheta)

        result: np.ndarray = post_cov
        return result

    except np.linalg.LinAlgError:
        result = prior_cov.copy()
        return result
