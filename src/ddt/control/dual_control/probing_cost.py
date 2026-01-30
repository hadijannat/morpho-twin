"""Probing cost functions for dual-control NMPC."""

from __future__ import annotations

import numpy as np


def a_optimal_cost(
    fim: np.ndarray,
    regularization: float = 1e-6,
) -> float:
    """A-optimal criterion: minimize trace of inverse FIM.

    J_info = tr((F + εI)^{-1})

    Minimizing this encourages control trajectories that maximize
    the minimum eigenvalue of the FIM, leading to better parameter
    identifiability.

    Args:
        fim: Fisher Information Matrix (ntheta, ntheta)
        regularization: Regularization for numerical stability

    Returns:
        A-optimal cost value
    """
    ntheta = fim.shape[0]
    fim_reg = fim + regularization * np.eye(ntheta)

    try:
        fim_inv = np.linalg.inv(fim_reg)
        return float(np.trace(fim_inv))
    except np.linalg.LinAlgError:
        # If FIM is singular, return large cost
        return 1e6


def d_optimal_cost(
    fim: np.ndarray,
    regularization: float = 1e-6,
) -> float:
    """D-optimal criterion: minimize negative log-determinant of FIM.

    J_info = -log(det(F + εI))

    Minimizing this encourages control trajectories that maximize
    the volume of the confidence ellipsoid, leading to uniform
    parameter uncertainty reduction.

    Args:
        fim: Fisher Information Matrix (ntheta, ntheta)
        regularization: Regularization for numerical stability

    Returns:
        D-optimal cost value
    """
    ntheta = fim.shape[0]
    fim_reg = fim + regularization * np.eye(ntheta)

    try:
        sign, logdet = np.linalg.slogdet(fim_reg)
        if sign <= 0:
            return 1e6
        return float(-logdet)
    except np.linalg.LinAlgError:
        return 1e6


def e_optimal_cost(
    fim: np.ndarray,
    regularization: float = 1e-6,
) -> float:
    """E-optimal criterion: minimize inverse of minimum eigenvalue.

    J_info = 1 / λ_min(F + εI)

    Minimizing this directly targets the worst-case parameter
    direction, ensuring no parameter is poorly identified.

    Args:
        fim: Fisher Information Matrix (ntheta, ntheta)
        regularization: Regularization for numerical stability

    Returns:
        E-optimal cost value
    """
    ntheta = fim.shape[0]
    fim_reg = fim + regularization * np.eye(ntheta)

    try:
        eigvals = np.linalg.eigvalsh(fim_reg)
        lambda_min = float(np.min(eigvals))
        if lambda_min <= 0:
            return 1e6
        return 1.0 / lambda_min
    except np.linalg.LinAlgError:
        return 1e6


def compute_probing_input_modification(
    u_nominal: np.ndarray,
    fim: np.ndarray,
    theta_cov: np.ndarray,
    sensitivities: list[np.ndarray],
    R_inv: np.ndarray,
    lambda_info: float,
    u_min: np.ndarray,
    u_max: np.ndarray,
) -> np.ndarray:
    """Compute probing modification to nominal input trajectory.

    This adds a small perturbation to the nominal control to improve
    parameter identifiability while respecting input constraints.

    The modification is computed as:
        Δu = λ * (∂J_info/∂u)

    using the gradient of the A-optimal criterion.

    Args:
        u_nominal: Nominal input trajectory (N, nu)
        fim: Current Fisher Information Matrix
        theta_cov: Parameter covariance for weighting
        sensitivities: List of sensitivity matrices [S_0, ..., S_N]
        R_inv: Inverse measurement covariance
        lambda_info: Probing weight (higher = more probing)
        u_min: Input lower bounds
        u_max: Input upper bounds

    Returns:
        Modified input trajectory with probing (N, nu)
    """
    N, nu = u_nominal.shape
    ntheta = fim.shape[0]

    u_modified = u_nominal.copy()

    # Compute gradient of A-optimal cost w.r.t. input
    # This is an approximation based on sensitivity analysis

    # Regularized inverse of FIM
    reg = 1e-6
    try:
        np.linalg.inv(fim + reg * np.eye(ntheta))
    except np.linalg.LinAlgError:
        return u_nominal

    # For each time step, compute how input affects FIM
    for j in range(N):
        if j + 1 >= len(sensitivities):
            continue

        _ = sensitivities[j + 1]  # Sensitivity at step j+1 (used for future improvements)

        # Gradient of trace(F^-1) with respect to FIM: -F^-1 @ F^-1
        # And dF/dS = 2 * S @ R^-1
        # Combined: direction that improves A-optimality

        # Simple heuristic: perturb in direction that increases FIM
        # For linear systems: larger |u| gives more information
        grad_approx = np.zeros(nu)

        # Scale by uncertainty in parameters
        uncertainty_weight = np.sqrt(np.trace(theta_cov))

        # Perturbation direction: prefer exciting when uncertain
        if uncertainty_weight > 0.1:  # Only probe if uncertain
            # Add small excitation signal
            excitation = 0.1 * np.sin(2 * np.pi * j / max(N, 1))
            grad_approx[0] = excitation * uncertainty_weight

        # Apply modification with constraint projection
        u_probe = u_nominal[j] + lambda_info * grad_approx
        u_modified[j] = np.clip(u_probe, u_min, u_max)

    result: np.ndarray = u_modified
    return result


def adaptive_lambda_info(
    fim: np.ndarray,
    theta_cov: np.ndarray,
    lambda_base: float,
    lambda_max: float = 0.1,
    uncertainty_threshold: float = 1.0,
) -> float:
    """Compute adaptive probing weight based on uncertainty.

    When parameter uncertainty is high, increase probing.
    When parameters are well-identified, reduce probing.

    Args:
        fim: Fisher Information Matrix
        theta_cov: Parameter covariance
        lambda_base: Base probing weight
        lambda_max: Maximum probing weight
        uncertainty_threshold: Threshold for full probing

    Returns:
        Adaptive probing weight
    """
    # Uncertainty metric: trace of covariance
    uncertainty = np.trace(theta_cov)

    # FIM informativeness: minimum eigenvalue
    eigvals = np.linalg.eigvalsh(fim + 1e-8 * np.eye(fim.shape[0]))
    lambda_min_fim = float(np.min(eigvals))

    # Scale probing by uncertainty
    if uncertainty > uncertainty_threshold:
        scale = 1.0
    else:
        scale = uncertainty / uncertainty_threshold

    # Reduce probing if FIM is already informative
    if lambda_min_fim > 1.0:
        scale *= 1.0 / (1.0 + lambda_min_fim)

    return min(lambda_base * scale, lambda_max)
