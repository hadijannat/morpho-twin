"""Robust margin computation for CBF-QP with parameter uncertainty."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def compute_robust_margin(
    grad_h: np.ndarray,
    theta_cov: np.ndarray,
    gamma: float = 1.0,
    margin_factor: float = 1.0,
) -> float:
    """Compute robust safety margin from parameter uncertainty.

    The robust margin accounts for uncertainty in the system model
    by adding extra conservatism to the CBF constraint:

        Lf·h + Lg·h·u + α·h + ε_robust >= 0

    where:
        ε_robust = γ · |∇h| · σ_θ · margin_factor
        σ_θ = √trace(Σ_θθ)

    Args:
        grad_h: Barrier gradient ∇h(x)
        theta_cov: Parameter covariance matrix Σ_θθ
        gamma: Sensitivity scaling factor
        margin_factor: Additional safety factor (e.g., from mode manager)
            - 1.0 for NORMAL mode
            - 2.0 for CONSERVATIVE mode
            - 5.0 for SAFE_STOP mode

    Returns:
        Robust margin ε_robust (non-negative)
    """
    # Parameter uncertainty (standard deviation scale)
    sigma_theta = np.sqrt(np.trace(theta_cov))

    # Gradient magnitude
    grad_norm = np.linalg.norm(grad_h)

    # Robust margin
    epsilon = gamma * grad_norm * sigma_theta * margin_factor

    return max(0.0, float(epsilon))


def compute_worst_case_dynamics_bound(
    x: np.ndarray,
    u: np.ndarray,
    theta_nom: np.ndarray,
    theta_cov: np.ndarray,
    dynamics_jacobian_theta: Callable[..., Any],
    confidence: float = 0.95,
) -> float:
    """Compute worst-case dynamics error bound.

    For robust CBF, we need to bound:
        |f(x, u, θ) - f(x, u, θ_nom)|

    over the uncertainty set defined by theta_cov.

    Using first-order Taylor expansion:
        |Δf| ≤ |∂f/∂θ| · |Δθ|

    where |Δθ| is bounded by the confidence ellipsoid.

    Args:
        x: Current state
        u: Input
        theta_nom: Nominal parameter estimate
        theta_cov: Parameter covariance
        dynamics_jacobian_theta: Function computing ∂f/∂θ
        confidence: Confidence level for uncertainty bound

    Returns:
        Worst-case dynamics error bound
    """
    from scipy.stats import chi2

    # Jacobian of dynamics w.r.t. parameters
    df_dtheta = np.array(dynamics_jacobian_theta(x, u, theta_nom))

    # Chi-squared quantile for confidence ellipsoid
    ntheta = len(theta_nom)
    chi2_val = chi2.ppf(confidence, df=ntheta)

    # Bound on |Δθ| from confidence ellipsoid
    # |Δθ|² ≤ χ²_{1-α} · λ_max(Σ)
    eigvals = np.linalg.eigvalsh(theta_cov)
    lambda_max = float(np.max(eigvals))
    delta_theta_bound = np.sqrt(chi2_val * lambda_max)

    # Worst-case dynamics error
    jacobian_norm = np.linalg.norm(df_dtheta)
    error_bound = jacobian_norm * delta_theta_bound

    return float(error_bound)


def adaptive_alpha(
    h: float,
    h_min: float = 0.1,
    alpha_nom: float = 0.5,
    alpha_aggressive: float = 2.0,
) -> float:
    """Compute adaptive CBF class-K function parameter.

    When far from the boundary (h large), use nominal alpha.
    When close to boundary (h small), become more aggressive.

    This implements:
        α(h) = α_nom + (α_agg - α_nom) · exp(-h/h_min)

    Args:
        h: Current barrier value
        h_min: Characteristic distance for adaptation
        alpha_nom: Nominal class-K parameter
        alpha_aggressive: Aggressive parameter near boundary

    Returns:
        Adapted alpha value
    """
    if h <= 0:
        # Already violating, use maximum aggression
        return alpha_aggressive

    decay = np.exp(-h / h_min)
    alpha = alpha_nom + (alpha_aggressive - alpha_nom) * decay

    return float(alpha)


def tighten_constraints_for_uncertainty(
    x_min: np.ndarray,
    x_max: np.ndarray,
    theta_cov: np.ndarray,
    sensitivity: np.ndarray,
    confidence: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Tighten state constraints to account for parameter uncertainty.

    Computes tightened constraints:
        x_min_tight = x_min + Δx
        x_max_tight = x_max - Δx

    where Δx accounts for worst-case state deviation due to
    parameter uncertainty.

    Args:
        x_min: Original lower bounds
        x_max: Original upper bounds
        theta_cov: Parameter covariance
        sensitivity: State sensitivity to parameters (dx/dθ)
        confidence: Confidence level

    Returns:
        (x_min_tight, x_max_tight)
    """
    from scipy.stats import chi2

    nx = len(x_min)
    ntheta = theta_cov.shape[0]

    chi2_val = chi2.ppf(confidence, df=ntheta)

    # Compute worst-case state deviation
    # Δx = sensitivity @ Δθ, where |Δθ|² ≤ χ² · λ_max
    eigvals = np.linalg.eigvalsh(theta_cov)
    lambda_max = float(np.max(eigvals))

    # For each state dimension
    delta_x = np.zeros(nx)
    for i in range(nx):
        if sensitivity.ndim > 1:
            sens_i = sensitivity[i] if i < len(sensitivity) else sensitivity[0]
        else:
            sens_i = sensitivity
        sens_norm = np.linalg.norm(sens_i)
        delta_x[i] = sens_norm * np.sqrt(chi2_val * lambda_max)

    x_min_tight = x_min + delta_x
    x_max_tight = x_max - delta_x

    # Ensure constraints remain valid
    midpoint = (x_min + x_max) / 2
    x_min_tight = np.minimum(x_min_tight, midpoint)
    x_max_tight = np.maximum(x_max_tight, midpoint)

    return x_min_tight, x_max_tight
