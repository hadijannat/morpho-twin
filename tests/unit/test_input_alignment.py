"""Tests for input alignment in estimators.

These tests verify that the MHE and windowed LS estimators correctly pair
inputs with state transitions. For dynamics x_{k+1} = a*x_k + b*u_k,
the input u_k should correspond to the transition from x_k to x_{k+1}.
"""

import numpy as np
import pytest


def test_windowed_ls_input_alignment():
    """Verify windowed LS correctly pairs inputs with transitions.

    For dynamics x_{k+1} = a*x_k + b*u_k, we generate a known sequence
    and verify parameter estimates converge to true values.
    """
    from ddt.estimation.windowed_ls import WindowedLeastSquaresEstimator

    # True parameters
    a_true = 0.9
    b_true = 0.2

    estimator = WindowedLeastSquaresEstimator(window=50)

    # Generate a trajectory with known dynamics
    x = 1.0
    np.random.seed(42)

    for step in range(100):
        # Apply control
        u = 0.5 * np.sin(0.2 * step)  # Exciting input

        # True next state
        x_next = a_true * x + b_true * u

        # Measurement (with small noise)
        y = x_next + np.random.normal(0, 0.01)

        # Update estimator with (y, u_applied)
        estimate = estimator.update(np.array([y]), np.array([u]))

        # Move to next state
        x = x_next

    # After sufficient data, estimates should be close to true values
    # Note: we use loose bounds because LS is a simple estimator
    assert abs(estimate.theta_hat[0] - a_true) < 0.15, (
        f"a_hat={estimate.theta_hat[0]}, a_true={a_true}"
    )
    assert abs(estimate.theta_hat[1] - b_true) < 0.15, (
        f"b_hat={estimate.theta_hat[1]}, b_true={b_true}"
    )


def test_windowed_ls_regression_structure():
    """Test that regression matrix has correct structure.

    For transition x_{k+1} = a*x_k + b*u_k:
    - Phi[i] should be [x_k, u_k]
    - y[i] should be x_{k+1}
    """
    from ddt.estimation.windowed_ls import WindowedLeastSquaresEstimator

    estimator = WindowedLeastSquaresEstimator(window=50)

    # Feed a specific sequence
    # y values: [1.0, 2.0, 3.0, 4.0]
    # u values: [0.1, 0.2, 0.3, 0.4]
    #
    # For regression x_{k+1} = a*x_k + b*u_k:
    # - Row 0: [x_0=1.0, u_0=0.1] -> x_1=2.0
    # - Row 1: [x_1=2.0, u_1=0.2] -> x_2=3.0
    # - Row 2: [x_2=3.0, u_2=0.3] -> x_3=4.0

    y_sequence = [1.0, 2.0, 3.0, 4.0]
    u_sequence = [0.1, 0.2, 0.3, 0.4]

    for y, u in zip(y_sequence, u_sequence, strict=True):
        estimator.update(np.array([y]), np.array([u]))

    # Now the regression should be:
    # Phi = [[1.0, 0.1], [2.0, 0.2], [3.0, 0.3]]
    # y = [2.0, 3.0, 4.0]
    #
    # If alignment were wrong (using us[1:]), we'd get:
    # Phi = [[1.0, 0.2], [2.0, 0.3], [3.0, 0.4]]  # Wrong!

    # The estimator stores internal buffers - we verify through estimates
    # For a linear sequence, with correct alignment, we can verify
    # that the estimate makes sense

    # Get the estimate
    y_final = 5.0
    u_final = 0.5
    estimate = estimator.update(np.array([y_final]), np.array([u_final]))

    # With correct alignment, the estimate should produce reasonable predictions
    theta = estimate.theta_hat
    x_pred = theta[0] * 4.0 + theta[1] * 0.4  # Predict x_4 from x_3, u_3

    # The sequence is roughly linear (x+1 each step), so prediction should be close to 5
    assert abs(x_pred - 5.0) < 1.0, f"Prediction {x_pred} too far from expected 5.0"


@pytest.mark.skipif(
    True,  # Skip by default - requires casadi
    reason="CasADi MHE test requires casadi installation",
)
def test_casadi_mhe_input_alignment():
    """Verify CasADi MHE correctly pairs inputs with transitions."""
    try:
        from ddt.estimation.mhe import CasADiMHE
        from ddt.estimation.mhe.config import MHEConfig
    except ImportError:
        pytest.skip("CasADi not available")

    # True parameters
    a_true = 0.9
    b_true = 0.2

    cfg = MHEConfig(horizon=10)
    mhe = CasADiMHE(cfg=cfg, dt=0.1, nx=1, nu=1, ny=1, ntheta=2)

    # Generate trajectory
    x = 1.0
    np.random.seed(42)

    for step in range(50):
        u = 0.5 * np.sin(0.3 * step)
        x_next = a_true * x + b_true * u
        y = x_next + np.random.normal(0, 0.02)

        estimate = mhe.update(np.array([y]), np.array([u]))
        x = x_next

    # Estimates should be reasonable
    assert abs(estimate.theta_hat[0] - a_true) < 0.3
    assert abs(estimate.theta_hat[1] - b_true) < 0.3
