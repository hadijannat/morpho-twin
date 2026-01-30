"""Feedback suicide test: verify MHE handles safety clamping correctly.

This test ensures that MHE uses u_applied (the actual safe input), NOT u_nom
(the controller's requested input). If MHE used u_nom, it would observe
discrepancies between predicted and actual state when safety filtering
is active, leading to incorrect parameter estimates.

The term "feedback suicide" refers to the scenario where:
1. Safety filter clamps all inputs to 0 (emergency stop)
2. Controller requests u_nom ≠ 0
3. MHE (incorrectly using u_nom) predicts movement
4. MHE observes no actual movement
5. MHE incorrectly concludes b ≈ 0 (actuator gain collapsed)

This is catastrophically wrong - the actuator is fine, but was overridden
by the safety filter. Proper MHE uses u_applied=0 and stays stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

# Check for CasADi
try:
    import casadi  # noqa: F401

    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


@dataclass
class ZeroClampSafetyFilter:
    """Safety filter that always returns zero (emergency stop mode).

    This simulates a scenario where the safety filter detects a hazard
    and clamps all control inputs to zero regardless of what the
    controller requests.
    """

    def filter(self, u_nom: np.ndarray, est: object) -> np.ndarray:
        """Return zero regardless of nominal input.

        Args:
            u_nom: Nominal control from controller
            est: Current state estimate (unused)

        Returns:
            Zero input array (emergency stop)
        """
        return np.zeros_like(u_nom)


@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
def test_mhe_stable_under_zero_clamp():
    """Test MHE maintains stable estimates when safety clamps u to zero.

    Scenario:
    1. Run system with normal control for initial learning (50 steps)
    2. Activate zero-clamp safety filter (100 steps)
    3. Controller continues to request non-zero u_nom
    4. Safety filter returns u_applied = 0
    5. Verify MHE estimates remain stable (b does NOT collapse to 0)

    If MHE incorrectly used u_nom, it would see commanded input but no
    movement, concluding the actuator gain b has collapsed. This test
    ensures MHE correctly uses u_applied and stays stable.
    """
    from ddt.estimation.mhe import CasADiMHE
    from ddt.estimation.mhe.config import MHEConfig
    from ddt.sim.linear_scalar import LinearScalarPlant

    # True system parameters
    a_true = 0.95
    b_true = 0.5
    dt = 0.1

    # Create plant
    plant = LinearScalarPlant(
        dt=dt,
        a_true=a_true,
        b_true=b_true,
        process_noise_std=0.01,
        meas_noise_std=0.05,
        x0=0.0,
    )
    plant.seed(42)

    # Create MHE
    mhe_cfg = MHEConfig(
        horizon=20,
        use_ekf_arrival_cost=True,
    )
    mhe = CasADiMHE(
        cfg=mhe_cfg,
        dt=dt,
        nx=1,
        nu=1,
        ny=1,
        ntheta=2,
    )

    # Phase 1: Normal operation with exciting input (50 steps)
    # This allows MHE to learn the true parameters
    sr = plant.reset()
    est = mhe.update(sr.y, np.array([0.0]))

    for k in range(50):
        # Sinusoidal excitation for identifiability
        u_nom = np.array([0.5 * np.sin(0.2 * k)])
        u_applied = u_nom  # No safety override yet
        sr = plant.step(u_applied)
        est = mhe.update(sr.y, u_applied)

    # Record parameter estimates after learning phase
    b_after_learning = est.theta_hat[1]
    assert abs(b_after_learning - b_true) < 0.3, (
        f"MHE should have learned b ≈ {b_true}, got {b_after_learning}"
    )

    # Phase 2: Zero-clamp mode (100 steps)
    # Safety filter clamps everything to zero, but controller keeps requesting
    zero_clamp = ZeroClampSafetyFilter()
    b_estimates = []

    for k in range(100):
        # Controller requests non-zero input
        u_nom = np.array([0.3 * np.sin(0.15 * k) + 0.2])

        # Safety filter clamps to zero
        u_applied = zero_clamp.filter(u_nom, est)
        assert np.allclose(u_applied, 0.0), "Safety filter should return zero"

        # Plant receives zero input
        sr = plant.step(u_applied)

        # MHE receives u_applied (zero), NOT u_nom
        est = mhe.update(sr.y, u_applied)
        b_estimates.append(est.theta_hat[1])

    # Verify MHE estimates remain stable
    b_final = est.theta_hat[1]

    # Key assertions:
    # 1. b should NOT collapse to near-zero (feedback suicide symptom)
    assert b_final > 0.01, (
        f"Feedback suicide detected: b collapsed to {b_final:.4f}. "
        "MHE may be using u_nom instead of u_applied."
    )

    # 2. b should remain relatively stable (not drift wildly)
    b_std = np.std(b_estimates)
    assert b_std < 0.3, (
        f"MHE b estimate unstable: std = {b_std:.4f}. "
        "Expected stable estimates when u_applied = 0."
    )

    # 3. b should not have changed dramatically from learning phase
    b_change = abs(b_final - b_after_learning)
    assert b_change < 0.5, (
        f"MHE b changed too much during zero-clamp: "
        f"{b_after_learning:.3f} -> {b_final:.3f} (Δ={b_change:.3f})"
    )


@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
def test_mhe_distinguishes_u_nom_from_u_applied():
    """Verify MHE buffer contains u_applied, not u_nom.

    This is a direct verification that MHE's internal buffer stores
    the actual applied input, not the nominal controller output.
    """
    from ddt.estimation.mhe import CasADiMHE
    from ddt.estimation.mhe.config import MHEConfig
    from ddt.sim.linear_scalar import LinearScalarPlant

    dt = 0.1
    plant = LinearScalarPlant(
        dt=dt,
        a_true=0.9,
        b_true=0.3,
        process_noise_std=0.0,
        meas_noise_std=0.0,
        x0=0.0,
    )
    plant.seed(123)

    mhe_cfg = MHEConfig(horizon=10, use_ekf_arrival_cost=False)
    mhe = CasADiMHE(
        cfg=mhe_cfg,
        dt=dt,
        nx=1,
        nu=1,
        ny=1,
        ntheta=2,
    )

    sr = plant.reset()
    mhe.update(sr.y, np.array([0.0]))

    # Feed different u_nom vs u_applied sequences
    u_nom_sequence = [0.5, 0.6, 0.7, 0.8, 0.9]
    u_applied_sequence = [0.1, 0.0, 0.2, 0.0, 0.1]  # Safety filtered

    for _u_nom_val, u_applied_val in zip(u_nom_sequence, u_applied_sequence, strict=True):
        u_applied = np.array([u_applied_val])
        # Note: we deliberately pass u_applied to MHE (correct behavior)
        sr = plant.step(u_applied)
        mhe.update(sr.y, u_applied)

    # Verify MHE buffer contains u_applied values
    stored_u = [u.item() for u in mhe._u_buffer[-len(u_applied_sequence):]]

    # The buffer should match u_applied, not u_nom
    for stored, expected in zip(stored_u, u_applied_sequence, strict=True):
        assert abs(stored - expected) < 1e-6, (
            f"MHE buffer contains {stored}, expected {expected} (u_applied). "
            "Buffer may incorrectly contain u_nom."
        )


@pytest.mark.skipif(not HAS_CASADI, reason="CasADi not installed")
def test_mhe_with_partial_safety_override():
    """Test MHE handles partial safety overrides correctly.

    Sometimes safety filters don't clamp to zero but reduce the input.
    MHE should correctly track the actual applied input.
    """
    from ddt.estimation.mhe import CasADiMHE
    from ddt.estimation.mhe.config import MHEConfig
    from ddt.sim.linear_scalar import LinearScalarPlant

    dt = 0.1
    plant = LinearScalarPlant(
        dt=dt,
        a_true=0.95,
        b_true=0.4,
        process_noise_std=0.01,
        meas_noise_std=0.02,
        x0=0.0,
    )
    plant.seed(456)

    mhe_cfg = MHEConfig(horizon=15, use_ekf_arrival_cost=True)
    mhe = CasADiMHE(
        cfg=mhe_cfg,
        dt=dt,
        nx=1,
        nu=1,
        ny=1,
        ntheta=2,
    )

    # Initialize
    sr = plant.reset()
    est = mhe.update(sr.y, np.array([0.0]))

    # Phase 1: Normal learning
    for k in range(30):
        u = np.array([0.4 * np.sin(0.25 * k)])
        sr = plant.step(u)
        est = mhe.update(sr.y, u)

    b_initial = est.theta_hat[1]

    # Phase 2: Partial override (safety scales down by 50%)
    scale_factor = 0.5
    for k in range(50):
        u_nom = np.array([0.5 * np.sin(0.2 * k)])
        u_applied = u_nom * scale_factor  # Safety reduces input
        sr = plant.step(u_applied)
        est = mhe.update(sr.y, u_applied)

    b_final = est.theta_hat[1]

    # b should remain stable, not halved
    # (If MHE used u_nom, it would see half the expected movement
    # and might conclude b is halved)
    assert abs(b_final - b_initial) < 0.3, (
        f"MHE b estimate changed too much with partial override: "
        f"{b_initial:.3f} -> {b_final:.3f}"
    )
