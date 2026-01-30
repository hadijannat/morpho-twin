"""CBF-QP Safety Filter with OSQP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

try:
    import osqp
    from scipy import sparse

    OSQP_AVAILABLE = True
except ImportError:
    OSQP_AVAILABLE = False

from ..interfaces import Estimate, SafetyFilter
from .barriers import CompositeBarrier, create_box_barriers
from .robust_margin import adaptive_alpha, compute_robust_margin

if TYPE_CHECKING:
    pass


@dataclass
class CBFQPSafetyFilter(SafetyFilter):
    """Control Barrier Function QP Safety Filter.

    Solves at each step:
        u_safe = argmin |u - u_nom|² + ρ|δ|²

        s.t.  Lf·h + Lg·h·u + α·h + ε_robust ≥ -δ   (barrier)
              u_min ≤ u ≤ u_max                      (actuator)
              δ ≥ 0                                   (slack)

    The robust margin ε_robust is computed from parameter uncertainty
    to ensure safety under model uncertainty.
    """

    # Dimensions
    nx: int = 1
    nu: int = 1
    ntheta: int = 2
    dt: float = 0.1

    # Constraints
    x_min: np.ndarray = field(default_factory=lambda: np.array([-1.0]))
    x_max: np.ndarray = field(default_factory=lambda: np.array([1.0]))
    u_min: np.ndarray = field(default_factory=lambda: np.array([-1.0]))
    u_max: np.ndarray = field(default_factory=lambda: np.array([1.0]))

    # CBF parameters
    alpha: float = 0.5  # Class-K function parameter
    slack_weight: float = 1000.0  # Penalty on constraint violation
    gamma_robust: float = 1.0  # Robust margin scaling

    # Mode-dependent margin factor (set by supervisor)
    margin_factor: float = 1.0

    # Internal
    _barrier: CompositeBarrier | None = field(default=None, init=False)
    _qp_solver: osqp.OSQP | None = field(default=None, init=False)
    _qp_initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if not OSQP_AVAILABLE:
            raise ImportError(
                "OSQP required for CBFQPSafetyFilter. "
                "Install with: pip install osqp"
            )

        # Create barrier function for box constraints
        self._barrier = create_box_barriers(self.x_min, self.x_max)

    def reset(self) -> None:
        """Reset filter state."""
        self._qp_initialized = False
        self._qp_solver = None

    def set_margin_factor(self, factor: float) -> None:
        """Set margin factor from supervisor mode.

        Args:
            factor: Safety margin multiplier
                - 1.0 for NORMAL
                - 2.0 for CONSERVATIVE
                - 5.0 for SAFE_STOP
        """
        self.margin_factor = max(1.0, factor)

    def filter(self, u_nom: np.ndarray, est: Estimate) -> np.ndarray:
        """Filter nominal input to ensure safety.

        Args:
            u_nom: Nominal control input from controller
            est: Current state and parameter estimate

        Returns:
            Safe control input u_safe
        """
        u_nom = np.atleast_1d(np.asarray(u_nom, dtype=np.float64))
        x = np.atleast_1d(np.asarray(est.x_hat, dtype=np.float64))
        theta = np.atleast_1d(np.asarray(est.theta_hat, dtype=np.float64))
        theta_cov = np.atleast_2d(np.asarray(est.theta_cov, dtype=np.float64))

        # Build and solve QP
        u_safe = self._solve_cbf_qp(u_nom, x, theta, theta_cov)

        return u_safe

    def _solve_cbf_qp(
        self,
        u_nom: np.ndarray,
        x: np.ndarray,
        theta: np.ndarray,
        theta_cov: np.ndarray,
    ) -> np.ndarray:
        """Solve the CBF-QP.

        Decision variable: z = [u; δ] where δ is slack

        min  0.5 * z' P z + q' z
        s.t. l <= A z <= u
        """
        nu = self.nu
        n_slack = 1  # One slack variable

        # Get dynamics at current state (for discrete-time CBF)
        # x_next = a*x + b*u for linear system
        a = theta[0] if len(theta) > 0 else 1.0
        b = theta[1] if len(theta) > 1 else 0.1

        # For discrete-time CBF on box constraints:
        # h(x) = x_max - x  (upper bound)
        # h_next = x_max - (a*x + b*u) >= (1-α)*h
        # => -b*u >= (1-α)*(x_max - x) - x_max + a*x - ε_robust
        # => -b*u >= a*x - α*x_max + (α-1)*x - ε_robust
        # => b*u <= x_max - a*x + (1-α)*(x_max - x) + ε_robust

        # Get barrier value and gradient
        assert self._barrier is not None
        h = self._barrier.evaluate(x)
        grad_h = self._barrier.gradient(x)

        # Robust margin from uncertainty
        epsilon = compute_robust_margin(
            grad_h=grad_h,
            theta_cov=theta_cov,
            gamma=self.gamma_robust,
            margin_factor=self.margin_factor,
        )

        # Adaptive alpha based on barrier value
        alpha_eff = adaptive_alpha(h, alpha_nom=self.alpha)

        # Build QP matrices
        # Cost: 0.5 * ||u - u_nom||² + 0.5 * ρ * δ²
        P_data = np.eye(nu + n_slack)
        P_data[nu:, nu:] = self.slack_weight * np.eye(n_slack)
        P = sparse.csc_matrix(P_data)

        q = np.zeros(nu + n_slack)
        q[:nu] = -u_nom

        # Constraints
        # 1. CBF constraint: Lf_h + Lg_h*u + alpha*h + epsilon >= -delta
        #    For discrete time: h(f(x,u)) >= (1-alpha)*h - epsilon + delta
        #    Rearranged: Lg_h*u + delta >= -Lf_h - alpha*h - epsilon

        # For linear dynamics x+ = a*x + b*u:
        # Using barrier h = min(x - x_min, x_max - x)
        # If lower bound active: h_next = a*x + b*u - x_min
        # If upper bound active: h_next = x_max - a*x - b*u

        # Simplified: use all barrier constraints
        assert self._barrier is not None
        constraints = self._barrier.get_all_constraints(
            x=x,
            f=np.array([a * x[0]]),  # f(x) for autonomous part
            g=np.array([[b]]),  # g(x) for input
            alpha=alpha_eff,
        )

        n_barriers = len(constraints)

        # Build constraint matrix
        # [CBF constraints   ]   [lower_cbf ]    [upper_cbf ]
        # [input bounds      ] * z in [u_min    ] to [u_max     ]
        # [slack non-negative]   [0         ]    [inf       ]

        n_constraints = n_barriers + nu + n_slack

        A_dense = np.zeros((n_constraints, nu + n_slack))
        lb = np.full(n_constraints, -np.inf)
        ub = np.full(n_constraints, np.inf)

        row = 0

        # CBF constraints: Lg_h @ u + delta >= -Lf_h - alpha*h - epsilon
        for Lf_h, Lg_h, h_i in constraints:
            A_dense[row, :nu] = Lg_h
            A_dense[row, nu:] = 1.0  # Slack
            lb[row] = -Lf_h - alpha_eff * h_i - epsilon
            ub[row] = np.inf
            row += 1

        # Input bounds: u_min <= u <= u_max
        for i in range(nu):
            A_dense[row, i] = 1.0
            lb[row] = self.u_min[i]
            ub[row] = self.u_max[i]
            row += 1

        # Slack non-negative: delta >= 0
        for i in range(n_slack):
            A_dense[row, nu + i] = 1.0
            lb[row] = 0.0
            ub[row] = np.inf
            row += 1

        A = sparse.csc_matrix(A_dense)

        # Solve QP
        if not self._qp_initialized:
            self._qp_solver = osqp.OSQP()
            self._qp_solver.setup(
                P=P,
                q=q,
                A=A,
                l=lb,
                u=ub,
                verbose=False,
                eps_abs=1e-6,
                eps_rel=1e-6,
            )
            self._qp_initialized = True
        else:
            assert self._qp_solver is not None
            self._qp_solver.update(q=q, l=lb, u=ub, Px=P.data, Ax=A.data)

        assert self._qp_solver is not None
        result = self._qp_solver.solve()

        if result.info.status != "solved":
            # Fallback: clamp to input bounds
            fallback: np.ndarray = np.clip(u_nom, self.u_min, self.u_max)
            return fallback

        u_safe = result.x[:nu]
        slack = result.x[nu:]

        # Warn if significant slack used
        if np.max(slack) > 0.01:
            # Could log this for monitoring
            pass

        u_result: np.ndarray = u_safe
        return u_result

    def get_barrier_value(self, x: np.ndarray) -> float:
        """Get current barrier function value."""
        x = np.atleast_1d(x)
        assert self._barrier is not None
        return self._barrier.evaluate(x)

    def is_safe(self, x: np.ndarray, margin: float = 0.0) -> bool:
        """Check if state is in safe set."""
        return self.get_barrier_value(x) >= margin
