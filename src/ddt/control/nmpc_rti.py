"""RTI-NMPC using acados (primary, high-performance)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from acados_template import AcadosOcpSolver  # noqa: F401

    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False

from .dual_control.fim_compute import fim_from_trajectory_linearization
from .nmpc_base import NMPCBase, freeze_theta  # noqa: F401
from .ocp import build_constraint_vectors, build_discrete_dynamics, build_tracking_cost
from .ocp.dynamics import DiscreteDynamics
from .solvers.acados_backend import AcadosSolver

if TYPE_CHECKING:
    pass


@dataclass
class RTINMPC(NMPCBase):
    """Real-Time Iteration NMPC using acados + HPIPM.

    Implements the RTI two-phase workflow:
    1. Prepare phase (background): Linearize, build QP, factorize KKT, compute FIM
    2. Feedback phase (real-time): Update x0, solve single SQP iteration

    This is the primary, high-performance implementation for real-time control.
    """

    rti_mode: bool = True
    max_sqp_iter: int = 1  # 1 for RTI
    tol: float = 1e-6

    _solver: AcadosSolver | None = field(default=None, init=False)
    _dynamics: DiscreteDynamics | Any = field(init=False)
    _prepared: bool = field(default=False, init=False)
    _prepare_theta: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not ACADOS_AVAILABLE:
            raise ImportError(
                "acados required for RTINMPC. "
                "Install with: pip install acados_template casadi"
            )

        super().__post_init__()

        # Build dynamics
        self._dynamics = build_discrete_dynamics(
            dt=self.dt,
            nx=self.nx,
            nu=self.nu,
            ntheta=self.ntheta,
        )

        # Build cost
        cost = build_tracking_cost(
            Q_diag=np.diag(self.Q).tolist(),
            R_diag=np.diag(self.R_u).tolist(),
            Q_terminal_scale=1.0,
        )
        cost.lambda_info = self.lambda_info

        # Build constraints
        constraints = build_constraint_vectors(
            x_min=self.x_min.tolist(),
            x_max=self.x_max.tolist(),
            u_min=self.u_min.tolist(),
            u_max=self.u_max.tolist(),
        )

        # Create solver
        self._solver = AcadosSolver(
            dynamics=self._dynamics,
            cost=cost,
            constraints=constraints,
            horizon=self.horizon,
            dt=self.dt,
            rti_mode=self.rti_mode,
            max_sqp_iter=self.max_sqp_iter,
            tol=self.tol,
        )

    def prepare(self, theta: np.ndarray) -> np.ndarray:
        """RTI prepare phase: linearize and build QP.

        This can be called in advance (e.g., in a background thread)
        before the measurement arrives. It also computes the predicted
        FIM for dual-control.

        Args:
            theta: Parameter estimate for linearization

        Returns:
            Predicted FIM from linearization
        """
        theta_frozen = freeze_theta(theta)
        assert self._solver is not None
        self._solver.prepare(theta_frozen)
        self._prepared = True
        self._prepare_theta = theta_frozen

        # Compute FIM from current trajectory linearization
        R_inv = np.eye(self.nu)  # Simplified for now
        fim, _ = fim_from_trajectory_linearization(
            x_ref=self._x_traj,
            u_ref=self._u_traj,
            theta=theta_frozen,
            model_A=self._dynamics.A,
            model_C=self._dynamics.C,
            R_inv=R_inv,
        )

        return fim

    def feedback(
        self,
        x0: np.ndarray,
        ref: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """RTI feedback phase: solve QP with updated x0.

        This is the time-critical part. If prepare() was not called,
        it will be called automatically.

        Args:
            x0: Current state measurement
            ref: Reference state
            theta: Parameter estimate (used if not prepared)

        Returns:
            u: First optimal input
            x_pred: Predicted state trajectory
        """
        # Ensure prepared
        if not self._prepared:
            self.prepare(theta)

        # Feedback
        assert self._solver is not None
        u_opt, x_opt, success = self._solver.feedback(x0, ref)

        if not success:
            # Fallback to previous or zero
            if self._initialized:
                u_opt = self._u_traj.copy()
                x_opt = self._x_traj.copy()
            else:
                u_opt = np.zeros((self.horizon, self.nu))
                x_opt = np.tile(x0, (self.horizon + 1, 1))

        # Store for next iteration
        self._u_traj = u_opt
        self._x_traj = x_opt
        self._initialized = True
        self._prepared = False  # Reset for next cycle

        return u_opt[0].copy(), x_opt

    def _solve_nmpc(
        self,
        x0: np.ndarray,
        ref: np.ndarray,
        theta: np.ndarray,
        theta_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve NMPC (combined prepare + feedback for Controller interface)."""
        # Prepare if not already done
        fim = self.prepare(theta)

        # Feedback
        u_first, x_opt = self.feedback(x0, ref, theta)

        # Get full trajectory
        u_opt = self._u_traj.copy()

        # Apply probing modification if lambda_info > 0
        if self.lambda_info > 0:
            u_opt = self._apply_probing(u_opt, fim, theta_cov)

        return u_opt, x_opt, fim

    def _apply_probing(
        self,
        u_opt: np.ndarray,
        fim: np.ndarray,
        theta_cov: np.ndarray,
    ) -> np.ndarray:
        """Apply probing modification for dual-control."""
        uncertainty = np.trace(theta_cov)

        if uncertainty > 0.5:
            u_modified = u_opt.copy()
            for k in range(len(u_opt)):
                probe = 0.1 * np.sin(2 * np.pi * k / max(self.horizon, 1))
                u_probe = u_opt[k] + self.lambda_info * probe * uncertainty
                u_modified[k] = np.clip(u_probe, self.u_min, self.u_max)
            result: np.ndarray = u_modified
            return result

        return u_opt

    def get_timing_info(self) -> dict[str, float]:
        """Get solver timing information for real-time analysis."""
        if self._solver is not None and self._solver._acados_solver is not None:
            stats = self._solver._acados_solver.get_stats("time_tot")
            return {"total_time_s": float(stats) if stats else 0.0}
        return {"total_time_s": 0.0}
