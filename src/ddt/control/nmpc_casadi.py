"""CasADi-based NMPC implementation (fallback)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None  # type: ignore[assignment]

from .dual_control import compute_fim_prediction
from .nmpc_base import NMPCBase
from .ocp import build_constraint_vectors, build_discrete_dynamics, build_tracking_cost
from .ocp.dynamics import DiscreteDynamics
from .solvers.casadi_backend import CasADiSolver

if TYPE_CHECKING:
    pass


@dataclass
class CasADiNMPC(NMPCBase):
    """NMPC using CasADi + IPOPT.

    This is the fallback implementation when acados is not available.
    Supports dual-control probing for parameter learning.
    """

    max_iter: int = 50
    tol: float = 1e-6
    print_level: int = 0
    fim_criterion: str = "d_optimal"

    _solver: CasADiSolver | None = field(default=None, init=False)
    _dynamics: DiscreteDynamics | Any = field(init=False)
    _w_prev: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if ca is None:
            raise ImportError(
                "CasADi required for CasADiNMPC. "
                "Install with: pip install casadi"
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
        self._solver = CasADiSolver(
            dynamics=self._dynamics,
            cost=cost,
            constraints=constraints,
            horizon=self.horizon,
            dt=self.dt,
            max_iter=self.max_iter,
            tol=self.tol,
            print_level=self.print_level,
        )

    def _solve_nmpc(
        self,
        x0: np.ndarray,
        ref: np.ndarray,
        theta: np.ndarray,
        theta_cov: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the NMPC optimization problem."""
        # Solve with frozen theta
        assert self._solver is not None
        u_opt, x_opt, success = self._solver.solve(
            x0=x0,
            x_ref=ref,
            theta=theta,
            w0=self._w_prev,
        )

        if not success:
            # Fallback to previous solution or zero
            if self._initialized:
                u_opt = self._u_traj.copy()
                x_opt = self._x_traj.copy()
            else:
                u_opt = np.zeros((self.horizon, self.nu))
                x_opt = np.tile(x0, (self.horizon + 1, 1))

        # Compute FIM for dual-control
        r_diag = np.diag(self.R_u)
        R_inv = np.diag(1.0 / r_diag) if np.all(r_diag > 0) else np.eye(self.nu)
        fim = compute_fim_prediction(
            x_traj=x_opt,
            u_traj=u_opt,
            theta=theta,
            A_func=self._dynamics.A,
            C_func=self._dynamics.C,
            R_inv=R_inv,
        )

        # Apply probing if lambda_info > 0
        if self.lambda_info > 0:
            u_opt = self._apply_probing(u_opt, fim, theta_cov)

        # Store for warm-start
        self._w_prev = self._build_warm_start(u_opt, x_opt)

        return u_opt, x_opt, fim

    def _apply_probing(
        self,
        u_opt: np.ndarray,
        fim: np.ndarray,
        theta_cov: np.ndarray,
    ) -> np.ndarray:
        """Apply probing modification for dual-control.

        Adds small excitation to improve parameter identifiability.
        """
        # Simple probing: add sinusoidal excitation when uncertain
        uncertainty = np.trace(theta_cov)

        if uncertainty > 0.5:  # Only probe when uncertain
            u_modified = u_opt.copy()
            for k in range(len(u_opt)):
                # Probing signal
                probe = 0.1 * np.sin(2 * np.pi * k / max(self.horizon, 1))
                u_probe = u_opt[k] + self.lambda_info * probe * uncertainty
                # Respect constraints
                u_modified[k] = np.clip(u_probe, self.u_min, self.u_max)
            result: np.ndarray = u_modified
            return result

        return u_opt

    def _build_warm_start(
        self,
        u_opt: np.ndarray,
        x_opt: np.ndarray,
    ) -> np.ndarray:
        """Build warm-start vector for next iteration."""
        N = self.horizon

        w = []
        w.extend(x_opt[0].tolist())

        for k in range(N):
            w.extend(u_opt[k].tolist())
            w.extend(x_opt[k + 1].tolist())

        return np.array(w)
