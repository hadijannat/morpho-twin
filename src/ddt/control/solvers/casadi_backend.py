"""CasADi + IPOPT solver backend for NMPC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ..ocp.constraints import BoxConstraints
    from ..ocp.cost import QuadraticCost
    from ..ocp.dynamics import DiscreteDynamics


@dataclass
class CasADiSolver:
    """CasADi + IPOPT NLP solver for NMPC."""

    dynamics: DiscreteDynamics
    cost: QuadraticCost
    constraints: BoxConstraints
    horizon: int
    dt: float
    max_iter: int = 50
    tol: float = 1e-6
    print_level: int = 0

    _nlp_solver: ca.Function | None = field(default=None, init=False)
    _n_w: int = field(default=0, init=False)
    _n_p: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if ca is None:
            raise ImportError("CasADi required for CasADiSolver.")
        self._build_nlp()

    def _build_nlp(self) -> None:
        """Build the NLP for NMPC."""
        N = self.horizon
        nx = self.dynamics.nx
        nu = self.dynamics.nu
        ntheta = self.dynamics.ntheta

        # Decision variables: [x_0, u_0, x_1, u_1, ..., x_{N-1}, u_{N-1}, x_N]
        w = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # Parameters: [x0, x_ref, theta]
        x0_param = ca.SX.sym("x0", nx)
        x_ref_param = ca.SX.sym("x_ref", nx)
        theta_param = ca.SX.sym("theta", ntheta)
        p = ca.vertcat(x0_param, x_ref_param, theta_param)

        # Initial state
        X_prev = ca.SX.sym("X_0", nx)
        w.append(X_prev)
        lbw.extend([-np.inf] * nx)  # Will be set to x0 at runtime via equality
        ubw.extend([np.inf] * nx)

        # Initial state constraint
        g.append(X_prev - x0_param)
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)

        for k in range(N):
            # Input at stage k
            U_k = ca.SX.sym(f"U_{k}", nu)
            w.append(U_k)
            lbw.extend(self.constraints.u_min.tolist())
            ubw.extend(self.constraints.u_max.tolist())

            # Stage cost
            dx = X_prev - x_ref_param
            J += ca.mtimes([dx.T, self.cost.Q, dx])
            J += ca.mtimes([U_k.T, self.cost.R, U_k])

            # Dynamics constraint
            X_next_pred = self.dynamics.f(X_prev, U_k, theta_param)

            # State at stage k+1
            X_next = ca.SX.sym(f"X_{k+1}", nx)
            w.append(X_next)
            lbw.extend(self.constraints.x_min.tolist())
            ubw.extend(self.constraints.x_max.tolist())

            # Dynamics equality constraint
            g.append(X_next - X_next_pred)
            lbg.extend([0.0] * nx)
            ubg.extend([0.0] * nx)

            X_prev = X_next

        # Terminal cost
        dx_N = X_prev - x_ref_param
        J += ca.mtimes([dx_N.T, self.cost.Q_terminal, dx_N])

        # Build NLP
        w = ca.vertcat(*w)
        g = ca.vertcat(*g)

        nlp = {"x": w, "f": J, "g": g, "p": p}

        opts = {
            "ipopt.print_level": self.print_level,
            "ipopt.max_iter": self.max_iter,
            "ipopt.tol": self.tol,
            "print_time": False,
        }

        self._nlp_solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)
        self._n_w = w.shape[0]  # type: ignore[attr-defined]
        self._n_p = p.shape[0]
        self._lbw = lbw
        self._ubw = ubw
        self._lbg = lbg
        self._ubg = ubg

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        theta: np.ndarray,
        w0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Solve NMPC problem.

        Args:
            x0: Initial state (nx,)
            x_ref: Reference state (nx,)
            theta: Parameter estimate (ntheta,)
            w0: Initial guess (optional)

        Returns:
            u_opt: Optimal input trajectory (N, nu)
            x_opt: Optimal state trajectory (N+1, nx)
            success: True if solver converged
        """
        # Parameter vector
        p = np.concatenate([x0, x_ref, theta])

        # Initial guess
        if w0 is None:
            w0 = self._build_initial_guess(x0)

        # Solve
        assert self._nlp_solver is not None
        sol = self._nlp_solver(
            x0=w0,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg,
            ubg=self._ubg,
            p=p,
        )

        w_opt = np.array(sol["x"]).flatten()

        # Check convergence
        stats = self._nlp_solver.stats()
        success = stats["return_status"] == "Solve_Succeeded"

        # Extract solution
        u_opt, x_opt = self._extract_solution(w_opt)

        return u_opt, x_opt, success

    def _build_initial_guess(self, x0: np.ndarray) -> np.ndarray:
        """Build initial guess for warm starting."""
        N = self.horizon
        nu = self.dynamics.nu

        w0 = []
        # Initial state
        w0.extend(x0.tolist())

        # Interleaved u, x
        for _k in range(N):
            w0.extend([0.0] * nu)  # Zero input
            w0.extend(x0.tolist())  # Assume state stays at x0

        return np.array(w0)

    def _extract_solution(
        self,
        w_opt: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract state and input trajectories from solution."""
        N = self.horizon
        nx = self.dynamics.nx
        nu = self.dynamics.nu

        x_opt = np.zeros((N + 1, nx))
        u_opt = np.zeros((N, nu))

        idx = 0
        # Initial state
        x_opt[0] = w_opt[idx : idx + nx]
        idx += nx

        for k in range(N):
            u_opt[k] = w_opt[idx : idx + nu]
            idx += nu
            x_opt[k + 1] = w_opt[idx : idx + nx]
            idx += nx

        return u_opt, x_opt
