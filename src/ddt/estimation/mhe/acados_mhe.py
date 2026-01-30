"""acados-based MHE implementation (primary, high-performance)."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

try:
    import casadi as ca
    from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

    ACADOS_AVAILABLE = True
except ImportError:
    ACADOS_AVAILABLE = False
    ca = None  # type: ignore[assignment]

from .base import MHEBase
from .covariance import compute_fim_from_trajectory, estimate_covariance_from_fim
from .model import build_linear_scalar_model

if TYPE_CHECKING:
    pass


@dataclass
class AcadosMHE(MHEBase):
    """Moving Horizon Estimator using acados + HPIPM.

    This is the primary, high-performance implementation.
    Uses acados for efficient QP solving with HPIPM.
    """

    _acados_solver: AcadosOcpSolver | None = field(default=None, init=False)
    _acados_model: AcadosModel | None = field(default=None, init=False)
    _code_export_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if not ACADOS_AVAILABLE:
            raise ImportError(
                "acados required for AcadosMHE. "
                "Install with: pip install acados_template casadi"
            )

        super().__post_init__()
        self._code_export_dir = Path(tempfile.mkdtemp(prefix="acados_mhe_"))
        self._build_acados_ocp()

    def _build_acados_ocp(self) -> None:
        """Build the acados OCP for MHE."""
        N = self.cfg.horizon
        nx = self.nx
        nu = self.nu

        # Build CasADi model for acados
        model = self._create_acados_model()
        self._acados_model = model

        # Create OCP
        ocp = AcadosOcp()
        ocp.model = model

        # Dimensions
        ocp.dims.N = N

        # Cost (Gauss-Newton formulation)
        # We use EXTERNAL cost type for flexibility

        # For MHE, we reformulate as a tracking problem:
        # min sum ||y_ref - h(x)||^2_R + ||w||^2_Q + arrival_cost

        # Cost weights
        np.diag(self.cfg.noise.Q_diag)  # Process noise weight
        np.diag(self.cfg.noise.R_diag)  # Measurement weight

        # Extended state for MHE includes parameters
        # We'll use a simpler formulation: track measurements

        # Cost type: LINEAR_LS
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # Cost matrices
        ny_cost = nx  # Tracking state (which equals measurement for our model)
        ocp.cost.W = np.diag(1.0 / np.array(self.cfg.noise.R_diag))
        ocp.cost.W_e = np.diag(1.0 / np.array(self.cfg.noise.R_diag))

        # Output selection matrices
        ocp.cost.Vx = np.eye(ny_cost, nx)
        ocp.cost.Vu = np.zeros((ny_cost, nu))
        ocp.cost.Vx_e = np.eye(ny_cost, nx)

        # Reference (to be updated at runtime)
        ocp.cost.yref = np.zeros(ny_cost)
        ocp.cost.yref_e = np.zeros(ny_cost)

        # Constraints
        ocp.constraints.x0 = np.zeros(nx)  # Will be updated

        # Solver options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"
        ocp.solver_options.nlp_solver_type = "SQP"
        ocp.solver_options.nlp_solver_max_iter = self.cfg.solver.max_iter
        ocp.solver_options.tol = self.cfg.solver.tol
        ocp.solver_options.tf = N * self.dt

        # Code generation
        ocp.code_export_directory = str(self._code_export_dir)

        # Create solver
        json_file = str(self._code_export_dir / "acados_ocp.json")
        self._acados_solver = AcadosOcpSolver(ocp, json_file=json_file)

    def _create_acados_model(self) -> AcadosModel:
        """Create acados model from CasADi dynamics."""
        model = AcadosModel()

        # Symbolic variables
        x = ca.SX.sym("x", self.nx)
        u = ca.SX.sym("u", self.nu)

        # For MHE, parameters are part of the model
        # We use a fixed theta (updated externally)
        theta = ca.SX.sym("theta", self.ntheta)

        a = theta[0]
        b = theta[1]

        # Discrete dynamics
        x_next = a * x + b * u

        model.name = "mhe_model"
        model.x = x
        model.u = u
        model.p = theta
        model.disc_dyn_expr = x_next

        return model

    def _solve_mhe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the MHE optimization problem using acados."""
        N = self.cfg.horizon
        nx = self.nx

        n_meas = len(self._y_buffer)
        n_inputs = len(self._u_buffer) - 1

        if n_inputs < 1:
            return (
                np.zeros((N + 1, nx)),
                self._theta_hat.copy(),
                self._theta_cov.copy(),
            )

        # Get data
        y_data = np.array(self._y_buffer[-min(n_meas, N + 1) :]).reshape(-1, 1)
        u_data = np.array(self._u_buffer[-min(n_inputs + 1, N + 1) : -1]).reshape(-1, 1)

        # Pad if needed
        while len(y_data) < N + 1:
            y_data = np.vstack([y_data[0:1], y_data])
        while len(u_data) < N:
            u_data = np.vstack([u_data[0:1], u_data])

        # Update solver with current data
        assert self._acados_solver is not None
        solver = self._acados_solver

        # Set initial state constraint
        solver.set(0, "lbx", y_data[0])
        solver.set(0, "ubx", y_data[0])

        # Set references (measurements)
        for i in range(N):
            solver.set(i, "yref", y_data[i].flatten())
            solver.set(i, "u", u_data[i].flatten())
            solver.set(i, "p", self._theta_hat)

        solver.set(N, "yref", y_data[N].flatten())
        solver.set(N, "p", self._theta_hat)

        # Solve
        status = solver.solve()

        if status != 0:
            # Solver failed, return previous estimate
            return (
                self._x_traj.copy(),
                self._theta_hat.copy(),
                self._theta_cov.copy(),
            )

        # Extract solution
        x_opt = np.zeros((N + 1, nx))
        for i in range(N + 1):
            x_opt[i] = solver.get(i, "x")

        # For parameter estimation, we use a separate optimization
        # Here we do a simple least-squares fit on the trajectory
        theta_opt = self._estimate_parameters_ls(x_opt, u_data)

        # Compute covariance
        casadi_model = build_linear_scalar_model(self.dt)
        R_inv = np.diag(1.0 / np.array(self.cfg.noise.R_diag))
        fim = compute_fim_from_trajectory(
            x_traj=x_opt,
            u_traj=u_data,
            theta=theta_opt,
            C_func=casadi_model.C_func,
            R_inv=R_inv,
        )
        cov = estimate_covariance_from_fim(fim, self._theta_cov)

        return x_opt, theta_opt, cov

    def _estimate_parameters_ls(
        self,
        x_traj: np.ndarray,
        u_traj: np.ndarray,
    ) -> np.ndarray:
        """Estimate parameters from trajectory using least squares.

        For x_{k+1} = a*x_k + b*u_k, we solve:
        [x_1]     [x_0  u_0] [a]
        [x_2]  =  [x_1  u_1] [b]
        [...]     [... ...]
        """
        N = len(u_traj)
        if N < 2:
            result: np.ndarray = self._theta_hat.copy()
            return result

        # Build regression
        x_k = x_traj[:-1].flatten()
        x_kp1 = x_traj[1:].flatten()
        u_k = u_traj.flatten()[:N]

        Phi = np.column_stack([x_k, u_k])
        y = x_kp1

        # Regularized least squares
        reg = 1e-6 * np.eye(2)
        try:
            theta: np.ndarray = np.linalg.solve(Phi.T @ Phi + reg, Phi.T @ y)
            return theta
        except np.linalg.LinAlgError:
            fallback: np.ndarray = self._theta_hat.copy()
            return fallback

    def __del__(self) -> None:
        """Cleanup generated code."""
        import shutil

        if hasattr(self, "_code_export_dir") and self._code_export_dir.exists():
            try:
                shutil.rmtree(self._code_export_dir)
            except Exception:
                pass
