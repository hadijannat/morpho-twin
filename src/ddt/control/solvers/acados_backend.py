"""acados + HPIPM solver backend for RTI-NMPC."""

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

if TYPE_CHECKING:
    from ..ocp.constraints import BoxConstraints
    from ..ocp.cost import QuadraticCost
    from ..ocp.dynamics import DiscreteDynamics


@dataclass
class AcadosSolver:
    """acados + HPIPM solver for RTI-NMPC.

    Implements Real-Time Iteration (RTI) scheme:
    1. Prepare phase: Linearize, build QP, factorize KKT
    2. Feedback phase: Update x0, solve single SQP iteration
    """

    dynamics: DiscreteDynamics
    cost: QuadraticCost
    constraints: BoxConstraints
    horizon: int
    dt: float
    rti_mode: bool = True  # True for RTI, False for full SQP
    max_sqp_iter: int = 1  # 1 for RTI
    tol: float = 1e-6

    _acados_solver: AcadosOcpSolver | None = field(default=None, init=False)
    _code_export_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        if not ACADOS_AVAILABLE:
            raise ImportError(
                "acados required for AcadosSolver. "
                "Install with: pip install acados_template casadi"
            )
        self._code_export_dir = Path(tempfile.mkdtemp(prefix="acados_nmpc_"))
        self._build_ocp()

    def _build_ocp(self) -> None:
        """Build the acados OCP for NMPC."""
        N = self.horizon
        nx = self.dynamics.nx
        nu = self.dynamics.nu

        # Create acados model
        model = self._create_acados_model()

        # Create OCP
        ocp = AcadosOcp()
        ocp.model = model

        # Dimensions
        ocp.dims.N = N

        # Cost (LINEAR_LS for tracking)
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.cost_type_e = "LINEAR_LS"

        # Output dimension for cost
        ny = nx + nu
        ny_e = nx

        # Weight matrices
        W = np.zeros((ny, ny))
        W[:nx, :nx] = self.cost.Q
        W[nx:, nx:] = self.cost.R
        ocp.cost.W = W
        ocp.cost.W_e = self.cost.Q_terminal

        # Selection matrices
        Vx = np.zeros((ny, nx))
        Vx[:nx, :nx] = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[nx:, :nu] = np.eye(nu)
        ocp.cost.Vx = Vx
        ocp.cost.Vu = Vu
        ocp.cost.Vx_e = np.eye(ny_e, nx)

        # Reference (to be updated at runtime)
        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(ny_e)

        # Constraints
        ocp.constraints.x0 = np.zeros(nx)

        # State bounds
        ocp.constraints.lbx = self.constraints.x_min
        ocp.constraints.ubx = self.constraints.x_max
        ocp.constraints.idxbx = np.arange(nx)

        # Input bounds
        ocp.constraints.lbu = self.constraints.u_min
        ocp.constraints.ubu = self.constraints.u_max
        ocp.constraints.idxbu = np.arange(nu)

        # Solver options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "DISCRETE"

        if self.rti_mode:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
        else:
            ocp.solver_options.nlp_solver_type = "SQP"

        ocp.solver_options.nlp_solver_max_iter = self.max_sqp_iter
        ocp.solver_options.tol = self.tol
        ocp.solver_options.tf = N * self.dt

        # Code generation
        ocp.code_export_directory = str(self._code_export_dir)

        # Create solver
        self._acados_solver = AcadosOcpSolver(
            ocp,
            json_file=str(self._code_export_dir / "acados_ocp.json"),
        )

    def _create_acados_model(self) -> AcadosModel:
        """Create acados model from dynamics."""
        model = AcadosModel()

        nx = self.dynamics.nx
        nu = self.dynamics.nu
        ntheta = self.dynamics.ntheta

        # Symbolic variables
        x = ca.SX.sym("x", nx)
        u = ca.SX.sym("u", nu)
        theta = ca.SX.sym("theta", ntheta)

        # Get dynamics expression
        x_next = self.dynamics.f(x, u, theta)

        model.name = "nmpc_model"
        model.x = x
        model.u = u
        model.p = theta
        model.disc_dyn_expr = x_next

        return model

    def prepare(self, theta: np.ndarray) -> None:
        """RTI prepare phase: linearize and build QP.

        This can be done in a background thread before
        the measurement arrives.

        Args:
            theta: Parameter estimate for linearization
        """
        assert self._acados_solver is not None
        solver = self._acados_solver
        N = self.horizon

        # Update parameter at all stages
        for i in range(N + 1):
            solver.set(i, "p", theta)

        # Prepare the QP (linearization, factorization)
        if self.rti_mode:
            solver.options_set("rti_phase", 1)  # Prepare phase
            solver.solve()

    def feedback(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """RTI feedback phase: update x0 and solve QP.

        This is the time-critical part that runs in real-time.

        Args:
            x0: Current state measurement
            x_ref: Reference state

        Returns:
            u_opt: Optimal input trajectory (N, nu)
            x_opt: Optimal state trajectory (N+1, nx)
            success: True if solver converged
        """
        assert self._acados_solver is not None
        solver = self._acados_solver
        N = self.horizon
        nx = self.dynamics.nx
        nu = self.dynamics.nu

        # Set initial state
        solver.set(0, "lbx", x0)
        solver.set(0, "ubx", x0)

        # Set reference at all stages
        yref = np.concatenate([x_ref, np.zeros(nu)])
        for i in range(N):
            solver.set(i, "yref", yref)
        solver.set(N, "yref", x_ref)

        # Feedback phase
        if self.rti_mode:
            solver.options_set("rti_phase", 2)  # Feedback phase

        status = solver.solve()

        success = status == 0

        # Extract solution
        x_opt = np.zeros((N + 1, nx))
        u_opt = np.zeros((N, nu))

        for i in range(N):
            x_opt[i] = solver.get(i, "x")
            u_opt[i] = solver.get(i, "u")
        x_opt[N] = solver.get(N, "x")

        return u_opt, x_opt, success

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        theta: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Combined prepare + feedback (for non-RTI mode or convenience).

        Args:
            x0: Initial state
            x_ref: Reference state
            theta: Parameter estimate

        Returns:
            u_opt: Optimal input trajectory
            x_opt: Optimal state trajectory
            success: Convergence flag
        """
        self.prepare(theta)
        return self.feedback(x0, x_ref)

    def __del__(self) -> None:
        """Cleanup generated code."""
        import shutil

        if hasattr(self, "_code_export_dir") and self._code_export_dir.exists():
            try:
                shutil.rmtree(self._code_export_dir)
            except Exception:
                pass
