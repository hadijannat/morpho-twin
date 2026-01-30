"""CasADi + IPOPT MHE implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

try:
    import casadi as ca
except ImportError:
    ca = None  # type: ignore[assignment]

from .base import MHEBase
from .covariance import compute_fim_from_trajectory, estimate_covariance_from_fim
from .model import SymbolicModel, build_linear_scalar_model

if TYPE_CHECKING:
    pass


@dataclass
class CasADiMHE(MHEBase):
    """Moving Horizon Estimator using CasADi + IPOPT.

    This is the fallback implementation when acados is not available.
    Uses direct collocation with IPOPT as the NLP solver.
    """

    _model: SymbolicModel = field(init=False)
    _nlp_solver: ca.Function | None = field(default=None, init=False)
    _w_opt_prev: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if ca is None:
            raise ImportError(
                "CasADi required for CasADiMHE. "
                "Install with: pip install casadi"
            )

        super().__post_init__()
        self._model = build_linear_scalar_model(self.dt)
        self._build_nlp()

    def _build_nlp(self) -> None:
        """Build the NLP for MHE optimization."""
        N = self.cfg.horizon
        nx = self.nx
        nu = self.nu
        ny = self.ny
        ntheta = self.ntheta

        # Decision variables: [x_0, ..., x_N, w_0, ..., w_{N-1}, theta]
        # Total: (N+1)*nx + N*nx + ntheta

        # Create symbolic decision variable
        w_sym = []
        w_idx = {}  # Track indices

        # States
        X = []
        idx = 0
        for i in range(N + 1):
            x_i = ca.SX.sym(f"x_{i}", nx)
            X.append(x_i)
            w_sym.append(x_i)
            w_idx[f"x_{i}"] = (idx, idx + nx)
            idx += nx

        # Process noise
        W = []
        for i in range(N):
            w_i = ca.SX.sym(f"w_{i}", nx)
            W.append(w_i)
            w_sym.append(w_i)
            w_idx[f"w_{i}"] = (idx, idx + nx)
            idx += nx

        # Parameters
        theta = ca.SX.sym("theta", ntheta)
        w_sym.append(theta)
        w_idx["theta"] = (idx, idx + ntheta)
        self._theta_indices = (idx, idx + ntheta)
        idx += ntheta

        # Flatten decision variable
        w = ca.vertcat(*w_sym)
        n_w = w.shape[0]

        # Parameters to NLP (measurements, inputs, priors)
        # y_meas: (N+1)*ny, u_applied: N*nu, x_prior: nx, theta_prior: ntheta, P0_diag: nx
        n_p = (N + 1) * ny + N * nu + nx + ntheta + nx + ntheta
        p = ca.SX.sym("p", n_p)

        # Unpack parameters
        p_idx = 0
        y_meas = []
        for _ in range(N + 1):
            y_meas.append(p[p_idx : p_idx + ny])
            p_idx += ny

        u_applied = []
        for _ in range(N):
            u_applied.append(p[p_idx : p_idx + nu])
            p_idx += nu

        x_prior = p[p_idx : p_idx + nx]
        p_idx += nx
        theta_prior = p[p_idx : p_idx + ntheta]
        p_idx += ntheta
        P0_diag = p[p_idx : p_idx + nx]
        p_idx += nx
        P_theta_diag = p[p_idx : p_idx + ntheta]

        # Build objective
        J = 0.0

        # Arrival cost: ||x_0 - x_prior||^2_{P0^{-1}}
        P0_inv_diag = 1.0 / (P0_diag + 1e-8)
        arrival_x = X[0] - x_prior
        J += self.cfg.arrival_cost_scaling * ca.dot(arrival_x * P0_inv_diag, arrival_x)

        # Parameter regularization: ||theta - theta_prior||^2
        P_theta_inv_diag = 1.0 / (P_theta_diag + 1e-8)
        theta_err = theta - theta_prior
        J += ca.dot(theta_err * P_theta_inv_diag, theta_err)

        # Process noise cost: sum ||w_i||^2_{Q^{-1}}
        Q_inv = np.diag(1.0 / np.array(self.cfg.noise.Q_diag))
        for i in range(N):
            J += ca.mtimes([W[i].T, Q_inv, W[i]])

        # Measurement cost: sum ||y_i - h(x_i)||^2_{R^{-1}}
        R_inv = np.diag(1.0 / np.array(self.cfg.noise.R_diag))
        for i in range(N + 1):
            y_pred = self._model.h_output(X[i])
            v_i = y_meas[i] - y_pred
            J += ca.mtimes([v_i.T, R_inv, v_i])

        # Build constraints
        g = []
        lbg = []
        ubg = []

        # Dynamics constraints: x_{i+1} = f(x_i, u_i, theta) + w_i
        for i in range(N):
            x_next = self._model.f_discrete(X[i], u_applied[i], theta)
            constraint = X[i + 1] - x_next - W[i]
            g.append(constraint)
            lbg.extend([0.0] * nx)
            ubg.extend([0.0] * nx)

        g = ca.vertcat(*g) if g else ca.SX(0, 1)

        # Variable bounds
        lbw = [-np.inf] * n_w
        ubw = [np.inf] * n_w

        # Build NLP
        nlp = {"x": w, "f": J, "g": g, "p": p}

        opts = {
            "ipopt.print_level": self.cfg.solver.print_level,
            "ipopt.max_iter": self.cfg.solver.max_iter,
            "ipopt.tol": self.cfg.solver.tol,
            "print_time": False,
        }

        self._nlp_solver = ca.nlpsol("mhe", "ipopt", nlp, opts)
        self._n_w = n_w
        self._n_p = n_p
        self._lbw = lbw
        self._ubw = ubw
        self._lbg = lbg
        self._ubg = ubg
        self._w_idx = w_idx

    def _solve_mhe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the MHE optimization problem."""
        N = self.cfg.horizon
        nx = self.nx
        nu = self.nu
        ny = self.ny
        ntheta = self.ntheta

        # Get actual horizon (may be less than N if buffer not full)
        n_meas = len(self._y_buffer)
        n_inputs = len(self._u_buffer) - 1  # One less input than measurements

        if n_inputs < 1:
            # Not enough data, return prior
            return (
                np.zeros((N + 1, nx)),
                self._theta_hat.copy(),
                self._theta_cov.copy(),
            )

        # Pad buffers to full horizon if needed
        y_data = np.array(self._y_buffer[-min(n_meas, N + 1) :]).reshape(-1, ny)
        u_data = np.array(self._u_buffer[-min(n_inputs + 1, N + 1) : -1]).reshape(-1, nu)

        min(len(u_data), N)

        # Pad if needed
        while len(y_data) < N + 1:
            y_data = np.vstack([y_data[0:1], y_data])
        while len(u_data) < N:
            u_data = np.vstack([u_data[0:1], u_data])

        # Build parameter vector
        p = []
        for i in range(N + 1):
            p.extend(y_data[i].tolist())
        for i in range(N):
            p.extend(u_data[i].tolist())

        # Priors - use EKF arrival cost if available
        if self._ekf_updater is not None and self._ekf_updater.is_initialized:
            x_prior, P0_diag = self._ekf_updater.get_arrival_cost_prior()
        else:
            x_prior = y_data[0]  # Use first measurement as prior
            P0_diag = np.ones(nx) * 10.0  # Loose prior on initial state
        theta_prior = self._theta_hat
        P_theta_diag = np.diag(self._theta_cov)

        p.extend(x_prior.tolist())
        p.extend(theta_prior.tolist())
        p.extend(P0_diag.tolist())
        p.extend(P_theta_diag.tolist())
        p_arr = np.array(p)

        # Initial guess (warm start)
        if self._w_opt_prev is not None:
            w0 = self._w_opt_prev.copy()
            # Shift for new horizon
            # (simplified: just use previous solution)
        else:
            # Cold start: initialize states from measurements
            w0 = np.zeros(self._n_w)
            idx = 0
            for i in range(N + 1):
                w0[idx : idx + nx] = y_data[i]
                idx += nx
            # Process noise = 0
            idx += N * nx
            # Parameters from prior
            w0[idx : idx + ntheta] = theta_prior

        # Solve
        assert self._nlp_solver is not None
        sol = self._nlp_solver(
            x0=w0,
            lbx=self._lbw,
            ubx=self._ubw,
            lbg=self._lbg,
            ubg=self._ubg,
            p=p_arr,
        )

        w_opt = np.array(sol["x"]).flatten()
        self._w_opt_prev = w_opt

        # Extract solution
        idx = 0
        x_opt = np.zeros((N + 1, nx))
        for i in range(N + 1):
            x_opt[i] = w_opt[idx : idx + nx]
            idx += nx

        # Skip process noise
        idx += N * nx

        # Parameters
        theta_opt = w_opt[idx : idx + ntheta]

        # Compute covariance from FIM
        R_inv = np.diag(1.0 / np.array(self.cfg.noise.R_diag))
        fim = compute_fim_from_trajectory(
            x_traj=x_opt,
            u_traj=u_data,
            theta=theta_opt,
            C_func=self._model.C_func,
            R_inv=R_inv,
        )
        cov = estimate_covariance_from_fim(fim, self._theta_cov)

        # Update EKF arrival cost updater with MHE solution
        if self._ekf_updater is not None:
            # Sync EKF with MHE estimate at window start
            theta_cov_diag = np.diag(self._theta_cov)
            if nx <= len(theta_cov_diag):
                P_mhe_start = theta_cov_diag[:nx]
            else:
                P_mhe_start = np.ones(nx) * 1.0
            self._ekf_updater.update_from_mhe_solution(
                x_mhe_start=x_opt[0],
                P_mhe=P_mhe_start,
                theta=theta_opt,
            )

        return x_opt, theta_opt, cov
