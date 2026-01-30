"""Factory functions for creating DDT components."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .baselines.pid import PIDController
from .config import AppConfig
from .estimation.windowed_ls import WindowedLeastSquaresEstimator
from .interfaces import Controller, Estimator, Plant, SafetyFilter
from .safety.state_box import StateBoxSafetyFilter
from .sim.linear_scalar import LinearScalarPlant

if TYPE_CHECKING:
    from .supervision import Supervisor


def make_plant(cfg: AppConfig) -> Plant:
    """Create plant from configuration."""
    if cfg.plant.type == "linear_scalar":
        assert cfg.plant.a_true is not None and cfg.plant.b_true is not None
        plant = LinearScalarPlant(
            dt=cfg.dt,
            a_true=cfg.plant.a_true,
            b_true=cfg.plant.b_true,
            process_noise_std=cfg.plant.process_noise_std,
            meas_noise_std=cfg.plant.meas_noise_std,
            x0=cfg.plant.x0,
        )
        plant.seed(cfg.seed)
        return plant
    raise ValueError(f"Unknown plant type: {cfg.plant.type}")


def make_estimator(cfg: AppConfig) -> Estimator:
    """Create estimator from configuration."""
    if cfg.estimation.type == "windowed_ls":
        return WindowedLeastSquaresEstimator(window=cfg.estimation.window)

    elif cfg.estimation.type == "mhe":
        # Try acados first, fall back to CasADi
        from .estimation.mhe.config import MHEConfig as MHEConfigInternal

        mhe_cfg_raw = cfg.estimation.mhe
        if mhe_cfg_raw is None:
            mhe_cfg = MHEConfigInternal()
        else:
            # Convert config to internal MHE config type
            mhe_cfg = MHEConfigInternal.model_validate(mhe_cfg_raw.model_dump())

        if mhe_cfg.solver.backend == "acados":
            try:
                from .estimation.mhe import AcadosMHE

                return AcadosMHE(
                    cfg=mhe_cfg,
                    dt=cfg.dt,
                    nx=1,
                    nu=1,
                    ny=1,
                    ntheta=2,
                )
            except ImportError:
                pass  # Fall through to CasADi

        # CasADi fallback
        try:
            from .estimation.mhe import CasADiMHE

            return CasADiMHE(
                cfg=mhe_cfg,
                dt=cfg.dt,
                nx=1,
                nu=1,
                ny=1,
                ntheta=2,
            )
        except ImportError as e:
            raise ImportError(
                "MHE requires CasADi. Install with: pip install casadi"
            ) from e

    raise ValueError(f"Unknown estimator type: {cfg.estimation.type}")


def make_controller(cfg: AppConfig) -> Controller:
    """Create controller from configuration."""
    if cfg.control.type == "pid":
        return PIDController(
            kp=cfg.control.kp,
            ki=cfg.control.ki,
            kd=cfg.control.kd,
            dt=cfg.dt,
            u_min=cfg.control.u_min,
            u_max=cfg.control.u_max,
        )

    elif cfg.control.type == "nmpc_rti":
        nmpc_cfg = cfg.control.nmpc
        if nmpc_cfg is None:
            from .config import NMPCConfig

            nmpc_cfg = NMPCConfig()

        try:
            from .control import RTINMPC

            return RTINMPC(
                dt=cfg.dt,
                horizon=nmpc_cfg.horizon,
                nx=1,
                nu=1,
                ntheta=2,
                Q=np.diag(nmpc_cfg.Q),
                R_u=np.diag(nmpc_cfg.R_u),
                lambda_info=nmpc_cfg.lambda_info,
                u_min=np.array([cfg.control.u_min]),
                u_max=np.array([cfg.control.u_max]),
                x_min=np.array([cfg.safety.x_min]),
                x_max=np.array([cfg.safety.x_max]),
                rti_mode=nmpc_cfg.rti_mode,
                max_sqp_iter=nmpc_cfg.max_iter if not nmpc_cfg.rti_mode else 1,
            )
        except ImportError as e:
            raise ImportError(
                "RTI-NMPC requires acados. Install with: pip install acados_template casadi"
            ) from e

    elif cfg.control.type == "nmpc_casadi":
        nmpc_cfg = cfg.control.nmpc
        if nmpc_cfg is None:
            from .config import NMPCConfig

            nmpc_cfg = NMPCConfig()

        try:
            from .control import CasADiNMPC

            return CasADiNMPC(
                dt=cfg.dt,
                horizon=nmpc_cfg.horizon,
                nx=1,
                nu=1,
                ntheta=2,
                Q=np.diag(nmpc_cfg.Q),
                R_u=np.diag(nmpc_cfg.R_u),
                lambda_info=nmpc_cfg.lambda_info,
                u_min=np.array([cfg.control.u_min]),
                u_max=np.array([cfg.control.u_max]),
                x_min=np.array([cfg.safety.x_min]),
                x_max=np.array([cfg.safety.x_max]),
                max_iter=nmpc_cfg.max_iter,
            )
        except ImportError as e:
            raise ImportError(
                "CasADi NMPC requires CasADi. Install with: pip install casadi"
            ) from e

    raise ValueError(f"Unknown controller type: {cfg.control.type}")


def make_safety(cfg: AppConfig) -> SafetyFilter:
    """Create safety filter from configuration."""
    if cfg.safety.type == "state_box":
        return StateBoxSafetyFilter(
            x_min=cfg.safety.x_min,
            x_max=cfg.safety.x_max,
            alpha=cfg.safety.alpha,
            slack_weight=cfg.safety.slack_weight,
        )

    elif cfg.safety.type == "cbf_qp":
        cbf_cfg = cfg.safety.cbf
        if cbf_cfg is None:
            from .config import CBFConfig

            cbf_cfg = CBFConfig()

        try:
            from .safety import CBFQPSafetyFilter

            return CBFQPSafetyFilter(
                nx=1,
                nu=1,
                ntheta=2,
                dt=cfg.dt,
                x_min=np.array([cfg.safety.x_min]),
                x_max=np.array([cfg.safety.x_max]),
                u_min=np.array([cfg.control.u_min]),
                u_max=np.array([cfg.control.u_max]),
                alpha=cbf_cfg.alpha,
                slack_weight=cbf_cfg.slack_weight,
                gamma_robust=cbf_cfg.gamma_robust,
            )
        except ImportError as e:
            raise ImportError(
                "CBF-QP requires OSQP. Install with: pip install osqp"
            ) from e

    raise ValueError(f"Unknown safety type: {cfg.safety.type}")


def make_supervisor(cfg: AppConfig) -> Supervisor | None:
    """Create supervisor from configuration.

    Returns None if supervision is disabled.
    """
    if not cfg.supervision.enabled:
        return None

    from .supervision import Supervisor
    from .supervision.mode_manager import ModeConfig as ModeManagerConfig

    pe_cfg = cfg.supervision.pe
    mode_cfg = cfg.supervision.mode

    mode_manager_config = ModeManagerConfig(
        uncertainty_normal_to_conservative=mode_cfg.uncertainty_normal_to_conservative,
        uncertainty_conservative_to_safe=mode_cfg.uncertainty_conservative_to_safe,
        uncertainty_safe_to_conservative=mode_cfg.uncertainty_safe_to_conservative,
        uncertainty_conservative_to_normal=mode_cfg.uncertainty_conservative_to_normal,
        pe_violation_count_conservative=mode_cfg.pe_violation_count_conservative,
        pe_violation_count_safe=mode_cfg.pe_violation_count_safe,
        pe_satisfaction_count_recovery=mode_cfg.pe_satisfaction_count_recovery,
        margin_factor_normal=mode_cfg.margin_factor_normal,
        margin_factor_conservative=mode_cfg.margin_factor_conservative,
        margin_factor_safe=mode_cfg.margin_factor_safe,
    )

    return Supervisor(
        pe_window=pe_cfg.window,
        pe_lambda_threshold=pe_cfg.lambda_threshold,
        ntheta=2,
        mode_config=mode_manager_config,
    )


def make_barriers(cfg: AppConfig) -> list:
    """Create barrier functions from configuration.

    Returns list of BarrierFunction objects.
    """
    from .safety.barriers import create_box_barriers

    # Default: box barriers for state constraints
    return [
        create_box_barriers(
            x_min=np.array([cfg.safety.x_min]),
            x_max=np.array([cfg.safety.x_max]),
        )
    ]
