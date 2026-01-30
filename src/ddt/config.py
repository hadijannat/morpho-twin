"""Configuration dataclasses for Morpho Twin."""

from __future__ import annotations

from pydantic import BaseModel, Field

# =============================================================================
# Plant Configuration
# =============================================================================


class PlantConfig(BaseModel):
    type: str = Field(..., description="Plant type identifier.")
    a_true: float | None = None
    b_true: float | None = None
    process_noise_std: float = 0.0
    meas_noise_std: float = 0.0
    x0: float = 0.0


# =============================================================================
# MHE Configuration
# =============================================================================


class MHENoiseConfig(BaseModel):
    """Process and measurement noise configuration for MHE."""

    Q_diag: list[float] = Field(
        default=[1.0],
        description="Process noise covariance diagonal.",
    )
    R_diag: list[float] = Field(
        default=[1.0],
        description="Measurement noise covariance diagonal.",
    )


class MHEParameterConfig(BaseModel):
    """Parameter estimation configuration for MHE."""

    mode: str = Field(
        default="static",
        description="Parameter dynamics: 'static' or 'random_walk'.",
    )
    theta_init: list[float] = Field(
        default=[1.0, 0.1],
        description="Initial parameter guess [a, b].",
    )
    P_theta_diag: list[float] = Field(
        default=[1.0, 1.0],
        description="Prior parameter covariance diagonal.",
    )


class MHESolverConfig(BaseModel):
    """MHE solver backend configuration."""

    backend: str = Field(
        default="casadi",
        description="Solver backend: 'acados' or 'casadi'.",
    )
    max_iter: int = Field(default=50, description="Maximum solver iterations.")
    tol: float = Field(default=1e-6, description="Solver tolerance.")
    print_level: int = Field(default=0, description="Solver verbosity.")


class MHEConfig(BaseModel):
    """Full MHE configuration."""

    horizon: int = Field(default=20, description="MHE horizon length.")
    noise: MHENoiseConfig = Field(default_factory=MHENoiseConfig)
    parameters: MHEParameterConfig = Field(default_factory=MHEParameterConfig)
    solver: MHESolverConfig = Field(default_factory=MHESolverConfig)
    arrival_cost_scaling: float = Field(default=1.0)
    use_ekf_arrival_cost: bool = Field(
        default=False,
        description="Use EKF to dynamically update arrival cost prior.",
    )


# =============================================================================
# Estimation Configuration
# =============================================================================


class EstimationConfig(BaseModel):
    type: str = Field(
        default="windowed_ls",
        description="Estimator type: 'windowed_ls' or 'mhe'.",
    )
    window: int = 50  # For windowed_ls
    mhe: MHEConfig | None = None  # For MHE


# =============================================================================
# NMPC Configuration
# =============================================================================


class NMPCConfig(BaseModel):
    """NMPC controller configuration."""

    horizon: int = Field(default=20, description="Prediction horizon.")
    Q: list[float] = Field(default=[10.0], description="State tracking weight diagonal.")
    R_u: list[float] = Field(default=[0.1], description="Input regularization diagonal.")
    lambda_info: float = Field(default=0.01, description="FIM probing weight for dual-control.")
    fim_criterion: str = Field(
        default="d_optimal",
        description="FIM criterion: 'd_optimal' (recommended), 'a_optimal', or 'e_optimal'.",
    )
    solver_backend: str = Field(default="casadi", description="'acados' or 'casadi'.")
    max_iter: int = Field(default=50, description="Max solver iterations (1 for RTI).")
    rti_mode: bool = Field(default=True, description="Use RTI scheme for acados.")


# =============================================================================
# Control Configuration
# =============================================================================


class ControlConfig(BaseModel):
    type: str = Field(
        default="pid",
        description="Controller type: 'pid', 'nmpc_rti', or 'nmpc_casadi'.",
    )
    # PID parameters
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0
    u_min: float = -1.0
    u_max: float = 1.0
    # NMPC configuration
    nmpc: NMPCConfig | None = None


# =============================================================================
# CBF Configuration
# =============================================================================


class BarrierConfig(BaseModel):
    """Barrier function configuration."""

    type: str = Field(default="box", description="Barrier type: 'box' or 'ellipsoid'.")
    # For ellipsoid
    center: list[float] | None = None
    P_diag: list[float] | None = None


class CBFConfig(BaseModel):
    """CBF-QP safety filter configuration."""

    alpha: float = Field(default=0.5, description="CBF class-K function parameter.")
    slack_weight: float = Field(default=1000.0, description="Slack penalty.")
    gamma_robust: float = Field(default=1.0, description="Robust margin scaling.")
    barriers: list[BarrierConfig] = Field(default_factory=list)


# =============================================================================
# Safety Configuration
# =============================================================================


class SafetyConfig(BaseModel):
    type: str = Field(
        default="state_box",
        description="Safety filter type: 'state_box' or 'cbf_qp'.",
    )
    x_min: float = -1.0
    x_max: float = 1.0
    alpha: float = 0.05
    slack_weight: float = 1000.0
    cbf: CBFConfig | None = None


# =============================================================================
# Supervision Configuration
# =============================================================================


class PEConfig(BaseModel):
    """PE monitoring configuration."""

    window: int = Field(default=100, description="Rolling FIM window size.")
    lambda_threshold: float = Field(default=0.1, description="PE eigenvalue threshold.")
    condition_threshold: float = Field(default=100.0, description="Max FIM condition number.")


class ModeConfig(BaseModel):
    """Mode manager configuration."""

    uncertainty_normal_to_conservative: float = 1.0
    uncertainty_conservative_to_safe: float = 5.0
    uncertainty_safe_to_conservative: float = 3.0
    uncertainty_conservative_to_normal: float = 0.5
    pe_violation_count_conservative: int = 10
    pe_violation_count_safe: int = 50
    pe_satisfaction_count_recovery: int = 20
    margin_factor_normal: float = 1.0
    margin_factor_conservative: float = 2.0
    margin_factor_safe: float = 5.0


class SupervisionConfig(BaseModel):
    """Full supervision configuration."""

    enabled: bool = Field(default=False, description="Enable supervision.")
    pe: PEConfig = Field(default_factory=PEConfig)
    mode: ModeConfig = Field(default_factory=ModeConfig)


# =============================================================================
# Scenario Configuration
# =============================================================================


class ReferenceConfig(BaseModel):
    type: str = "step"
    t_step: float = 0.0
    value: float = 0.0


class ScenarioConfig(BaseModel):
    steps: int = 1000
    reference: ReferenceConfig = Field(default_factory=ReferenceConfig)


# =============================================================================
# Top-Level Application Configuration
# =============================================================================


class AppConfig(BaseModel):
    """Top-level Morpho Twin application configuration."""

    seed: int = 0
    dt: float = 0.1
    horizon_steps: int = 20

    plant: PlantConfig
    estimation: EstimationConfig = Field(default_factory=EstimationConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)
    supervision: SupervisionConfig = Field(default_factory=SupervisionConfig)
