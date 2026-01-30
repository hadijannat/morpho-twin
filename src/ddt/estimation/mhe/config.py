"""MHE configuration dataclasses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MHENoiseConfig(BaseModel):
    """Process and measurement noise configuration."""

    Q_diag: list[float] = Field(
        default=[1.0],
        description="Process noise covariance diagonal (inverse weighting).",
    )
    R_diag: list[float] = Field(
        default=[1.0],
        description="Measurement noise covariance diagonal (inverse weighting).",
    )


class MHEParameterConfig(BaseModel):
    """Parameter estimation configuration."""

    mode: str = Field(
        default="static",
        description="Parameter dynamics: 'static' or 'random_walk'.",
    )
    theta_init: list[float] = Field(
        default=[1.0, 0.1],
        description="Initial parameter guess [a, b] for linear dynamics.",
    )
    P_theta_diag: list[float] = Field(
        default=[1.0, 1.0],
        description="Prior parameter covariance diagonal.",
    )
    random_walk_std: list[float] = Field(
        default=[0.0, 0.0],
        description="Random walk std for each parameter (if mode='random_walk').",
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
    arrival_cost_scaling: float = Field(
        default=1.0,
        description="Scaling factor for arrival cost term.",
    )
    use_ekf_arrival_cost: bool = Field(
        default=False,
        description="Use EKF to dynamically update arrival cost prior.",
    )
