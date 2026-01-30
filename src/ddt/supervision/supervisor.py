"""Unified supervisor interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .mode_manager import ModeConfig, ModeManager, OperationMode
from .pe_monitor import PEMonitor, PEStatus

if TYPE_CHECKING:
    from ..interfaces import Estimate


@dataclass(frozen=True)
class SupervisorState:
    """Current supervisor state snapshot."""

    mode: OperationMode
    pe_status: PEStatus
    safety_margin_factor: float
    uncertainty_norm: float


@dataclass
class Supervisor:
    """Unified supervisor for DDT system.

    Integrates:
    - PE monitoring for parameter identifiability
    - Mode management for safety margin adjustment
    - Recommended probing weight for dual-control

    The supervisor observes estimation and control state and
    adjusts the safety filter margins accordingly.
    """

    # Configuration
    pe_window: int = 100
    pe_lambda_threshold: float = 0.1
    ntheta: int = 2
    mode_config: ModeConfig = field(default_factory=ModeConfig)

    # Components
    _pe_monitor: PEMonitor = field(init=False)
    _mode_manager: ModeManager = field(init=False)
    _last_state: SupervisorState | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._pe_monitor = PEMonitor(
            window=self.pe_window,
            lambda_threshold=self.pe_lambda_threshold,
            ntheta=self.ntheta,
        )
        self._mode_manager = ModeManager(config=self.mode_config)

    def reset(self) -> None:
        """Reset supervisor state."""
        self._pe_monitor.reset()
        self._mode_manager.reset()
        self._last_state = None

    def update(
        self,
        estimate: Estimate,
        regressor: np.ndarray | None = None,
    ) -> SupervisorState:
        """Update supervisor with new estimation data.

        Args:
            estimate: Current state and parameter estimate
            regressor: Optional regressor for PE monitoring
                If None, uses default [x_hat, 1] for scalar systems

        Returns:
            Current supervisor state
        """
        # Compute uncertainty norm
        theta_cov = np.atleast_2d(estimate.theta_cov)
        uncertainty_norm = float(np.trace(theta_cov))

        # Update PE monitor
        if regressor is None:
            # Default regressor for linear scalar system
            x = np.atleast_1d(estimate.x_hat)
            regressor = np.concatenate([x, np.ones(1)])

        pe_status = self._pe_monitor.update(regressor)

        # Update mode manager
        mode = self._mode_manager.update(
            uncertainty_norm=uncertainty_norm,
            is_pe_satisfied=pe_status.is_pe_satisfied,
        )

        # Build state snapshot
        state = SupervisorState(
            mode=mode,
            pe_status=pe_status,
            safety_margin_factor=self._mode_manager.safety_margin_factor,
            uncertainty_norm=uncertainty_norm,
        )

        self._last_state = state
        return state

    def get_state(self) -> SupervisorState | None:
        """Get last supervisor state."""
        return self._last_state

    @property
    def mode(self) -> OperationMode:
        """Current operation mode."""
        return self._mode_manager.mode

    @property
    def safety_margin_factor(self) -> float:
        """Current safety margin factor."""
        return self._mode_manager.safety_margin_factor

    def get_recommended_probe_weight(self) -> float:
        """Get recommended probing weight for dual-control."""
        if self._last_state is None:
            return 0.0
        return self._last_state.pe_status.recommended_probe_weight

    def is_safe_to_operate(self) -> bool:
        """Check if system is safe to operate."""
        return self._mode_manager.mode != OperationMode.SAFE_STOP

    def trigger_safe_stop(self) -> None:
        """Explicitly trigger safe stop mode."""
        self._mode_manager.trigger_safe_stop()

    def get_zero_control(self, nu: int = 1) -> np.ndarray:
        """Get zero control for safe stop mode.

        In SAFE_STOP mode, the system should apply zero or
        minimum control to bring the system to rest.

        Args:
            nu: Input dimension

        Returns:
            Zero control vector
        """
        return np.zeros(nu)


def create_supervisor(
    ntheta: int = 2,
    pe_window: int = 100,
    pe_threshold: float = 0.1,
    **mode_kwargs: Any,
) -> Supervisor:
    """Factory function for creating a supervisor.

    Args:
        ntheta: Number of parameters
        pe_window: Rolling window for PE monitoring
        pe_threshold: Minimum eigenvalue threshold for PE
        **mode_kwargs: Additional arguments for ModeConfig

    Returns:
        Configured Supervisor instance
    """
    mode_config = ModeConfig(**mode_kwargs)

    return Supervisor(
        pe_window=pe_window,
        pe_lambda_threshold=pe_threshold,
        ntheta=ntheta,
        mode_config=mode_config,
    )
