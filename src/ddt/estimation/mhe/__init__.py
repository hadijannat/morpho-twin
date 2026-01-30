"""Moving Horizon Estimation (MHE) module.

Provides production-grade MHE with:
- CasADi symbolic model building
- acados-based MHE (primary, uses HPIPM)
- CasADi + IPOPT fallback
- Covariance extraction from Hessian
- Warm-start trajectory shifting
- EKF-based arrival cost updating
"""

from __future__ import annotations

from .base import MHEBase
from .casadi_mhe import CasADiMHE
from .covariance import extract_covariance
from .ekf_arrival import EKFArrivalCostUpdater
from .warm_start import shift_trajectory

__all__ = [
    "MHEBase",
    "CasADiMHE",
    "EKFArrivalCostUpdater",
    "extract_covariance",
    "shift_trajectory",
]

# acados MHE is conditionally available
try:
    from .acados_mhe import AcadosMHE  # noqa: F401

    __all__.append("AcadosMHE")
except ImportError:
    pass
