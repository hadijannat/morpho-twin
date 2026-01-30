"""Dual-control module for active learning in NMPC."""

from __future__ import annotations

from .fim_compute import compute_fim_prediction, propagate_sensitivity
from .probing_cost import (
    a_optimal_cost,
    compute_fim_cost,
    compute_probing_input_modification,
    d_optimal_cost,
    e_optimal_cost,
)

__all__ = [
    "compute_fim_prediction",
    "propagate_sensitivity",
    "a_optimal_cost",
    "d_optimal_cost",
    "e_optimal_cost",
    "compute_fim_cost",
    "compute_probing_input_modification",
]
