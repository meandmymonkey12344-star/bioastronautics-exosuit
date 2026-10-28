"""Active exoskeleton actuator model and limits.

The decision variable is the **joint torque** τ_a (N·m, positive = flexion
direction). A force-specified actuator acting through an effective
perpendicular moment arm r maps to joint torque as ``τ_a = F_a · r``.
The moment arm is treated as constant; see docs/MODEL.md §5 for why that is a
simplification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ActuatorModel:
    """Actuator capability limits.

    Attributes
    ----------
    max_torque_Nm
        Rated joint-torque limit of the drive (N·m).
    max_force_N
        Force limit of the linear/cable stage (N).
    moment_arm_m
        Effective perpendicular moment arm r (m), constant.
    max_torque_rate_Nm_s
        Optional |dτ/dt| limit (N·m/s); ``None`` disables it.
    efficiency
        Drive efficiency η ∈ (0, 1] for converting positive mechanical work to electrical energy.
    regeneration_efficiency
        Fraction η_r ∈ [0, 1] of absorbed (negative) work recovered electrically.
    bidirectional
        If False, the actuator can only push in the flexion direction (e.g. a pulling cable).
    """

    max_torque_Nm: float = 20.0
    max_force_N: float = 600.0
    moment_arm_m: float = 0.05
    max_torque_rate_Nm_s: Optional[float] = 40.0
    efficiency: float = 0.70
    regeneration_efficiency: float = 0.0
    bidirectional: bool = True

    def __post_init__(self) -> None:
        for name in ("max_torque_Nm", "max_force_N", "moment_arm_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")
        if self.max_torque_rate_Nm_s is not None and (
            not np.isfinite(self.max_torque_rate_Nm_s) or self.max_torque_rate_Nm_s <= 0
        ):
            raise ValueError(f"max_torque_rate_Nm_s must be positive or None, got {self.max_torque_rate_Nm_s!r}")
        if not 0 < self.efficiency <= 1:
            raise ValueError(f"efficiency must lie in (0, 1], got {self.efficiency!r}")
        if not 0 <= self.regeneration_efficiency <= 1:
            raise ValueError(f"regeneration_efficiency must lie in [0, 1], got {self.regeneration_efficiency!r}")

    @property
    def effective_torque_limit_Nm(self) -> float:
        """Joint-torque limit including the force limit: ``min(τ_max, F_max·r)``."""
        return float(min(self.max_torque_Nm, self.max_force_N * self.moment_arm_m))

    @property
    def limiting_factor(self) -> str:
        """Which rating sets the effective torque limit."""
        return "torque rating" if self.max_torque_Nm <= self.max_force_N * self.moment_arm_m else "force rating x moment arm"

    def torque_bounds(self) -> Tuple[float, float]:
        """Lower and upper joint-torque bounds (N·m)."""
        limit = self.effective_torque_limit_Nm
        return (-limit if self.bidirectional else 0.0), limit

    def force_from_torque(self, torque_Nm: ArrayLike) -> FloatArray:
        """Actuator force ``F = τ / r`` (N)."""
        return np.asarray(torque_Nm, dtype=float) / self.moment_arm_m

    def torque_from_force(self, force_N: ArrayLike) -> FloatArray:
        """Joint torque ``τ = F · r`` (N·m)."""
        return np.asarray(force_N, dtype=float) * self.moment_arm_m

    def limit_violations(self, torque_Nm: ArrayLike, dt_s: float) -> Dict[str, float]:
        """Largest amount (≥ 0) by which each limit is exceeded by a torque profile."""
        tau = np.asarray(torque_Nm, dtype=float).reshape(-1)
        lb, ub = self.torque_bounds()
        force = np.abs(self.force_from_torque(tau))
        result = {
            "torque_Nm": float(max(0.0, np.max(tau - ub), np.max(lb - tau))),
            "force_N": float(max(0.0, np.max(force - self.max_force_N))),
            "torque_rate_Nm_s": 0.0,
        }
        if self.max_torque_rate_Nm_s is not None and tau.size > 1:
            rate = np.abs(np.diff(tau)) / dt_s
            result["torque_rate_Nm_s"] = float(max(0.0, np.max(rate - self.max_torque_rate_Nm_s)))
        return result

    def is_feasible(self, torque_Nm: ArrayLike, dt_s: float, tolerance: float = 1e-6) -> bool:
        """True if no limit is exceeded by more than ``tolerance`` (in each limit's units)."""
        return all(v <= tolerance for v in self.limit_violations(torque_Nm, dt_s).values())
