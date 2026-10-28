"""Illustrative nonlinear, hysteretic spacesuit knee-resistance model.

    τ_suit(φ, s) = E(φ) · (1 + h·s) + τ_f · s,      E(φ) = k₁ φ + k₂ φ |φ|

* ``τ_suit`` is the torque the suit applies **against flexion** (positive acts in
  the extension direction). Its signed flexion-direction contribution to the
  knee moment balance is ``−τ_suit``.
* ``s ∈ [−1, 1]`` is a rate-independent branch state with memory
  (+1 = flexion/loading branch, −1 = extension/unloading branch), updated per
  angle increment with ``s ← s + (sgn Δφ − s)(1 − exp(−|Δφ|/φ_t))``.

With ``k₁ = τ_f = h = 0`` this reduces exactly to the brief's ``τ = k θ²``.

This law is **not** a validated model of gas displacement or EVA suit
mechanics; coefficients are illustrative (docs/MODEL.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SuitResistanceModel:
    """Parameters of the suit knee-resistance law.

    Attributes
    ----------
    k_linear_Nm_per_rad
        Linear stiffness k₁ (N·m/rad).
    k_quadratic_Nm_per_rad2
        Quadratic stiffness k₂ (N·m/rad²) — the brief's ``k``.
    friction_Nm
        Direction-dependent breakaway/friction torque τ_f (N·m).
    hysteresis_fraction
        h ∈ [0, 1): fractional elastic-torque increase on the loading branch and
        decrease on the unloading branch.
    transition_angle_rad
        φ_t (rad): angle over which the branch state switches after a reversal.
    initial_branch_state
        s at the first sample (0 = relaxed, +1 loading, −1 unloading).
    """

    k_linear_Nm_per_rad: float = 6.0
    k_quadratic_Nm_per_rad2: float = 4.0
    friction_Nm: float = 1.0
    hysteresis_fraction: float = 0.25
    transition_angle_rad: float = np.radians(3.0)
    initial_branch_state: float = 0.0

    def __post_init__(self) -> None:
        for name in ("k_linear_Nm_per_rad", "k_quadratic_Nm_per_rad2", "friction_Nm"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if not 0 <= self.hysteresis_fraction < 1:
            raise ValueError(f"hysteresis_fraction must lie in [0, 1), got {self.hysteresis_fraction!r}")
        if not np.isfinite(self.transition_angle_rad) or self.transition_angle_rad <= 0:
            raise ValueError(f"transition_angle_rad must be > 0, got {self.transition_angle_rad!r}")
        if not -1 <= self.initial_branch_state <= 1:
            raise ValueError(f"initial_branch_state must lie in [-1, 1], got {self.initial_branch_state!r}")

    @classmethod
    def quadratic_only(cls, k_Nm_per_rad2: float) -> "SuitResistanceModel":
        """The brief's original ``τ = k θ²`` law (no linear term, friction or hysteresis)."""
        return cls(k_linear_Nm_per_rad=0.0, k_quadratic_Nm_per_rad2=k_Nm_per_rad2, friction_Nm=0.0, hysteresis_fraction=0.0)

    def scaled(self, factor: float) -> "SuitResistanceModel":
        """Return a copy with stiffness and friction scaled by ``factor`` (sensitivity studies)."""
        if not np.isfinite(factor) or factor < 0:
            raise ValueError(f"scale factor must be finite and >= 0, got {factor!r}")
        return SuitResistanceModel(
            k_linear_Nm_per_rad=self.k_linear_Nm_per_rad * factor,
            k_quadratic_Nm_per_rad2=self.k_quadratic_Nm_per_rad2 * factor,
            friction_Nm=self.friction_Nm * factor,
            hysteresis_fraction=self.hysteresis_fraction,
            transition_angle_rad=self.transition_angle_rad,
            initial_branch_state=self.initial_branch_state,
        )

    def elastic_torque(self, knee_rad: ArrayLike) -> FloatArray:
        """Restoring elastic torque E(φ) = k₁φ + k₂φ|φ| (N·m); odd in φ, zero at neutral."""
        phi = np.asarray(knee_rad, dtype=float)
        return self.k_linear_Nm_per_rad * phi + self.k_quadratic_Nm_per_rad2 * phi * np.abs(phi)

    def branch_state(self, knee_path_rad: ArrayLike) -> FloatArray:
        """Rate-independent hysteresis branch state s along a sampled angle path."""
        phi = np.asarray(knee_path_rad, dtype=float).reshape(-1)
        if phi.size == 0:
            raise ValueError("knee_path_rad must contain at least one sample")
        s = np.empty_like(phi)
        s[0] = self.initial_branch_state
        increments = np.diff(phi)
        relax = 1.0 - np.exp(-np.abs(increments) / self.transition_angle_rad)
        direction = np.sign(increments)
        for i, (d, r) in enumerate(zip(direction, relax)):
            s[i + 1] = s[i] + (d - s[i]) * r if d != 0 else s[i]
        return s

    def torque(self, knee_path_rad: ArrayLike) -> FloatArray:
        """Suit resistive torque τ_suit along a sampled knee-angle path (N·m)."""
        phi = np.asarray(knee_path_rad, dtype=float).reshape(-1)
        s = self.branch_state(phi)
        elastic = self.elastic_torque(phi)
        return elastic * (1.0 + self.hysteresis_fraction * s) + self.friction_Nm * s
