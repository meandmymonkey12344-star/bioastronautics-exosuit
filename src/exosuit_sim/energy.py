"""Mechanical work, electrical-energy estimate and performance metrics (docs/MODEL.md §7).

Mechanical work is ``W = ∫ τ dφ``, discretised with trapezoidal increments
``ΔWᵢ = ½(τᵢ + τᵢ₊₁)(φᵢ₊₁ − φᵢ)``. This rule is exact for constant torque on any
trajectory and makes ``W⁺ + W⁻ = W_net`` hold identically.

None of these quantities is human metabolic energy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import trapezoid

from .actuator import ActuatorModel

FloatArray = NDArray[np.float64]


def _as_matching(torque: ArrayLike, angle: ArrayLike) -> "tuple[FloatArray, FloatArray]":
    tau = np.asarray(torque, dtype=float).reshape(-1)
    phi = np.asarray(angle, dtype=float).reshape(-1)
    if tau.size != phi.size:
        raise ValueError(f"torque ({tau.size}) and angle ({phi.size}) must have the same number of samples")
    if tau.size < 2:
        raise ValueError("At least two samples are required to integrate work")
    return tau, phi


def work_increments(torque_Nm: ArrayLike, angle_rad: ArrayLike) -> FloatArray:
    """Trapezoidal work increments ΔWᵢ (J) between consecutive samples."""
    tau, phi = _as_matching(torque_Nm, angle_rad)
    return 0.5 * (tau[:-1] + tau[1:]) * np.diff(phi)


def cumulative_work(torque_Nm: ArrayLike, angle_rad: ArrayLike) -> FloatArray:
    """Cumulative work W(t) (J), starting at 0, same length as the inputs."""
    return np.concatenate([[0.0], np.cumsum(work_increments(torque_Nm, angle_rad))])


@dataclass(frozen=True)
class WorkSummary:
    """Positive, negative and net mechanical work (J), plus an electrical estimate."""

    positive_J: float
    negative_J: float
    net_J: float
    electrical_J: float

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def electrical_energy(
    positive_work_J: float, negative_work_J: float, efficiency: float, regeneration_efficiency: float = 0.0
) -> float:
    """Electrical energy estimate ``E = W⁺/η − η_r |W⁻|`` (J).

    Ignores holding (I²R) losses, standby power and transmission friction, so it
    is a lower bound for a real drive.
    """
    if not 0 < efficiency <= 1:
        raise ValueError(f"efficiency must lie in (0, 1], got {efficiency!r}")
    if not 0 <= regeneration_efficiency <= 1:
        raise ValueError(f"regeneration_efficiency must lie in [0, 1], got {regeneration_efficiency!r}")
    if positive_work_J < 0 or negative_work_J > 0:
        raise ValueError("positive_work_J must be >= 0 and negative_work_J must be <= 0")
    return positive_work_J / efficiency - regeneration_efficiency * abs(negative_work_J)


def mechanical_work(
    torque_Nm: ArrayLike, angle_rad: ArrayLike, efficiency: float = 1.0, regeneration_efficiency: float = 0.0
) -> WorkSummary:
    """Split the work done by a torque along an angle path into W⁺, W⁻ and W_net."""
    dw = work_increments(torque_Nm, angle_rad)
    positive = float(np.sum(dw[dw > 0]))
    negative = float(np.sum(dw[dw < 0]))
    return WorkSummary(
        positive_J=positive,
        negative_J=negative,
        net_J=float(np.sum(dw)),
        electrical_J=electrical_energy(positive, negative, efficiency, regeneration_efficiency),
    )


def time_rms(values: ArrayLike, time_s: ArrayLike) -> float:
    """Time-weighted RMS ``sqrt(∫x² dt / T)`` (trapezoidal)."""
    x = np.asarray(values, dtype=float)
    t = np.asarray(time_s, dtype=float)
    duration = t[-1] - t[0]
    if duration <= 0:
        raise ValueError("time_s must span a positive duration")
    return float(np.sqrt(trapezoid(x**2, t) / duration))


def time_mean_abs(values: ArrayLike, time_s: ArrayLike) -> float:
    """Time-weighted mean absolute value ``∫|x| dt / T`` (trapezoidal)."""
    x = np.asarray(values, dtype=float)
    t = np.asarray(time_s, dtype=float)
    duration = t[-1] - t[0]
    if duration <= 0:
        raise ValueError("time_s must span a positive duration")
    return float(trapezoid(np.abs(x), t) / duration)


def performance_metrics(
    time_s: ArrayLike,
    knee_rad: ArrayLike,
    knee_vel_rad_s: ArrayLike,
    suit_torque_Nm: ArrayLike,
    actuator_torque_Nm: ArrayLike,
    baseline_torque_Nm: ArrayLike,
    actuator: ActuatorModel,
) -> Dict[str, float]:
    """Engineering metrics for one actuator profile.

    "Additional human torque" is Δτh = τsuit − τa (docs/MODEL.md §6.1); the
    reported reductions compare it with the unassisted case Δτh = τsuit.
    """
    t = np.asarray(time_s, dtype=float)
    phi = np.asarray(knee_rad, dtype=float)
    suit = np.asarray(suit_torque_Nm, dtype=float)
    act = np.asarray(actuator_torque_Nm, dtype=float)
    base = np.asarray(baseline_torque_Nm, dtype=float)
    power = act * np.asarray(knee_vel_rad_s, dtype=float)
    force = actuator.force_from_torque(act)

    work = mechanical_work(act, phi, actuator.efficiency, actuator.regeneration_efficiency)
    extra_unassisted = suit
    extra_assisted = suit - act
    human_unassisted = base + suit
    human_assisted = base + suit - act

    rms_extra_unassisted = time_rms(extra_unassisted, t)
    mean_extra_unassisted = time_mean_abs(extra_unassisted, t)
    rms_extra_assisted = time_rms(extra_assisted, t)
    mean_extra_assisted = time_mean_abs(extra_assisted, t)

    def _reduction(assisted: float, unassisted: float) -> float:
        return float(100.0 * (1.0 - assisted / unassisted)) if unassisted > 0 else float("nan")

    suit_work = mechanical_work(suit, phi)
    human_work_unassisted = mechanical_work(human_unassisted, phi)
    human_work_assisted = mechanical_work(human_assisted, phi)
    baseline_work = mechanical_work(base, phi)

    return {
        "peak_actuator_torque_Nm": float(np.max(np.abs(act))),
        "peak_actuator_force_N": float(np.max(np.abs(force))),
        "rms_actuator_torque_Nm": time_rms(act, t),
        "mean_abs_actuator_torque_Nm": time_mean_abs(act, t),
        "peak_actuator_power_W": float(np.max(np.abs(power))),
        "peak_actuator_torque_rate_Nm_s": float(np.max(np.abs(np.diff(act) / np.diff(t)))),
        "actuator_positive_work_J": work.positive_J,
        "actuator_negative_work_J": work.negative_J,
        "actuator_net_work_J": work.net_J,
        "actuator_electrical_energy_estimate_J": work.electrical_J,
        "peak_suit_torque_Nm": float(np.max(np.abs(suit))),
        "work_done_against_suit_J": suit_work.net_J,
        "rms_additional_human_torque_unassisted_Nm": rms_extra_unassisted,
        "rms_additional_human_torque_assisted_Nm": rms_extra_assisted,
        "peak_additional_human_torque_assisted_Nm": float(np.max(np.abs(extra_assisted))),
        "reduction_rms_additional_torque_pct": _reduction(rms_extra_assisted, rms_extra_unassisted),
        "reduction_mean_abs_additional_torque_pct": _reduction(mean_extra_assisted, mean_extra_unassisted),
        "rms_human_torque_baseline_Nm": time_rms(base, t),
        "rms_human_torque_suited_unassisted_Nm": time_rms(human_unassisted, t),
        "rms_human_torque_suited_assisted_Nm": time_rms(human_assisted, t),
        "human_positive_work_baseline_J": baseline_work.positive_J,
        "human_positive_work_suited_unassisted_J": human_work_unassisted.positive_J,
        "human_positive_work_suited_assisted_J": human_work_assisted.positive_J,
    }
