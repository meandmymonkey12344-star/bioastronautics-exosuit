"""Prescribed joint-angle trajectories (minimum-jerk).

All angles are in radians, angular velocities in rad/s and angular accelerations
in rad/s^2. The knee angle is the *anatomical flexion* angle ``phi >= 0``; the hip
angle is hip flexion measured from the downward vertical (see docs/MODEL.md §2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointTrajectory:
    """Sampled hip and knee kinematics on a uniform time grid."""

    time_s: FloatArray
    hip_rad: FloatArray
    hip_vel_rad_s: FloatArray
    hip_acc_rad_s2: FloatArray
    knee_rad: FloatArray
    knee_vel_rad_s: FloatArray
    knee_acc_rad_s2: FloatArray

    def __post_init__(self) -> None:
        n = self.time_s.shape
        for name in (
            "hip_rad",
            "hip_vel_rad_s",
            "hip_acc_rad_s2",
            "knee_rad",
            "knee_vel_rad_s",
            "knee_acc_rad_s2",
        ):
            arr = getattr(self, name)
            if arr.shape != n:
                raise ValueError(f"{name} has shape {arr.shape}, expected {n} to match time_s")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains non-finite values")
        if self.time_s.ndim != 1 or self.time_s.size < 3:
            raise ValueError("time_s must be a 1-D array with at least 3 samples")
        steps = np.diff(self.time_s)
        if np.any(steps <= 0):
            raise ValueError("time_s must be strictly increasing")
        if not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-12):
            raise ValueError("time_s must be uniformly spaced")

    @property
    def n_samples(self) -> int:
        return int(self.time_s.size)

    @property
    def dt(self) -> float:
        return float(self.time_s[1] - self.time_s[0])

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def knee_deg(self) -> FloatArray:
        return np.degrees(self.knee_rad)

    @property
    def hip_deg(self) -> FloatArray:
        return np.degrees(self.hip_rad)


def minimum_jerk(
    start: float, end: float, duration_s: float, time_s: FloatArray
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Minimum-jerk point-to-point profile (Flash & Hogan, 1985).

    ``q(t) = q0 + (q1 - q0) * (10 s^3 - 15 s^4 + 6 s^5)`` with ``s = t / T``.
    Velocity and acceleration are zero at both ends.

    Returns
    -------
    position, velocity, acceleration
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    s = np.clip(np.asarray(time_s, dtype=float) / duration_s, 0.0, 1.0)
    delta = end - start
    pos = start + delta * (10 * s**3 - 15 * s**4 + 6 * s**5)
    vel = delta * (30 * s**2 - 60 * s**3 + 30 * s**4) / duration_s
    acc = delta * (60 * s - 180 * s**2 + 120 * s**3) / duration_s**2
    return pos, vel, acc


def build_trajectory(
    knee_start_deg: float,
    knee_end_deg: float,
    duration_s: float,
    n_samples: int,
    mode: str = "flexion",
    hip_start_deg: float = 0.0,
    hip_end_deg: float = 0.0,
) -> JointTrajectory:
    """Build a uniformly sampled hip/knee trajectory.

    Parameters
    ----------
    knee_start_deg, knee_end_deg
        Anatomical knee flexion at the start and the turning point (deg).
    duration_s
        Duration of *one* movement phase (s). ``flexion_extension`` lasts ``2 * duration_s``.
    n_samples
        Number of samples per phase (the turning point is shared between phases).
    mode
        ``"flexion"`` (start -> end) or ``"flexion_extension"`` (start -> end -> start).
    hip_start_deg, hip_end_deg
        Hip flexion; equal values give a fixed hip (pure knee movement).
    """
    if mode not in ("flexion", "flexion_extension"):
        raise ValueError(f"mode must be 'flexion' or 'flexion_extension', got {mode!r}")
    if n_samples < 3:
        raise ValueError(f"n_samples must be >= 3, got {n_samples}")
    k0, k1 = np.radians(knee_start_deg), np.radians(knee_end_deg)
    h0, h1 = np.radians(hip_start_deg), np.radians(hip_end_deg)
    t_phase = np.linspace(0.0, duration_s, n_samples)

    knee = minimum_jerk(k0, k1, duration_s, t_phase)
    hip = minimum_jerk(h0, h1, duration_s, t_phase)
    time = t_phase
    if mode == "flexion_extension":
        knee_back = minimum_jerk(k1, k0, duration_s, t_phase)
        hip_back = minimum_jerk(h1, h0, duration_s, t_phase)
        time = np.concatenate([t_phase, duration_s + t_phase[1:]])
        knee = tuple(np.concatenate([a, b[1:]]) for a, b in zip(knee, knee_back))  # type: ignore[assignment]
        hip = tuple(np.concatenate([a, b[1:]]) for a, b in zip(hip, hip_back))  # type: ignore[assignment]

    return JointTrajectory(
        time_s=time,
        hip_rad=hip[0],
        hip_vel_rad_s=hip[1],
        hip_acc_rad_s2=hip[2],
        knee_rad=knee[0],
        knee_vel_rad_s=knee[1],
        knee_acc_rad_s2=knee[2],
    )
