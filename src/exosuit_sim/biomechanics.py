"""Baseline (unsuited) human joint mechanics.

Two sources are supported (docs/MODEL.md §3):

1. ``inverse_dynamics`` (default): net hip and knee torques from rigid-body
   inverse dynamics of a planar two-link leg with anthropometric ratios that
   approximately follow Winter (2009). This is a **synthetic, physics-based**
   reference, *not* measured physiological data.
2. ``csv``: a user-supplied, angle-indexed knee-torque table evaluated with
   monotone PCHIP interpolation.

Sign convention: torques are **net joint torques in the anatomical flexion
direction** (positive = flexion moment acting on the distal segment).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from .kinematics import LegGeometry, anatomical_to_link_angles, planar_forward_kinematics
from .trajectory import JointTrajectory

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SegmentInertia:
    """Rigid-segment inertial properties.

    Attributes
    ----------
    mass_kg
        Segment mass.
    com_from_proximal_m
        Distance from the proximal joint to the centre of mass, along the segment.
    inertia_com_kg_m2
        Moment of inertia about the transverse axis through the centre of mass.
    """

    mass_kg: float
    com_from_proximal_m: float
    inertia_com_kg_m2: float

    def __post_init__(self) -> None:
        if self.mass_kg < 0 or self.com_from_proximal_m < 0 or self.inertia_com_kg_m2 < 0:
            raise ValueError("Segment mass, COM distance and inertia must all be non-negative")


@dataclass(frozen=True)
class LegInertialModel:
    """Thigh and shank-plus-foot inertial model in a gravity field (m/s²)."""

    thigh: SegmentInertia
    shank_foot: SegmentInertia
    gravity_m_s2: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.gravity_m_s2) or self.gravity_m_s2 < 0:
            raise ValueError(f"gravity_m_s2 must be finite and >= 0, got {self.gravity_m_s2!r}")


def segment_from_fractions(
    body_mass_kg: float, length_m: float, mass_fraction: float, com_fraction: float, rog_fraction: float
) -> SegmentInertia:
    """Segment properties from body-mass and segment-length fractions.

    ``I_com = m (ρ L)²`` where ``ρ`` is the radius of gyration about the COM as a
    fraction of segment length.
    """
    mass = body_mass_kg * mass_fraction
    return SegmentInertia(
        mass_kg=mass,
        com_from_proximal_m=com_fraction * length_m,
        inertia_com_kg_m2=mass * (rog_fraction * length_m) ** 2,
    )


def combine_with_point_mass(segment: SegmentInertia, point_mass_kg: float, point_distance_m: float) -> SegmentInertia:
    """Lump a point mass (e.g. the foot at the ankle) into a segment (parallel-axis theorem)."""
    total = segment.mass_kg + point_mass_kg
    if total <= 0:
        raise ValueError("Combined segment mass must be positive")
    com = (segment.mass_kg * segment.com_from_proximal_m + point_mass_kg * point_distance_m) / total
    inertia = (
        segment.inertia_com_kg_m2
        + segment.mass_kg * (segment.com_from_proximal_m - com) ** 2
        + point_mass_kg * (point_distance_m - com) ** 2
    )
    return SegmentInertia(mass_kg=total, com_from_proximal_m=com, inertia_com_kg_m2=inertia)


def build_inertial_model(
    body_mass_kg: float,
    geometry: LegGeometry,
    gravity_m_s2: float,
    thigh_fractions: Tuple[float, float, float] = (0.100, 0.433, 0.323),
    shank_fractions: Tuple[float, float, float] = (0.0465, 0.433, 0.302),
    foot_mass_fraction: float = 0.0145,
) -> LegInertialModel:
    """Build the two-link inertial model; the foot is a point mass at the ankle.

    Fractions are ``(mass, COM-from-proximal, radius-of-gyration-about-COM)``.
    The defaults approximately follow Winter (2009) and are illustrative.
    """
    thigh = segment_from_fractions(body_mass_kg, geometry.thigh_length_m, *thigh_fractions)
    shank = segment_from_fractions(body_mass_kg, geometry.shank_length_m, *shank_fractions)
    shank_foot = combine_with_point_mass(shank, body_mass_kg * foot_mass_fraction, geometry.shank_length_m)
    return LegInertialModel(thigh=thigh, shank_foot=shank_foot, gravity_m_s2=gravity_m_s2)


# --------------------------------------------------------------------------- #
# Inverse dynamics
# --------------------------------------------------------------------------- #
def two_link_inverse_dynamics(
    trajectory: JointTrajectory, geometry: LegGeometry, model: LegInertialModel
) -> Tuple[FloatArray, FloatArray]:
    """Net hip and knee torques ``τ = M(θ)θ̈ + C(θ, θ̇) + G(θ)`` (hip fixed in space).

    Link angles are measured CCW from the downward vertical (docs/MODEL.md §3).

    Returns
    -------
    hip_flexion_torque_Nm, knee_flexion_torque_Nm
        Net joint torques in the anatomical flexion direction.
    """
    th1, th2 = anatomical_to_link_angles(trajectory.hip_rad, trajectory.knee_rad)
    dth1, dth2 = trajectory.hip_vel_rad_s, -trajectory.knee_vel_rad_s
    ddth1, ddth2 = trajectory.hip_acc_rad_s2, -trajectory.knee_acc_rad_s2

    l1 = geometry.thigh_length_m
    m1, c1, i1 = model.thigh.mass_kg, model.thigh.com_from_proximal_m, model.thigh.inertia_com_kg_m2
    m2, c2, i2 = model.shank_foot.mass_kg, model.shank_foot.com_from_proximal_m, model.shank_foot.inertia_com_kg_m2
    g = model.gravity_m_s2

    cos2, sin2 = np.cos(th2), np.sin(th2)
    m11 = i1 + i2 + m1 * c1**2 + m2 * (l1**2 + c2**2 + 2 * l1 * c2 * cos2)
    m12 = i2 + m2 * (c2**2 + l1 * c2 * cos2)
    m22 = i2 + m2 * c2**2
    h = m2 * l1 * c2 * sin2
    g1 = g * ((m1 * c1 + m2 * l1) * np.sin(th1) + m2 * c2 * np.sin(th1 + th2))
    g2 = g * m2 * c2 * np.sin(th1 + th2)

    tau1 = m11 * ddth1 + m12 * ddth2 - h * (2 * dth1 * dth2 + dth2**2) + g1
    tau2 = m12 * ddth1 + m22 * ddth2 + h * dth1**2 + g2
    # θ1 is hip flexion (CCW); θ2 = −knee flexion, so the knee flexion torque is −τ2.
    return tau1, -tau2


def leg_mechanical_energy(trajectory: JointTrajectory, geometry: LegGeometry, model: LegInertialModel) -> FloatArray:
    """Kinetic + gravitational potential energy of the leg (J), for validation."""
    th1, th2 = anatomical_to_link_angles(trajectory.hip_rad, trajectory.knee_rad)
    dth1, dth2 = trajectory.hip_vel_rad_s, -trajectory.knee_vel_rad_s
    l1 = geometry.thigh_length_m
    m1, c1, i1 = model.thigh.mass_kg, model.thigh.com_from_proximal_m, model.thigh.inertia_com_kg_m2
    m2, c2, i2 = model.shank_foot.mass_kg, model.shank_foot.com_from_proximal_m, model.shank_foot.inertia_com_kg_m2
    g = model.gravity_m_s2

    a1, a2 = th1, th1 + th2
    # COM velocities from differentiating p1 = c1(sin a1, −cos a1), p2 = L1(...) + c2(sin a2, −cos a2)
    v1_sq = (c1 * dth1) ** 2
    da2 = dth1 + dth2
    v2x = l1 * np.cos(a1) * dth1 + c2 * np.cos(a2) * da2
    v2y = l1 * np.sin(a1) * dth1 + c2 * np.sin(a2) * da2
    kinetic = 0.5 * m1 * v1_sq + 0.5 * i1 * dth1**2 + 0.5 * m2 * (v2x**2 + v2y**2) + 0.5 * i2 * da2**2
    y1 = -c1 * np.cos(a1)
    y2 = -l1 * np.cos(a1) - c2 * np.cos(a2)
    potential = m1 * g * y1 + m2 * g * y2
    return kinetic + potential


# --------------------------------------------------------------------------- #
# Tabulated (measured or synthetic) torque–angle profiles
# --------------------------------------------------------------------------- #
class TabulatedTorqueProfile:
    """Angle-indexed knee torque evaluated with monotone PCHIP interpolation.

    Valid only for quasi-static or monotonic movements: the same angle visited
    twice (e.g. flexion then extension) is assumed to require the same torque.
    Extrapolation outside the tabulated range is refused.
    """

    def __init__(self, angles_rad: ArrayLike, torques_Nm: ArrayLike, label: str = "tabulated") -> None:
        angles = np.asarray(angles_rad, dtype=float).reshape(-1)
        torques = np.asarray(torques_Nm, dtype=float).reshape(-1)
        if angles.size != torques.size:
            raise ValueError(f"angles ({angles.size}) and torques ({torques.size}) must have the same length")
        if angles.size < 2:
            raise ValueError("At least two (angle, torque) samples are required for interpolation")
        if not (np.all(np.isfinite(angles)) and np.all(np.isfinite(torques))):
            raise ValueError("Tabulated angles and torques must be finite")
        if np.any(np.diff(angles) <= 0):
            raise ValueError("Tabulated angles must be strictly increasing")
        self.angles_rad = angles
        self.torques_Nm = torques
        self.label = label
        self._interp = PchipInterpolator(angles, torques, extrapolate=False)

    @property
    def angle_range_rad(self) -> Tuple[float, float]:
        return float(self.angles_rad[0]), float(self.angles_rad[-1])

    def __call__(self, angle_rad: ArrayLike, tolerance_rad: float = 1e-9) -> FloatArray:
        angle = np.asarray(angle_rad, dtype=float)
        lo, hi = self.angle_range_rad
        if np.any(angle < lo - tolerance_rad) or np.any(angle > hi + tolerance_rad):
            raise ValueError(
                f"Requested knee angle outside tabulated range [{np.degrees(lo):.2f}, {np.degrees(hi):.2f}] deg; "
                "extend the data rather than extrapolating."
            )
        return self._interp(np.clip(angle, lo, hi))


def load_torque_csv(path: Union[str, Path]) -> TabulatedTorqueProfile:
    """Load a ``knee_flexion_deg, knee_torque_Nm`` CSV (header required; '#' lines ignored)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Baseline torque CSV not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        rows = [line for line in handle if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(rows)
    required = {"knee_flexion_deg", "knee_torque_Nm"}
    if reader.fieldnames is None or not required.issubset({f.strip() for f in reader.fieldnames}):
        raise ValueError(f"{path} must have header columns {sorted(required)}, got {reader.fieldnames}")
    angles, torques = [], []
    for line_no, row in enumerate(reader, start=2):
        row = {k.strip(): v for k, v in row.items()}
        try:
            angles.append(float(row["knee_flexion_deg"]))
            torques.append(float(row["knee_torque_Nm"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: could not parse numeric values on data row {line_no}: {row}") from exc
    return TabulatedTorqueProfile(np.radians(angles), torques, label=path.name)


def quasi_static_knee_torque(knee_rad: ArrayLike, hip_rad: float, model: LegInertialModel) -> FloatArray:
    """Gravity-only knee flexion torque ``m₂ g c₂ sin(φ − q_h)`` (used for example tables)."""
    knee = np.asarray(knee_rad, dtype=float)
    sf = model.shank_foot
    return sf.mass_kg * model.gravity_m_s2 * sf.com_from_proximal_m * np.sin(knee - hip_rad)


def ankle_path(trajectory: JointTrajectory, geometry: LegGeometry) -> FloatArray:
    """Global ankle trajectory (m), convenience wrapper over forward kinematics."""
    th1, th2 = anatomical_to_link_angles(trajectory.hip_rad, trajectory.knee_rad)
    return planar_forward_kinematics(th1, th2, geometry, frame="global")[1]
