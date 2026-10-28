"""Kinematic transformation engine for the planar two-link leg.

Frames (docs/MODEL.md §2)
-------------------------
* Global frame {G}: origin at the hip joint centre, x anterior, y superior,
  z = x × y lateral. Positive rotations are counter-clockwise (CCW) about +z.
* Leg frame {L}: {G} rotated by -90° about z, so x_L points down the neutral
  (standing) leg. The textbook formula
  ``P = R(θ1)[L1, 0] + R(θ1 + θ2)[L2, 0]`` is exact in {L}.

Joint variables
---------------
* ``θ1`` (link angle) = anatomical hip flexion (CCW, thigh forward).
* ``θ2`` (link angle) = −(anatomical knee flexion): flexion folds the shank
  backward, which is a clockwise rotation.

The homogeneous-transform chain uses 4×4 SE(3) matrices so that the planar
model can later be replaced by a 3-D chain (hip ball joint, tibial rotation)
without changing the interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Scalar = Union[float, np.floating]

#: Rotation (rad) from the global frame {G} to the leg frame {L}.
LEG_FRAME_OFFSET_RAD: float = -np.pi / 2


@dataclass(frozen=True)
class LegGeometry:
    """Rigid segment lengths in metres."""

    thigh_length_m: float = 0.45
    shank_length_m: float = 0.43

    def __post_init__(self) -> None:
        for name in ("thigh_length_m", "shank_length_m"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite length, got {value!r}")

    @property
    def total_length_m(self) -> float:
        return self.thigh_length_m + self.shank_length_m


# --------------------------------------------------------------------------- #
# Elementary rotations and transforms
# --------------------------------------------------------------------------- #
def rotation_2d(theta: ArrayLike) -> FloatArray:
    """Planar rotation matrix ``R(θ)``; vectorised: shape ``(..., 2, 2)``."""
    theta = np.asarray(theta, dtype=float)
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([np.stack([c, -s], axis=-1), np.stack([s, c], axis=-1)], axis=-2)


def rotation_z(theta: float) -> FloatArray:
    """3×3 rotation about the z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def homogeneous(rotation: ArrayLike, translation: ArrayLike) -> FloatArray:
    """Assemble a 4×4 homogeneous transform from a 3×3 rotation and a 3-vector."""
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).reshape(-1)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3×3, got shape {rotation.shape}")
    if translation.shape != (3,):
        raise ValueError(f"translation must have 3 components, got shape {translation.shape}")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-9
    ):
        raise ValueError("rotation must be a proper orthonormal matrix (R Rᵀ = I, det R = +1)")
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def rot_z_h(theta: float) -> FloatArray:
    """Homogeneous rotation about z."""
    return homogeneous(rotation_z(theta), np.zeros(3))


def trans_x_h(distance: float) -> FloatArray:
    """Homogeneous translation along the local x axis."""
    return homogeneous(np.eye(3), np.array([distance, 0.0, 0.0]))


def invert_homogeneous(transform: ArrayLike) -> FloatArray:
    """Closed-form inverse ``[Rᵀ, −Rᵀp; 0, 1]`` of a rigid transform."""
    transform = np.asarray(transform, dtype=float)
    rotation, translation = transform[:3, :3], transform[:3, 3]
    return homogeneous(rotation.T, -rotation.T @ translation)


# --------------------------------------------------------------------------- #
# Joint-angle conventions
# --------------------------------------------------------------------------- #
def anatomical_to_link_angles(hip_flexion_rad: ArrayLike, knee_flexion_rad: ArrayLike) -> Tuple[FloatArray, FloatArray]:
    """Map anatomical angles to CCW link angles: ``θ1 = hip``, ``θ2 = −knee``."""
    return np.asarray(hip_flexion_rad, dtype=float), -np.asarray(knee_flexion_rad, dtype=float)


def segment_absolute_angles(theta1: ArrayLike, theta2: ArrayLike) -> Tuple[FloatArray, FloatArray]:
    """Absolute segment angles measured CCW from the downward vertical."""
    theta1 = np.asarray(theta1, dtype=float)
    return theta1, theta1 + np.asarray(theta2, dtype=float)


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
def planar_forward_kinematics(
    theta1: ArrayLike, theta2: ArrayLike, geometry: LegGeometry, frame: str = "global"
) -> Tuple[FloatArray, FloatArray]:
    """Knee and ankle positions of the planar two-link leg (vectorised).

    Parameters
    ----------
    theta1, theta2
        CCW link angles in rad (scalars or equally shaped arrays).
    geometry
        Segment lengths.
    frame
        ``"leg"`` returns coordinates in {L} (the textbook formula),
        ``"global"`` returns coordinates in {G}.

    Returns
    -------
    knee_xy, ankle_xy
        Arrays of shape ``(..., 2)`` in metres.
    """
    if frame not in ("global", "leg"):
        raise ValueError(f"frame must be 'global' or 'leg', got {frame!r}")
    theta1 = np.asarray(theta1, dtype=float)
    theta2 = np.asarray(theta2, dtype=float)
    if theta1.shape != theta2.shape:
        raise ValueError(f"theta1 and theta2 must have the same shape, got {theta1.shape} and {theta2.shape}")
    offset = LEG_FRAME_OFFSET_RAD if frame == "global" else 0.0
    thigh_vec = np.array([geometry.thigh_length_m, 0.0])
    shank_vec = np.array([geometry.shank_length_m, 0.0])
    knee = rotation_2d(theta1 + offset) @ thigh_vec
    ankle = knee + rotation_2d(theta1 + theta2 + offset) @ shank_vec
    return knee, ankle


def homogeneous_chain(theta1: float, theta2: float, geometry: LegGeometry) -> List[FloatArray]:
    """Frame transforms ``[ᴳT_thigh, ᴳT_shank, ᴳT_ankle]`` for one configuration.

    ``ᴳT_ankle = R_z(−π/2) · R_z(θ1) · T_x(L1) · R_z(θ2) · T_x(L2)``.
    The thigh frame sits at the hip, the shank frame at the knee, and both have
    their x axis along the segment.
    """
    base = rot_z_h(LEG_FRAME_OFFSET_RAD)
    t_thigh = base @ rot_z_h(float(theta1))
    t_shank = t_thigh @ trans_x_h(geometry.thigh_length_m) @ rot_z_h(float(theta2))
    t_ankle = t_shank @ trans_x_h(geometry.shank_length_m)
    return [t_thigh, t_shank, t_ankle]


def leg_points(
    hip_flexion_rad: ArrayLike, knee_flexion_rad: ArrayLike, geometry: LegGeometry
) -> Tuple[FloatArray, FloatArray, FloatArray]:
    """Global hip, knee and ankle positions for anatomical angles (for plotting)."""
    theta1, theta2 = anatomical_to_link_angles(hip_flexion_rad, knee_flexion_rad)
    knee, ankle = planar_forward_kinematics(theta1, theta2, geometry, frame="global")
    hip = np.zeros_like(knee)
    return hip, knee, ankle
