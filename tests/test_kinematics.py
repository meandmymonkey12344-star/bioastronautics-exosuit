"""Kinematics: rotations, homogeneous transforms and forward kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from exosuit_sim.kinematics import (
    LegGeometry,
    anatomical_to_link_angles,
    homogeneous,
    homogeneous_chain,
    invert_homogeneous,
    leg_points,
    planar_forward_kinematics,
    rotation_2d,
    rotation_z,
)

GEOM = LegGeometry(0.45, 0.43)
RNG = np.random.default_rng(1)


def test_rotation_2d_preserves_vector_length() -> None:
    angles = RNG.uniform(-2 * np.pi, 2 * np.pi, 500)
    vectors = RNG.normal(size=(500, 2))
    rotated = np.einsum("nij,nj->ni", rotation_2d(angles), vectors)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(vectors, axis=1), rtol=1e-12)


def test_rotation_2d_is_proper_orthonormal_and_composes() -> None:
    for a, b in RNG.uniform(-np.pi, np.pi, (50, 2)):
        r = rotation_2d(a)
        np.testing.assert_allclose(r @ r.T, np.eye(2), atol=1e-12)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(rotation_2d(a) @ rotation_2d(b), rotation_2d(a + b), atol=1e-12)


def test_rotation_2d_ccw_convention() -> None:
    np.testing.assert_allclose(rotation_2d(np.pi / 2) @ np.array([1.0, 0.0]), [0.0, 1.0], atol=1e-12)


def test_rotation_z_preserves_length_and_axis() -> None:
    v = RNG.normal(size=3)
    r = rotation_z(0.7)
    assert np.linalg.norm(r @ v) == pytest.approx(np.linalg.norm(v), rel=1e-12)
    np.testing.assert_allclose(r @ np.array([0, 0, 1.0]), [0, 0, 1.0])


@pytest.mark.parametrize(
    "hip_deg, knee_deg, expected",
    [
        (0.0, 0.0, (0.0, -0.88)),  # standing: leg hangs straight down
        (90.0, 0.0, (0.88, 0.0)),  # hip flexed 90°, knee straight: leg horizontal forward
        (0.0, 90.0, (-0.43, -0.45)),  # thigh vertical, shank folded backward
        (90.0, 90.0, (0.45, -0.43)),  # seated posture
    ],
)
def test_ankle_position_known_configurations(hip_deg: float, knee_deg: float, expected: tuple) -> None:
    th1, th2 = anatomical_to_link_angles(np.radians(hip_deg), np.radians(knee_deg))
    knee, ankle = planar_forward_kinematics(th1, th2, GEOM, frame="global")
    np.testing.assert_allclose(ankle, expected, atol=1e-12)


def test_textbook_formula_in_leg_frame() -> None:
    # P = R(θ1)[L1,0] + R(θ1+θ2)[L2,0] holds exactly in the leg frame.
    th1, th2 = 0.3, -0.8
    _, ankle = planar_forward_kinematics(th1, th2, GEOM, frame="leg")
    expected = rotation_2d(th1) @ [GEOM.thigh_length_m, 0] + rotation_2d(th1 + th2) @ [GEOM.shank_length_m, 0]
    np.testing.assert_allclose(ankle, expected, atol=1e-12)
    _, zero = planar_forward_kinematics(0.0, 0.0, GEOM, frame="leg")
    np.testing.assert_allclose(zero, [0.88, 0.0], atol=1e-12)


def test_segment_lengths_are_preserved_for_random_configurations() -> None:
    hip = RNG.uniform(-0.5, 1.8, 200)
    knee = RNG.uniform(0.0, 2.4, 200)
    _, k, a = leg_points(hip, knee, GEOM)
    np.testing.assert_allclose(np.linalg.norm(k, axis=1), GEOM.thigh_length_m, rtol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(a - k, axis=1), GEOM.shank_length_m, rtol=1e-12)


def test_knee_flexion_moves_ankle_posteriorly_from_standing() -> None:
    _, _, ankle = leg_points(0.0, np.radians(10.0), GEOM)
    assert ankle[0] < 0  # anatomical flexion → ankle behind the knee


def test_homogeneous_chain_matches_planar_kinematics() -> None:
    for th1, th2 in RNG.uniform(-2.0, 2.0, (100, 2)):
        transforms = homogeneous_chain(th1, th2, GEOM)
        knee, ankle = planar_forward_kinematics(th1, th2, GEOM)
        np.testing.assert_allclose(transforms[1][:2, 3], knee, atol=1e-12)
        np.testing.assert_allclose(transforms[2][:2, 3], ankle, atol=1e-12)
        assert transforms[2][2, 3] == pytest.approx(0.0, abs=1e-15)  # planar motion stays in z = 0
        for t in transforms:
            np.testing.assert_allclose(t[:3, :3] @ t[:3, :3].T, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(t[3], [0, 0, 0, 1])


def test_vectorised_and_scalar_kinematics_agree() -> None:
    th1, th2 = RNG.uniform(-1, 1, 20), RNG.uniform(-2, 0, 20)
    knees, ankles = planar_forward_kinematics(th1, th2, GEOM)
    for i in range(20):
        k, a = planar_forward_kinematics(th1[i], th2[i], GEOM)
        np.testing.assert_allclose(knees[i], k, atol=1e-14)
        np.testing.assert_allclose(ankles[i], a, atol=1e-14)


def test_invert_homogeneous() -> None:
    t = homogeneous(rotation_z(0.9), [0.1, -0.2, 0.3])
    np.testing.assert_allclose(t @ invert_homogeneous(t), np.eye(4), atol=1e-12)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        LegGeometry(-0.45, 0.43)
    with pytest.raises(ValueError):
        homogeneous(np.diag([1.0, 1.0, 2.0]), [0, 0, 0])
    with pytest.raises(ValueError):
        planar_forward_kinematics(np.zeros(3), np.zeros(4), GEOM)
    with pytest.raises(ValueError):
        planar_forward_kinematics(0.0, 0.0, GEOM, frame="world")
