"""Work and energy calculations on cases with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from exosuit_sim.energy import (
    cumulative_work,
    electrical_energy,
    mechanical_work,
    time_mean_abs,
    time_rms,
    work_increments,
)
from exosuit_sim.trajectory import build_trajectory


def test_constant_torque_work_is_exact_on_nonuniform_angle_steps() -> None:
    traj = build_trajectory(0, 90, 1.0, 101)  # minimum-jerk: angle steps are non-uniform
    tau0 = 12.5
    w = mechanical_work(np.full(traj.n_samples, tau0), traj.knee_rad)
    assert w.net_J == pytest.approx(tau0 * np.pi / 2, rel=1e-12)
    assert w.positive_J == pytest.approx(w.net_J) and w.negative_J == 0.0
    assert cumulative_work(np.full(traj.n_samples, tau0), traj.knee_rad)[-1] == pytest.approx(w.net_J)


def test_constant_torque_against_motion_is_negative_work() -> None:
    traj = build_trajectory(0, 90, 1.0, 51)
    w = mechanical_work(np.full(traj.n_samples, -3.0), traj.knee_rad)
    assert w.net_J == pytest.approx(-3.0 * np.pi / 2) and w.positive_J == 0.0


def test_constant_torque_over_closed_cycle_has_zero_net_work() -> None:
    traj = build_trajectory(0, 90, 1.0, 101, mode="flexion_extension")
    w = mechanical_work(np.full(traj.n_samples, 5.0), traj.knee_rad)
    assert w.net_J == pytest.approx(0.0, abs=1e-12)
    assert w.positive_J == pytest.approx(-w.negative_J)
    assert w.positive_J == pytest.approx(5.0 * np.pi / 2)


def test_work_split_is_consistent_for_arbitrary_profiles() -> None:
    rng = np.random.default_rng(3)
    phi = np.cumsum(rng.normal(0, 0.02, 300))
    tau = rng.normal(0, 10, 300)
    w = mechanical_work(tau, phi)
    assert w.positive_J + w.negative_J == pytest.approx(w.net_J, abs=1e-12)
    assert w.net_J == pytest.approx(np.sum(work_increments(tau, phi)))


def test_linear_torque_work_matches_integral() -> None:
    phi = np.linspace(0, 1.0, 11)
    k = 7.0
    assert mechanical_work(k * phi, phi).net_J == pytest.approx(k / 2, rel=1e-12)  # trapezoid exact for linear τ


def test_electrical_energy_with_efficiency_and_regeneration() -> None:
    assert electrical_energy(10.0, 0.0, 0.5) == pytest.approx(20.0)
    assert electrical_energy(10.0, -4.0, 0.8, 0.5) == pytest.approx(10.0 / 0.8 - 2.0)
    with pytest.raises(ValueError):
        electrical_energy(10.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        electrical_energy(-1.0, 0.0, 0.5)


def test_time_statistics() -> None:
    t = np.linspace(0, 2, 201)
    assert time_rms(np.full_like(t, -3.0), t) == pytest.approx(3.0)
    assert time_mean_abs(np.full_like(t, -3.0), t) == pytest.approx(3.0)
    assert time_rms(np.sin(2 * np.pi * t), t) == pytest.approx(1 / np.sqrt(2), rel=1e-3)


def test_mismatched_inputs_raise() -> None:
    with pytest.raises(ValueError):
        work_increments(np.zeros(3), np.zeros(4))
