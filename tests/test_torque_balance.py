"""Suit model, baseline dynamics, actuator limits and the torque-balance sign convention."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import trapezoid

from exosuit_sim.actuator import ActuatorModel
from exosuit_sim.biomechanics import (
    TabulatedTorqueProfile,
    build_inertial_model,
    leg_mechanical_energy,
    load_torque_csv,
    two_link_inverse_dynamics,
)
from exosuit_sim.config import ConfigError, SimulationConfig, config_from_dict, load_config
from exosuit_sim.kinematics import LegGeometry
from exosuit_sim.simulation import ScenarioResult, run_scenario
from exosuit_sim.spacesuit import SuitResistanceModel
from exosuit_sim.trajectory import JointTrajectory, build_trajectory

GEOM = LegGeometry(0.45, 0.43)


# --------------------------------------------------------------------------- #
# Suit resistance
# --------------------------------------------------------------------------- #
def test_zero_suit_stiffness_gives_zero_resistance() -> None:
    suit = SuitResistanceModel(0.0, 0.0, 0.0, 0.0)
    path = np.concatenate([np.linspace(0, 1.5, 100), np.linspace(1.5, 0, 100)])
    np.testing.assert_array_equal(suit.torque(path), 0.0)


def test_quadratic_only_reproduces_k_theta_squared() -> None:
    k = 4.0  # N·m/rad²
    phi = np.linspace(0, np.pi / 2, 50)  # radians
    np.testing.assert_allclose(SuitResistanceModel.quadratic_only(k).torque(phi), k * phi**2, rtol=1e-12)
    # Dimensional check: at 90° (π/2 rad) the torque is k·(π/2)² N·m, not k·90².
    assert SuitResistanceModel.quadratic_only(k).torque(np.array([0.0, np.pi / 2]))[-1] == pytest.approx(
        k * (np.pi / 2) ** 2
    )


def _cycle_work(suit: SuitResistanceModel, up: np.ndarray) -> float:
    path = np.concatenate([up, up[::-1][1:]])
    tau = suit.torque(path)
    return float(np.sum(0.5 * (tau[:-1] + tau[1:]) * np.diff(path)))


def test_suit_resists_flexion_and_has_hysteresis() -> None:
    suit = SuitResistanceModel()
    up = np.linspace(0, np.pi / 2, 401)
    path = np.concatenate([up, up[::-1][1:]])
    tau = suit.torque(path)
    assert np.all(tau[1:401] > 0)  # resists flexion throughout the loading phase
    loading = np.interp(np.radians(45), up, tau[:401])
    unloading = np.interp(np.radians(45), up, tau[400:][::-1])
    assert loading > unloading  # hysteresis: loading branch above unloading branch

    # Loop area = energy dissipated. With an (almost) instantaneous branch switch it
    # equals 2∫(hE + τf)dφ; a finite transition angle rounds the corners and
    # makes the loop strictly smaller, but still positive.
    ideal = 2 * trapezoid(suit.hysteresis_fraction * suit.elastic_torque(up) + suit.friction_Nm, up)
    sharp = SuitResistanceModel(transition_angle_rad=np.radians(0.01), initial_branch_state=1.0)
    assert _cycle_work(sharp, np.linspace(0, np.pi / 2, 20001)) == pytest.approx(ideal, rel=1e-3)
    assert 0 < _cycle_work(suit, up) < ideal
    # Purely elastic suit: no dissipation over a closed cycle
    elastic_only = SuitResistanceModel(friction_Nm=0.0, hysteresis_fraction=0.0)
    assert _cycle_work(elastic_only, up) == pytest.approx(0.0, abs=1e-9)


def test_branch_state_memory_and_discretisation_invariance() -> None:
    suit = SuitResistanceModel()
    hold = np.concatenate([np.linspace(0, 1.0, 50), np.full(30, 1.0)])
    s = suit.branch_state(hold)
    assert np.all(s[50:] == s[49])  # no motion → branch state (and torque) held
    coarse = np.linspace(0, 1.2, 61)
    fine = np.linspace(0, 1.2, 1201)
    np.testing.assert_allclose(suit.torque(coarse), suit.torque(fine)[::20], rtol=1e-10, atol=1e-12)


def test_suit_parameter_validation() -> None:
    with pytest.raises(ValueError):
        SuitResistanceModel(k_linear_Nm_per_rad=-1)
    with pytest.raises(ValueError):
        SuitResistanceModel(hysteresis_fraction=1.0)


# --------------------------------------------------------------------------- #
# Torque balance and sign convention
# --------------------------------------------------------------------------- #
def test_torque_balance_identity(primary_result: ScenarioResult) -> None:
    r = primary_result
    np.testing.assert_allclose(
        r.human_assisted_Nm, r.baseline_knee_torque_Nm + r.suit_torque_Nm - r.actuator_torque_Nm, atol=1e-12
    )
    np.testing.assert_allclose(r.human_unassisted_Nm, r.baseline_knee_torque_Nm + r.suit_torque_Nm, atol=1e-12)
    # Additional human torque equals suit torque minus actuator torque
    np.testing.assert_allclose(r.human_assisted_Nm - r.baseline_knee_torque_Nm, r.suit_torque_Nm - r.actuator_torque_Nm,
                               atol=1e-12)


def test_sign_convention_in_flexion(primary_result: ScenarioResult) -> None:
    r = primary_result
    moving = r.trajectory.knee_vel_rad_s > 1e-3
    assert np.all(r.suit_torque_Nm[moving] > 0)  # suit opposes flexion
    assert np.all(r.actuator_torque_Nm[moving] >= 0)  # assistance acts in the flexion direction
    assert np.all(r.actuator_power_W >= -1e-12)  # actuator drives (never brakes) a pure flexion


def test_ideal_compensation_restores_baseline(primary_result: ScenarioResult) -> None:
    r = primary_result
    human = r.baseline_knee_torque_Nm + r.suit_torque_Nm - r.ideal_profile_Nm
    np.testing.assert_allclose(human, r.baseline_knee_torque_Nm, atol=1e-12)


# --------------------------------------------------------------------------- #
# Baseline inverse dynamics
# --------------------------------------------------------------------------- #
def _model(g: float = 9.81):  # type: ignore[no-untyped-def]
    return build_inertial_model(75.0, GEOM, g)


def test_inverse_dynamics_energy_balance() -> None:
    traj = build_trajectory(0, 90, 0.8, 4001, mode="flexion", hip_start_deg=-10, hip_end_deg=40)
    model = _model()
    hip, knee = two_link_inverse_dynamics(traj, GEOM, model)
    power = hip * traj.hip_vel_rad_s + knee * traj.knee_vel_rad_s
    energy = leg_mechanical_energy(traj, GEOM, model)
    assert trapezoid(power, traj.time_s) == pytest.approx(energy[-1] - energy[0], rel=1e-5, abs=1e-6)


def test_inverse_dynamics_static_and_inertial_limits() -> None:
    model = _model()
    phi = np.radians(60.0)
    n = 5
    static = JointTrajectory(np.linspace(0, 1, n), *(np.zeros(n),) * 3, np.full(n, phi), np.zeros(n), np.zeros(n))
    _, knee = two_link_inverse_dynamics(static, GEOM, model)
    sf = model.shank_foot
    np.testing.assert_allclose(knee, sf.mass_kg * 9.81 * sf.com_from_proximal_m * np.sin(phi), rtol=1e-12)

    space = _model(g=0.0)
    traj = build_trajectory(0, 90, 1.0, 101)
    _, knee0 = two_link_inverse_dynamics(traj, GEOM, space)
    inertia_knee = space.shank_foot.inertia_com_kg_m2 + space.shank_foot.mass_kg * space.shank_foot.com_from_proximal_m**2
    np.testing.assert_allclose(knee0, inertia_knee * traj.knee_acc_rad_s2, atol=1e-12)


def test_tabulated_profile_interpolation_and_bounds(tmp_path: Path) -> None:
    angles = np.radians(np.linspace(0, 90, 19))
    prof = TabulatedTorqueProfile(angles, np.sin(angles))
    query = np.radians(np.linspace(0, 90, 181))
    np.testing.assert_allclose(prof(query), np.sin(query), atol=5e-4)  # PCHIP on 5° spacing
    with pytest.raises(ValueError):
        prof(np.radians(95.0))
    with pytest.raises(ValueError):
        TabulatedTorqueProfile([0.0, 0.0], [1.0, 2.0])
    csv = tmp_path / "t.csv"
    csv.write_text("# synthetic\nknee_flexion_deg,knee_torque_Nm\n0,0\n45,1\n90,2\n")
    loaded = load_torque_csv(csv)
    assert loaded(np.radians(45.0)) == pytest.approx(1.0)


def test_csv_baseline_source_runs(repo_root: Path, default_config: SimulationConfig) -> None:
    cfg = default_config.with_section("baseline_torque", source="csv", csv_path="data/example_synthetic_knee_torque.csv")
    r = run_scenario(cfg, base_dir=repo_root)
    assert "tabulated" in r.baseline_source
    assert np.all(np.isfinite(r.baseline_knee_torque_Nm))


# --------------------------------------------------------------------------- #
# Actuator
# --------------------------------------------------------------------------- #
def test_actuator_force_torque_conversion_and_limits() -> None:
    act = ActuatorModel(max_torque_Nm=20, max_force_N=300, moment_arm_m=0.05, max_torque_rate_Nm_s=None)
    assert act.effective_torque_limit_Nm == pytest.approx(15.0)  # force-limited: 300 N × 0.05 m
    assert act.force_from_torque(10.0) == pytest.approx(200.0)
    assert act.torque_from_force(200.0) == pytest.approx(10.0)
    assert act.torque_bounds() == (-15.0, 15.0)
    one_way = ActuatorModel(bidirectional=False)
    assert one_way.torque_bounds()[0] == 0.0
    violations = act.limit_violations(np.array([0.0, 16.0]), 0.01)
    assert violations["torque_Nm"] == pytest.approx(1.0)
    assert violations["force_N"] == pytest.approx(20.0)


# --------------------------------------------------------------------------- #
# Configuration validation
# --------------------------------------------------------------------------- #
def test_default_config_file_loads(default_config: SimulationConfig) -> None:
    assert default_config.trajectory.knee_end_deg == 90.0
    assert default_config.environment.gravity_m_s2 == pytest.approx(1.62)


@pytest.mark.parametrize(
    "patch",
    [
        {"spacesuit": {"k_quadratic": 4.0}},  # unknown key (typo)
        {"actuator": {"max_torque_Nm": -1.0}},
        {"trajectory": {"knee_end_deg": 200.0}},
        {"trajectory": {"mode": "squat"}},
        {"optimization": {"w_tracking": 0.0, "lambda_effort": 0.0}},
        {"unknown_section": {}},
    ],
)
def test_invalid_configs_raise(patch: dict) -> None:
    with pytest.raises(ConfigError):
        config_from_dict(patch)


def test_load_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigError):
        load_config(bad)
    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"_comment": "ignored", "spacesuit": {"_units": "ignored", "friction_Nm": 0.5}}))
    assert load_config(ok).spacesuit.friction_Nm == 0.5
