"""Optimisation: objective properties, closed forms, constraints and solver agreement."""

from __future__ import annotations

import numpy as np
import pytest

from exosuit_sim.actuator import ActuatorModel
from exosuit_sim.optimization import (
    ObjectiveWeights,
    analytic_box_solution,
    analytic_unconstrained,
    build_problem,
    optimize_actuator_profile,
    projected_gradient_descent,
    random_feasible_profile,
)
from exosuit_sim.simulation import ScenarioResult, validation_checks
from exosuit_sim.spacesuit import SuitResistanceModel
from exosuit_sim.trajectory import build_trajectory

TRAJ = build_trajectory(0, 90, 1.0, 121)
SUIT = SuitResistanceModel().torque(TRAJ.knee_rad)
DT = TRAJ.dt
UNLIMITED = ActuatorModel(max_torque_Nm=1e6, max_force_N=1e9, moment_arm_m=1.0, max_torque_rate_Nm_s=None)
RNG = np.random.default_rng(7)


def _problem(lam: float = 0.01, mu: float = 0.0):  # type: ignore[no-untyped-def]
    return build_problem(SUIT, DT, ObjectiveWeights(1.0, lam, mu))


@pytest.mark.parametrize("mu", [0.0, 1e-3])
def test_objective_is_finite_nonnegative_and_matches_quadratic_form(mu: float) -> None:
    p = _problem(mu=mu)
    for _ in range(50):
        tau = RNG.normal(0, 30, p.n)
        j = p.objective(tau)
        assert np.isfinite(j) and j >= 0
        quad = 0.5 * np.einsum("i,ij,j->", tau, p.hessian, tau) + np.sum(p.linear * tau) + p.constant
        assert j == pytest.approx(quad, rel=1e-9, abs=1e-12)


def test_gradient_matches_finite_differences() -> None:
    p = _problem(mu=1e-3)
    tau = RNG.normal(0, 10, p.n)
    grad = p.gradient(tau)
    # Central differences are exact for a quadratic (no truncation error), so a large
    # step avoids the round-off cancellation that a tiny step would introduce.
    h = 0.5
    for i in RNG.choice(p.n, 10, replace=False):
        e = np.zeros(p.n)
        e[i] = h
        fd = (p.objective(tau + e) - p.objective(tau - e)) / (2 * h)
        assert grad[i] == pytest.approx(fd, rel=1e-8, abs=1e-15)


@pytest.mark.parametrize("lam", [0.0, 0.01, 1.0, 10.0])
def test_closed_form_shrinkage(lam: float) -> None:
    p = _problem(lam=lam)
    np.testing.assert_allclose(analytic_unconstrained(p), SUIT / (1 + lam), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("lam, mu", [(0.01, 0.0), (0.3, 0.0), (0.01, 1e-3)])
def test_numerical_matches_analytic_unconstrained(lam: float, mu: float) -> None:
    p = _problem(lam=lam, mu=mu)
    res = optimize_actuator_profile(p, UNLIMITED, enforce_rate_limit=False)
    assert res.success
    # μ > 0 couples samples and raises the condition number (~1e2), so SLSQP's
    # stopping tolerance leaves ~1e-6 N·m error; 1e-5 N·m is < 1e-6 of the peak torque.
    np.testing.assert_allclose(res.torque_Nm, analytic_unconstrained(p), atol=1e-6 if mu == 0 else 1e-5)


def test_lambda_zero_without_limits_gives_exact_compensation() -> None:
    res = optimize_actuator_profile(_problem(lam=0.0), UNLIMITED, enforce_rate_limit=False)
    np.testing.assert_allclose(res.torque_Nm, SUIT, atol=1e-6)


@pytest.mark.parametrize("method", ["SLSQP", "L-BFGS-B"])
def test_numerical_matches_clipped_analytic_with_bounds(method: str) -> None:
    act = ActuatorModel(max_torque_Nm=15.0, max_torque_rate_Nm_s=None)
    p = _problem()
    lb, ub = act.torque_bounds()
    res = optimize_actuator_profile(p, act, enforce_rate_limit=False, method=method)
    np.testing.assert_allclose(res.torque_Nm, analytic_box_solution(p, lb, ub), atol=1e-6)
    pgd = projected_gradient_descent(p, lb, ub)
    np.testing.assert_allclose(pgd.torque_Nm, analytic_box_solution(p, lb, ub), atol=1e-8)


ACTUATORS = [
    ActuatorModel(max_torque_Nm=20, max_force_N=600, moment_arm_m=0.05, max_torque_rate_Nm_s=40),
    ActuatorModel(max_torque_Nm=30, max_force_N=200, moment_arm_m=0.05, max_torque_rate_Nm_s=25),  # force-limited
    ActuatorModel(max_torque_Nm=8, max_force_N=600, moment_arm_m=0.05, max_torque_rate_Nm_s=None),
    ActuatorModel(max_torque_Nm=12, max_torque_rate_Nm_s=15, bidirectional=False),
]


@pytest.mark.parametrize("act", ACTUATORS)
def test_optimised_profile_never_exceeds_limits(act: ActuatorModel) -> None:
    res = optimize_actuator_profile(_problem(), act)
    assert res.success, res.message
    lb, ub = act.torque_bounds()
    assert np.all(res.torque_Nm <= ub) and np.all(res.torque_Nm >= lb)
    assert np.max(np.abs(act.force_from_torque(res.torque_Nm))) <= act.max_force_N + 1e-9
    if act.max_torque_rate_Nm_s is not None:
        assert np.max(np.abs(np.diff(res.torque_Nm))) / DT <= act.max_torque_rate_Nm_s + 1e-6
    assert act.is_feasible(res.torque_Nm, DT)


@pytest.mark.parametrize("act", ACTUATORS)
def test_optimum_beats_feasible_alternatives(act: ActuatorModel) -> None:
    p = _problem()
    res = optimize_actuator_profile(p, act)
    for _ in range(100):
        alt = random_feasible_profile(act, p.n, DT, RNG)
        assert act.is_feasible(alt, DT)
        assert res.objective <= p.objective(alt) + 1e-12
    assert res.objective <= p.objective(np.zeros(p.n)) + 1e-12
    lb, ub = act.torque_bounds()
    clipped = np.clip(SUIT, lb, ub)
    if act.is_feasible(clipped, DT):
        assert res.objective <= p.objective(clipped) + 1e-12


def test_rate_limit_is_a_restriction_of_box_problem() -> None:
    p = _problem()
    act = ActuatorModel(max_torque_Nm=20, max_torque_rate_Nm_s=30)
    lb, ub = act.torque_bounds()
    constrained = optimize_actuator_profile(p, act)
    assert constrained.objective >= p.objective(analytic_box_solution(p, lb, ub)) - 1e-12


def test_more_actuator_capacity_never_hurts() -> None:
    p = _problem()
    objectives = [
        optimize_actuator_profile(p, ActuatorModel(max_torque_Nm=t, max_force_N=1e6, max_torque_rate_Nm_s=40)).objective
        for t in (2, 5, 10, 15, 20, 30, 50)
    ]
    assert all(b <= a + 1e-10 for a, b in zip(objectives, objectives[1:]))
    rates = [
        optimize_actuator_profile(p, ActuatorModel(max_torque_Nm=30, max_torque_rate_Nm_s=r)).objective
        for r in (5, 10, 20, 40, 80)
    ]
    assert all(b <= a + 1e-10 for a, b in zip(rates, rates[1:]))


def test_larger_effort_penalty_reduces_assistance() -> None:
    act = ActuatorModel(max_torque_Nm=50, max_torque_rate_Nm_s=None)
    rms = [
        np.sqrt(np.mean(optimize_actuator_profile(_problem(lam=lam), act, enforce_rate_limit=False).torque_Nm ** 2))
        for lam in (0.0, 0.1, 1.0, 10.0)
    ]
    assert all(b < a for a, b in zip(rms, rms[1:]))


def test_history_and_status_are_recorded() -> None:
    res = optimize_actuator_profile(_problem(), ACTUATORS[0])
    assert len(res.history) >= 2 and res.history[-1] <= res.history[0]
    assert res.n_iterations > 0 and isinstance(res.message, str)
    assert res.rate_limit_enforced


def test_invalid_requests_raise() -> None:
    with pytest.raises(ValueError):
        optimize_actuator_profile(_problem(), ACTUATORS[0], method="L-BFGS-B")  # cannot enforce rate limits
    with pytest.raises(ValueError):
        ObjectiveWeights(0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        analytic_box_solution(_problem(mu=1e-3), -1, 1)


def test_validation_checks_on_default_scenario(primary_result: ScenarioResult) -> None:
    checks = validation_checks(primary_result, n_random=50)
    assert checks["optimized_not_worse_than_zero_assistance"]
    assert checks["optimized_not_worse_than_best_random"]
    assert checks["optimized_profile_feasible"]
    assert checks["optimized_not_better_than_relaxation"]
    assert checks["unconstrained_numeric_vs_analytic_max_abs_diff_Nm"] < 1e-6
    assert checks["box_numeric_vs_analytic_max_abs_diff_Nm"] < 1e-5
