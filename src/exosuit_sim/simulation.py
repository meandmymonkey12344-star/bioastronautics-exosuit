"""Scenario pipeline: configuration → kinematics → torques → optimisation → metrics.

This module holds no plotting or file I/O (other than reading an optional
baseline CSV) so that tests, the CLI and the sensitivity study share exactly
the same physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from .actuator import ActuatorModel
from .biomechanics import (
    LegInertialModel,
    build_inertial_model,
    load_torque_csv,
    two_link_inverse_dynamics,
)
from .config import SimulationConfig
from .energy import cumulative_work, performance_metrics
from .kinematics import LegGeometry, leg_points
from .optimization import (
    ObjectiveWeights,
    OptimizationResult,
    QuadraticProblem,
    analytic_box_solution,
    analytic_unconstrained,
    build_problem,
    ideal_compensation,
    optimize_actuator_profile,
    profile_difference,
    projected_gradient_descent,
    random_feasible_profile,
)
from .spacesuit import SuitResistanceModel
from .trajectory import JointTrajectory, build_trajectory

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------- #
# Builders: config sections → physics objects
# --------------------------------------------------------------------------- #
def geometry_from_config(cfg: SimulationConfig) -> LegGeometry:
    a = cfg.anthropometry
    return LegGeometry(thigh_length_m=a.thigh_length_m, shank_length_m=a.shank_length_m)


def inertial_model_from_config(cfg: SimulationConfig, geometry: LegGeometry) -> LegInertialModel:
    a = cfg.anthropometry
    return build_inertial_model(
        body_mass_kg=a.body_mass_kg,
        geometry=geometry,
        gravity_m_s2=cfg.environment.gravity_m_s2,
        thigh_fractions=(a.thigh_mass_fraction, a.thigh_com_fraction, a.thigh_radius_of_gyration_fraction),
        shank_fractions=(a.shank_mass_fraction, a.shank_com_fraction, a.shank_radius_of_gyration_fraction),
        foot_mass_fraction=a.foot_mass_fraction,
    )


def suit_from_config(cfg: SimulationConfig) -> SuitResistanceModel:
    s = cfg.spacesuit
    return SuitResistanceModel(
        k_linear_Nm_per_rad=s.k_linear_Nm_per_rad,
        k_quadratic_Nm_per_rad2=s.k_quadratic_Nm_per_rad2,
        friction_Nm=s.friction_Nm,
        hysteresis_fraction=s.hysteresis_fraction,
        transition_angle_rad=float(np.radians(s.branch_transition_deg)),
        initial_branch_state=s.initial_branch_state,
    )


def actuator_from_config(cfg: SimulationConfig) -> ActuatorModel:
    a = cfg.actuator
    return ActuatorModel(
        max_torque_Nm=a.max_torque_Nm,
        max_force_N=a.max_force_N,
        moment_arm_m=a.moment_arm_m,
        max_torque_rate_Nm_s=a.max_torque_rate_Nm_s,
        efficiency=a.efficiency,
        regeneration_efficiency=a.regeneration_efficiency,
        bidirectional=a.bidirectional,
    )


def weights_from_config(cfg: SimulationConfig) -> ObjectiveWeights:
    o = cfg.optimization
    return ObjectiveWeights(w_tracking=o.w_tracking, lambda_effort=o.lambda_effort, mu_rate=o.mu_rate)


def trajectory_from_config(cfg: SimulationConfig, mode: Optional[str] = None) -> JointTrajectory:
    t = cfg.trajectory
    return build_trajectory(
        knee_start_deg=t.knee_start_deg,
        knee_end_deg=t.knee_end_deg,
        duration_s=t.duration_s,
        n_samples=t.n_samples,
        mode=mode or t.mode,
        hip_start_deg=t.hip_start_deg,
        hip_end_deg=t.hip_end_deg,
    )


# --------------------------------------------------------------------------- #
# Scenario
# --------------------------------------------------------------------------- #
@dataclass
class ScenarioResult:
    """Everything computed for one movement scenario."""

    label: str
    config: SimulationConfig
    geometry: LegGeometry
    inertial_model: LegInertialModel
    suit_model: SuitResistanceModel
    actuator: ActuatorModel
    trajectory: JointTrajectory
    knee_xy: FloatArray
    ankle_xy: FloatArray
    baseline_source: str
    baseline_knee_torque_Nm: FloatArray
    hip_torque_Nm: Optional[FloatArray]
    suit_branch_state: FloatArray
    suit_torque_Nm: FloatArray
    problem: QuadraticProblem
    ideal_profile_Nm: FloatArray
    analytic_unconstrained_Nm: FloatArray
    analytic_box_Nm: Optional[FloatArray]
    optimized: OptimizationResult
    human_unassisted_Nm: FloatArray
    human_assisted_Nm: FloatArray
    actuator_power_W: FloatArray
    actuator_cumulative_work_J: FloatArray
    metrics: Dict[str, float]
    reference_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def actuator_torque_Nm(self) -> FloatArray:
        return self.optimized.torque_Nm

    @property
    def actuator_force_N(self) -> FloatArray:
        return self.actuator.force_from_torque(self.optimized.torque_Nm)


def _baseline_torque(
    cfg: SimulationConfig,
    trajectory: JointTrajectory,
    geometry: LegGeometry,
    model: LegInertialModel,
    base_dir: Optional[Path],
) -> "tuple[FloatArray, Optional[FloatArray], str]":
    if cfg.baseline_torque.source == "inverse_dynamics":
        hip, knee = two_link_inverse_dynamics(trajectory, geometry, model)
        return knee, hip, "synthetic two-link inverse dynamics"
    path = Path(cfg.baseline_torque.csv_path or "")
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    profile = load_torque_csv(path)
    return profile(trajectory.knee_rad), None, f"tabulated data ({profile.label}, PCHIP)"


def run_scenario(
    cfg: SimulationConfig,
    mode: Optional[str] = None,
    label: str = "primary",
    base_dir: Optional[Path] = None,
) -> ScenarioResult:
    """Run the complete physics + optimisation pipeline for one movement."""
    geometry = geometry_from_config(cfg)
    model = inertial_model_from_config(cfg, geometry)
    suit_model = suit_from_config(cfg)
    actuator = actuator_from_config(cfg)
    trajectory = trajectory_from_config(cfg, mode)

    _, knee_xy, ankle_xy = leg_points(trajectory.hip_rad, trajectory.knee_rad, geometry)
    baseline, hip_torque, baseline_source = _baseline_torque(cfg, trajectory, geometry, model, base_dir)
    branch = suit_model.branch_state(trajectory.knee_rad)
    suit = suit_model.torque(trajectory.knee_rad)

    problem = build_problem(suit, trajectory.dt, weights_from_config(cfg), cfg.optimization.torque_scale_Nm)
    lb, ub = actuator.torque_bounds()
    analytic_box = analytic_box_solution(problem, lb, ub) if problem.weights.mu_rate == 0 else None

    optimized = optimize_actuator_profile(
        problem,
        actuator,
        enforce_rate_limit=cfg.optimization.enforce_rate_limit,
        method=cfg.optimization.method,
        max_iter=cfg.optimization.max_iter,
        ftol=cfg.optimization.ftol,
    )
    act = optimized.torque_Nm

    def _metrics(profile: FloatArray) -> Dict[str, float]:
        m = performance_metrics(
            trajectory.time_s, trajectory.knee_rad, trajectory.knee_vel_rad_s, suit, profile, baseline, actuator
        )
        m["objective_J"] = problem.objective(profile)
        return m

    metrics = _metrics(act)
    metrics.update({f"objective_term_{k}": v for k, v in problem.terms(act).items()})
    metrics["effective_torque_limit_Nm"] = actuator.effective_torque_limit_Nm
    reference = {"ideal_compensation_unconstrained": _metrics(ideal_compensation(problem))}
    if analytic_box is not None:
        reference["analytic_box_constrained"] = _metrics(analytic_box)

    return ScenarioResult(
        label=label,
        config=cfg,
        geometry=geometry,
        inertial_model=model,
        suit_model=suit_model,
        actuator=actuator,
        trajectory=trajectory,
        knee_xy=knee_xy,
        ankle_xy=ankle_xy,
        baseline_source=baseline_source,
        baseline_knee_torque_Nm=baseline,
        hip_torque_Nm=hip_torque,
        suit_branch_state=branch,
        suit_torque_Nm=suit,
        problem=problem,
        ideal_profile_Nm=ideal_compensation(problem),
        analytic_unconstrained_Nm=analytic_unconstrained(problem),
        analytic_box_Nm=analytic_box,
        optimized=optimized,
        human_unassisted_Nm=baseline + suit,
        human_assisted_Nm=baseline + suit - act,
        actuator_power_W=act * trajectory.knee_vel_rad_s,
        actuator_cumulative_work_J=cumulative_work(act, trajectory.knee_rad),
        metrics=metrics,
        reference_metrics=reference,
    )


# --------------------------------------------------------------------------- #
# Validation of the optimiser against closed forms and feasible alternatives
# --------------------------------------------------------------------------- #
@dataclass
class SolverComparison:
    """Box-constrained problem (no rate limit) solved several ways."""

    exact_Nm: FloatArray
    exact_objective: float
    results: Dict[str, OptimizationResult]


def compare_solvers(result: ScenarioResult, mu_rate: float = 1e-3) -> SolverComparison:
    """Solve a bounds-only comparison problem with SLSQP, L-BFGS-B and projected gradient descent.

    With μ = 0 the problem is separable (diagonal Hessian) and every method
    converges in one or two iterations, which teaches nothing. The comparison
    therefore adds a small smoothness weight ``mu_rate`` that couples
    neighbouring samples and makes the Hessian ill-conditioned. No closed form
    exists for that case, so the reference J* is the lowest objective reached by
    any method (all three agree to near machine precision when converged).
    """
    base = result.problem
    weights = ObjectiveWeights(base.weights.w_tracking, base.weights.lambda_effort, mu_rate)
    problem = build_problem(base.suit_torque_Nm, base.dt_s, weights, base.torque_scale_Nm)
    actuator = result.actuator
    lb, ub = actuator.torque_bounds()
    runs = {
        "SLSQP": optimize_actuator_profile(problem, actuator, enforce_rate_limit=False, method="SLSQP"),
        "L-BFGS-B": optimize_actuator_profile(problem, actuator, enforce_rate_limit=False, method="L-BFGS-B"),
        "Projected gradient": projected_gradient_descent(problem, lb, ub, max_iter=50000, tol=1e-13),
    }
    best = min(runs.values(), key=lambda r: r.objective)
    return SolverComparison(exact_Nm=best.torque_Nm, exact_objective=best.objective, results=runs)


def validation_checks(result: ScenarioResult, n_random: int = 200) -> Dict[str, Any]:
    """Numerical sanity checks reported alongside every run (not a substitute for the test suite)."""
    problem, actuator, opt = result.problem, result.actuator, result.optimized
    rng = np.random.default_rng(result.config.random_seed)
    use_rate = opt.rate_limit_enforced
    lb, ub = actuator.torque_bounds()
    dt = result.trajectory.dt

    random_objectives = [
        problem.objective(random_feasible_profile(actuator, problem.n, dt, rng, enforce_rate_limit=use_rate))
        for _ in range(n_random)
    ]
    clipped_ideal = np.clip(result.ideal_profile_Nm, lb, ub)
    checks: Dict[str, Any] = {
        "objective_optimized": opt.objective,
        "objective_zero_assistance": problem.objective(np.zeros(problem.n)),
        "objective_best_random_feasible": float(np.min(random_objectives)),
        "n_random_feasible_profiles": n_random,
        "optimized_not_worse_than_zero_assistance": opt.objective <= problem.objective(np.zeros(problem.n)) + 1e-12,
        "optimized_not_worse_than_best_random": opt.objective <= float(np.min(random_objectives)) + 1e-12,
        "optimized_profile_feasible": actuator.is_feasible(opt.torque_Nm, dt) if use_rate
        else actuator.limit_violations(opt.torque_Nm, dt)["torque_Nm"] <= 1e-9,
        "max_limit_violations": opt.max_violations,
    }
    if result.analytic_box_Nm is not None:
        # Dropping the rate limit relaxes the problem, so its exact optimum is a lower bound.
        relaxed = problem.objective(result.analytic_box_Nm)
        checks["objective_rate_relaxed_lower_bound"] = relaxed
        checks["optimized_not_better_than_relaxation"] = opt.objective >= relaxed - 1e-12
    if actuator.is_feasible(clipped_ideal, dt) or not use_rate:
        checks["objective_clipped_ideal"] = problem.objective(clipped_ideal)
    else:
        checks["clipped_ideal_note"] = "clipped ideal profile violates the torque-rate limit (infeasible)"

    unbounded = ActuatorModel(
        max_torque_Nm=1e6, max_force_N=1e9, moment_arm_m=1.0, max_torque_rate_Nm_s=None, efficiency=actuator.efficiency
    )
    numeric_free = optimize_actuator_profile(problem, unbounded, enforce_rate_limit=False)
    max_diff, rms_diff = profile_difference(numeric_free.torque_Nm, result.analytic_unconstrained_Nm)
    checks["unconstrained_numeric_vs_analytic_max_abs_diff_Nm"] = max_diff
    checks["unconstrained_numeric_vs_analytic_rms_diff_Nm"] = rms_diff
    if result.analytic_box_Nm is not None:
        box_numeric = optimize_actuator_profile(problem, actuator, enforce_rate_limit=False)
        checks["box_numeric_vs_analytic_max_abs_diff_Nm"] = profile_difference(
            box_numeric.torque_Nm, result.analytic_box_Nm
        )[0]
    return checks
