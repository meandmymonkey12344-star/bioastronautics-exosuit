"""Actuator torque-profile optimisation (docs/MODEL.md §6).

Objective (dimensionless; weights only matter through their ratios)::

    J(τa) = (1/N) [ w  Σ ((τsuit,i − τa,i)/τref)²
                  + λ  Σ (τa,i/τref)²
                  + μ  Σ ((τa,i+1 − τa,i)/(Δt·τ̇ref))² ]

with τ̇ref = τref / T. Because τh,suited − τh,baseline = τsuit − τa, the baseline
human torque cancels from J.

J is a convex quadratic ``½ τᵀHτ + gᵀτ + c``. Closed forms exist for the
unconstrained problem and for box bounds when μ = 0; rate limits couple
neighbouring samples and need a numerical QP solver (SLSQP here).
"""

from __future__ import annotations

import time
import warnings as _warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, LinearConstraint, minimize

from .actuator import ActuatorModel

FloatArray = NDArray[np.float64]


def _matmul(a: FloatArray, b: FloatArray) -> FloatArray:
    """Matrix product with a finiteness check.

    NumPy 2.0.x linked against Apple Accelerate raises spurious divide/overflow
    floating-point warnings from ``@`` even when every operand and the result are
    finite. We silence those flags here and instead verify the result explicitly,
    so a genuine overflow still raises.
    """
    with np.errstate(all="ignore"):
        out = a @ b
    if not np.all(np.isfinite(out)):
        raise FloatingPointError("Non-finite value produced in matrix product")
    return out


@dataclass(frozen=True)
class ObjectiveWeights:
    """Dimensionless weights of the tracking, effort and smoothness terms."""

    w_tracking: float = 1.0
    lambda_effort: float = 0.01
    mu_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("w_tracking", "lambda_effort", "mu_rate"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        if self.w_tracking + self.lambda_effort <= 0:
            raise ValueError("w_tracking + lambda_effort must be > 0 for a strictly convex problem")


def difference_matrix(n: int, dt_s: float) -> FloatArray:
    """Forward-difference operator D ((n−1)×n) so that (Dτ)ᵢ = (τᵢ₊₁ − τᵢ)/Δt."""
    if n < 2:
        raise ValueError("Need at least two samples for a difference operator")
    d = np.zeros((n - 1, n))
    idx = np.arange(n - 1)
    d[idx, idx] = -1.0
    d[idx, idx + 1] = 1.0
    return d / dt_s


@dataclass(frozen=True)
class QuadraticProblem:
    """The discretised actuator-profile problem ``J = ½ τᵀHτ + gᵀτ + c``."""

    suit_torque_Nm: FloatArray
    dt_s: float
    weights: ObjectiveWeights
    torque_scale_Nm: float
    rate_scale_Nm_s: float
    hessian: FloatArray
    linear: FloatArray
    constant: float

    @property
    def n(self) -> int:
        return int(self.suit_torque_Nm.size)

    def terms(self, torque_Nm: ArrayLike) -> Dict[str, float]:
        """Individual (weighted, normalised) objective terms; each is ≥ 0."""
        tau = np.asarray(torque_Nm, dtype=float).reshape(-1)
        if tau.size != self.n:
            raise ValueError(f"torque profile has {tau.size} samples, expected {self.n}")
        n, w = self.n, self.weights
        tracking = w.w_tracking * np.sum(((self.suit_torque_Nm - tau) / self.torque_scale_Nm) ** 2) / n
        effort = w.lambda_effort * np.sum((tau / self.torque_scale_Nm) ** 2) / n
        rate = np.diff(tau) / self.dt_s
        smooth = w.mu_rate * np.sum((rate / self.rate_scale_Nm_s) ** 2) / n
        return {"tracking": float(tracking), "effort": float(effort), "smoothness": float(smooth)}

    def objective(self, torque_Nm: ArrayLike) -> float:
        """J evaluated as a sum of squares (numerically non-negative by construction)."""
        return float(sum(self.terms(torque_Nm).values()))

    def gradient(self, torque_Nm: ArrayLike) -> FloatArray:
        """∇J = Hτ + g."""
        tau = np.asarray(torque_Nm, dtype=float).reshape(-1)
        return _matmul(self.hessian, tau) + self.linear


def build_problem(
    suit_torque_Nm: ArrayLike,
    dt_s: float,
    weights: ObjectiveWeights,
    torque_scale_Nm: Optional[float] = None,
) -> QuadraticProblem:
    """Assemble H, g, c for a sampled suit-torque profile.

    ``torque_scale_Nm`` defaults to max|τsuit| (or 1 N·m if the suit torque is zero).
    """
    suit = np.asarray(suit_torque_Nm, dtype=float).reshape(-1)
    if suit.size < 3:
        raise ValueError("At least 3 samples are required")
    if not np.all(np.isfinite(suit)):
        raise ValueError("suit_torque_Nm must be finite")
    if not np.isfinite(dt_s) or dt_s <= 0:
        raise ValueError(f"dt_s must be > 0, got {dt_s!r}")
    if torque_scale_Nm is None:
        peak = float(np.max(np.abs(suit)))
        torque_scale_Nm = peak if peak > 0 else 1.0
    if torque_scale_Nm <= 0:
        raise ValueError("torque_scale_Nm must be > 0")
    n = suit.size
    duration = dt_s * (n - 1)
    rate_scale = torque_scale_Nm / duration
    d = difference_matrix(n, dt_s)
    w = weights
    hessian = (2.0 / n) * (
        (w.w_tracking + w.lambda_effort) / torque_scale_Nm**2 * np.eye(n) + w.mu_rate / rate_scale**2 * _matmul(d.T, d)
    )
    linear = -(2.0 / n) * w.w_tracking / torque_scale_Nm**2 * suit
    constant = w.w_tracking * float(np.sum(suit**2)) / (n * torque_scale_Nm**2)
    return QuadraticProblem(
        suit_torque_Nm=suit,
        dt_s=float(dt_s),
        weights=weights,
        torque_scale_Nm=float(torque_scale_Nm),
        rate_scale_Nm_s=float(rate_scale),
        hessian=hessian,
        linear=linear,
        constant=constant,
    )


# --------------------------------------------------------------------------- #
# Closed-form references
# --------------------------------------------------------------------------- #
def ideal_compensation(problem: QuadraticProblem) -> FloatArray:
    """Reference profile τa = τsuit (exact cancellation; optimal only if λ = μ = 0, no limits)."""
    return problem.suit_torque_Nm.copy()


def analytic_unconstrained(problem: QuadraticProblem) -> FloatArray:
    """Exact unconstrained minimiser, solving Hτ = −g.

    For μ = 0 this equals ``w/(w+λ) · τsuit``.
    """
    return np.linalg.solve(problem.hessian, -problem.linear)


def analytic_box_solution(problem: QuadraticProblem, lower: float, upper: float) -> FloatArray:
    """Exact minimiser under box bounds when μ = 0: clip the separable optimum."""
    if problem.weights.mu_rate != 0:
        raise ValueError("The clipped closed form is only exact when mu_rate == 0 (samples decoupled)")
    w = problem.weights
    unconstrained = w.w_tracking / (w.w_tracking + w.lambda_effort) * problem.suit_torque_Nm
    return np.clip(unconstrained, lower, upper)


# --------------------------------------------------------------------------- #
# Numerical optimisation
# --------------------------------------------------------------------------- #
@dataclass
class OptimizationResult:
    """Outcome of a numerical optimisation run."""

    torque_Nm: FloatArray
    objective: float
    history: List[float]
    success: bool
    message: str
    n_iterations: int
    method: str
    rate_limit_enforced: bool
    max_violations: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    wall_time_s: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "success": self.success,
            "message": self.message,
            "n_iterations": self.n_iterations,
            "objective": self.objective,
            "rate_limit_enforced": self.rate_limit_enforced,
            "max_violations": self.max_violations,
            "warnings": self.warnings,
            "wall_time_s": self.wall_time_s,
        }


def _rate_constraint_matrix(n: int) -> FloatArray:
    """A such that (Aτ)ᵢ = τᵢ₊₁ − τᵢ."""
    return difference_matrix(n, 1.0)


def optimize_actuator_profile(
    problem: QuadraticProblem,
    actuator: ActuatorModel,
    enforce_rate_limit: bool = True,
    method: str = "SLSQP",
    x0: Optional[ArrayLike] = None,
    max_iter: int = 1000,
    ftol: float = 1e-12,
) -> OptimizationResult:
    """Minimise J subject to the actuator limits with ``scipy.optimize.minimize``.

    The solver works on the dimensionless variable x = τ/τref and objective N·J
    (better conditioned than raw N·m and a 1/N-scaled cost). Bounds come from
    :meth:`ActuatorModel.torque_bounds`; the optional rate limit becomes the
    linear inequalities |τᵢ₊₁ − τᵢ| ≤ τ̇max Δt.
    """
    if method not in ("SLSQP", "trust-constr", "L-BFGS-B"):
        raise ValueError(f"Unsupported method {method!r}")
    n, scale = problem.n, problem.torque_scale_Nm
    lb, ub = actuator.torque_bounds()
    use_rate = enforce_rate_limit and actuator.max_torque_rate_Nm_s is not None
    if use_rate and method == "L-BFGS-B":
        raise ValueError("L-BFGS-B only supports bound constraints; use SLSQP or trust-constr for rate limits")

    x_start = np.zeros(n) if x0 is None else np.clip(np.asarray(x0, dtype=float).reshape(-1), lb, ub) / scale
    if x_start.size != n:
        raise ValueError(f"x0 has {x_start.size} samples, expected {n}")

    def fun(x: FloatArray) -> float:
        return n * problem.objective(x * scale)

    def jac(x: FloatArray) -> FloatArray:
        return n * scale * problem.gradient(x * scale)

    def hess(x: FloatArray) -> FloatArray:
        return n * scale**2 * problem.hessian

    history: List[float] = [problem.objective(x_start * scale)]

    def callback(xk: Any, *args: Any) -> bool:
        x = xk if isinstance(xk, np.ndarray) else np.asarray(getattr(xk, "x", xk))
        history.append(problem.objective(np.asarray(x) * scale))
        return False

    bounds = Bounds(np.full(n, lb / scale), np.full(n, ub / scale))
    constraints: Any = ()
    if use_rate:
        a = _rate_constraint_matrix(n)
        step = actuator.max_torque_rate_Nm_s * problem.dt_s / scale  # type: ignore[operator]
        if method == "SLSQP":
            a_full = np.vstack([a, -a])
            b_full = np.full(2 * (n - 1), step)
            constraints = [{"type": "ineq", "fun": lambda x: b_full - _matmul(a_full, x), "jac": lambda x: -a_full}]
        else:
            constraints = [LinearConstraint(a, -step, step)]

    options: Dict[str, Any] = {"maxiter": max_iter}
    kwargs: Dict[str, Any] = {}
    if method == "SLSQP":
        options["ftol"] = ftol
    elif method == "L-BFGS-B":
        options.update({"ftol": ftol, "gtol": 1e-12})
    else:
        options.update({"gtol": 1e-12, "xtol": 1e-14})
        kwargs["hess"] = hess

    caught: List[str] = []
    start = time.perf_counter()
    with _warnings.catch_warnings(record=True) as record:
        _warnings.simplefilter("always")
        res = minimize(
            fun,
            x_start,
            jac=jac,
            method=method,
            bounds=bounds,
            constraints=constraints,
            callback=callback,
            options=options,
            **kwargs,
        )
    elapsed = time.perf_counter() - start
    caught.extend(f"{w.category.__name__}: {w.message}" for w in record)

    torque = np.asarray(res.x, dtype=float) * scale
    pre_clip = float(max(0.0, np.max(torque - ub), np.max(lb - torque)))
    torque = np.clip(torque, lb, ub)  # remove round-off-level bound excursions
    if pre_clip > 1e-6:
        caught.append(f"Solver returned bound excursion of {pre_clip:.3g} N·m (clipped)")
    violations = actuator.limit_violations(torque, problem.dt_s)
    if not use_rate:
        violations["torque_rate_Nm_s"] = 0.0  # rate limit intentionally not enforced in this run
    if not res.success:
        caught.append(f"Optimizer reported failure: {res.message}")
    for key, value in violations.items():
        if value > 1e-6:
            caught.append(f"Constraint '{key}' violated by {value:.3g}")
    final = problem.objective(torque)
    if history[-1] != final:
        history.append(final)

    return OptimizationResult(
        torque_Nm=torque,
        objective=final,
        history=history,
        success=bool(res.success),
        message=str(res.message),
        n_iterations=int(getattr(res, "nit", len(history) - 1)),
        method=method,
        rate_limit_enforced=use_rate,
        max_violations=violations,
        warnings=caught,
        wall_time_s=elapsed,
    )


def projected_gradient_descent(
    problem: QuadraticProblem,
    lower: float,
    upper: float,
    x0: Optional[ArrayLike] = None,
    max_iter: int = 20000,
    tol: float = 1e-12,
) -> OptimizationResult:
    """Educational projected gradient descent for the box-constrained problem.

    Uses the fixed step 1/L with L = λmax(H), which guarantees monotone decrease
    for a convex quadratic. Rate limits are *not* supported (projection onto that
    polytope has no simple closed form).
    """
    n = problem.n
    tau = np.zeros(n) if x0 is None else np.clip(np.asarray(x0, dtype=float).reshape(-1), lower, upper)
    lipschitz = float(np.linalg.eigvalsh(problem.hessian)[-1])
    step = 1.0 / lipschitz
    history = [problem.objective(tau)]
    converged = False
    start = time.perf_counter()
    iteration = 0
    for iteration in range(1, max_iter + 1):
        new_tau = np.clip(tau - step * problem.gradient(tau), lower, upper)
        change = float(np.max(np.abs(new_tau - tau)))
        tau = new_tau
        history.append(problem.objective(tau))
        if change < tol * max(1.0, problem.torque_scale_Nm):
            converged = True
            break
    elapsed = time.perf_counter() - start
    return OptimizationResult(
        torque_Nm=tau,
        objective=history[-1],
        history=history,
        success=converged,
        message="converged (step change below tolerance)" if converged else "maximum iterations reached",
        n_iterations=iteration,
        method="projected-gradient",
        rate_limit_enforced=False,
        max_violations={"torque_Nm": float(max(0.0, np.max(tau - upper), np.max(lower - tau)))},
        warnings=[] if converged else ["Projected gradient descent hit max_iter"],
        wall_time_s=elapsed,
    )


def random_feasible_profile(
    actuator: ActuatorModel, n: int, dt_s: float, rng: np.random.Generator, enforce_rate_limit: bool = True
) -> FloatArray:
    """A random torque profile that satisfies every actuator limit (for validation).

    A bounded random walk: steps are limited to τ̇max·Δt and each step is projected
    onto the torque bounds. Projection onto an interval is non-expansive and the
    previous sample is inside it, so the projected step never exceeds the rate limit.
    """
    lb, ub = actuator.torque_bounds()
    use_rate = enforce_rate_limit and actuator.max_torque_rate_Nm_s is not None
    max_step = actuator.max_torque_rate_Nm_s * dt_s if use_rate else (ub - lb)  # type: ignore[operator]
    tau = np.empty(n)
    tau[0] = rng.uniform(lb, ub) if not use_rate else rng.uniform(max(lb, -max_step), min(ub, max_step))
    for i in range(1, n):
        tau[i] = np.clip(tau[i - 1] + rng.uniform(-max_step, max_step), lb, ub)
    return tau


def profile_difference(a: ArrayLike, b: ArrayLike) -> Tuple[float, float]:
    """(max |a − b|, RMS(a − b)) between two sampled profiles."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return float(np.max(np.abs(diff))), float(np.sqrt(np.mean(diff**2)))
