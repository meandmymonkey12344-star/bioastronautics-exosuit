"""Configuration loading and validation.

Every physical parameter of the simulation lives in a JSON file (see
``config/default_parameters.json``). This module turns that file into frozen,
validated dataclasses so that the physics modules never receive a missing,
mistyped or physically meaningless value.

JSON has no comments, so any key that starts with ``_`` (for example
``"_units"``) is treated as documentation and ignored.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unknown or invalid."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _check_finite(name: str, value: float) -> None:
    _check(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"'{name}' must be a finite number, got {value!r}",
    )


def _check_positive(name: str, value: float) -> None:
    _check_finite(name, value)
    _check(value > 0, f"'{name}' must be > 0, got {value!r}")


def _check_nonnegative(name: str, value: float) -> None:
    _check_finite(name, value)
    _check(value >= 0, f"'{name}' must be >= 0, got {value!r}")


def _check_fraction(name: str, value: float, *, allow_one: bool = True) -> None:
    _check_finite(name, value)
    upper_ok = value <= 1 if allow_one else value < 1
    _check(0 <= value and upper_ok, f"'{name}' must lie in [0, 1{']' if allow_one else ')'}, got {value!r}")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Gravitational environment. 9.81 Earth, 3.71 Mars, 1.62 Moon, 0 orbit."""

    gravity_m_s2: float = 1.62
    label: str = "lunar surface"

    def __post_init__(self) -> None:
        _check_nonnegative("environment.gravity_m_s2", self.gravity_m_s2)


@dataclass(frozen=True)
class AnthropometryConfig:
    """Segment lengths and inertial ratios (fractions of body mass / segment length)."""

    body_mass_kg: float = 75.0
    thigh_length_m: float = 0.45
    shank_length_m: float = 0.43
    thigh_mass_fraction: float = 0.100
    thigh_com_fraction: float = 0.433
    thigh_radius_of_gyration_fraction: float = 0.323
    shank_mass_fraction: float = 0.0465
    shank_com_fraction: float = 0.433
    shank_radius_of_gyration_fraction: float = 0.302
    foot_mass_fraction: float = 0.0145

    def __post_init__(self) -> None:
        _check_positive("anthropometry.body_mass_kg", self.body_mass_kg)
        _check_positive("anthropometry.thigh_length_m", self.thigh_length_m)
        _check_positive("anthropometry.shank_length_m", self.shank_length_m)
        for name in (
            "thigh_mass_fraction",
            "thigh_com_fraction",
            "thigh_radius_of_gyration_fraction",
            "shank_mass_fraction",
            "shank_com_fraction",
            "shank_radius_of_gyration_fraction",
            "foot_mass_fraction",
        ):
            _check_fraction(f"anthropometry.{name}", getattr(self, name))


@dataclass(frozen=True)
class TrajectoryConfig:
    """Prescribed movement. Angles in degrees here; converted to radians internally."""

    mode: str = "flexion"
    knee_start_deg: float = 0.0
    knee_end_deg: float = 90.0
    hip_start_deg: float = 0.0
    hip_end_deg: float = 0.0
    duration_s: float = 1.0
    n_samples: int = 201

    def __post_init__(self) -> None:
        _check(
            self.mode in ("flexion", "flexion_extension"),
            f"trajectory.mode must be 'flexion' or 'flexion_extension', got {self.mode!r}",
        )
        for name in ("knee_start_deg", "knee_end_deg", "hip_start_deg", "hip_end_deg"):
            _check_finite(f"trajectory.{name}", getattr(self, name))
        for name in ("knee_start_deg", "knee_end_deg"):
            value = getattr(self, name)
            _check(
                0.0 <= value <= 150.0,
                f"trajectory.{name} must be within the anatomical range [0, 150] deg "
                f"(knee hyperextension is not modelled), got {value!r}",
            )
        _check_positive("trajectory.duration_s", self.duration_s)
        _check(
            isinstance(self.n_samples, int) and not isinstance(self.n_samples, bool) and self.n_samples >= 5,
            f"trajectory.n_samples must be an integer >= 5, got {self.n_samples!r}",
        )


@dataclass(frozen=True)
class BaselineConfig:
    """Source of the unsuited human knee torque: physics-based or tabulated data."""

    source: str = "inverse_dynamics"
    csv_path: Optional[str] = None

    def __post_init__(self) -> None:
        _check(
            self.source in ("inverse_dynamics", "csv"),
            f"baseline_torque.source must be 'inverse_dynamics' or 'csv', got {self.source!r}",
        )
        if self.source == "csv":
            _check(
                isinstance(self.csv_path, str) and len(self.csv_path) > 0,
                "baseline_torque.csv_path is required when source is 'csv'",
            )


@dataclass(frozen=True)
class SuitConfig:
    """Illustrative suit resistance law tau = E(phi)(1 + h s) + tau_f s."""

    k_linear_Nm_per_rad: float = 6.0
    k_quadratic_Nm_per_rad2: float = 4.0
    friction_Nm: float = 1.0
    hysteresis_fraction: float = 0.25
    branch_transition_deg: float = 3.0
    initial_branch_state: float = 0.0

    def __post_init__(self) -> None:
        _check_nonnegative("spacesuit.k_linear_Nm_per_rad", self.k_linear_Nm_per_rad)
        _check_nonnegative("spacesuit.k_quadratic_Nm_per_rad2", self.k_quadratic_Nm_per_rad2)
        _check_nonnegative("spacesuit.friction_Nm", self.friction_Nm)
        _check_fraction("spacesuit.hysteresis_fraction", self.hysteresis_fraction, allow_one=False)
        _check_positive("spacesuit.branch_transition_deg", self.branch_transition_deg)
        _check_finite("spacesuit.initial_branch_state", self.initial_branch_state)
        _check(
            -1.0 <= self.initial_branch_state <= 1.0,
            f"spacesuit.initial_branch_state must lie in [-1, 1], got {self.initial_branch_state!r}",
        )


@dataclass(frozen=True)
class ActuatorConfig:
    """Illustrative actuator limits; ``max_torque_rate_Nm_s = null`` disables the rate limit."""

    max_torque_Nm: float = 20.0
    max_force_N: float = 600.0
    moment_arm_m: float = 0.05
    max_torque_rate_Nm_s: Optional[float] = 40.0
    efficiency: float = 0.70
    regeneration_efficiency: float = 0.0
    bidirectional: bool = True

    def __post_init__(self) -> None:
        _check_positive("actuator.max_torque_Nm", self.max_torque_Nm)
        _check_positive("actuator.max_force_N", self.max_force_N)
        _check_positive("actuator.moment_arm_m", self.moment_arm_m)
        if self.max_torque_rate_Nm_s is not None:
            _check_positive("actuator.max_torque_rate_Nm_s", self.max_torque_rate_Nm_s)
        _check_finite("actuator.efficiency", self.efficiency)
        _check(0 < self.efficiency <= 1, f"actuator.efficiency must lie in (0, 1], got {self.efficiency!r}")
        _check_fraction("actuator.regeneration_efficiency", self.regeneration_efficiency)
        _check(isinstance(self.bidirectional, bool), "actuator.bidirectional must be true or false")


@dataclass(frozen=True)
class OptimizationConfig:
    """Objective weights (dimensionless after normalisation) and solver settings."""

    w_tracking: float = 1.0
    lambda_effort: float = 0.01
    mu_rate: float = 0.0
    torque_scale_Nm: Optional[float] = None
    method: str = "SLSQP"
    max_iter: int = 1000
    ftol: float = 1e-12
    enforce_rate_limit: bool = True

    def __post_init__(self) -> None:
        _check_nonnegative("optimization.w_tracking", self.w_tracking)
        _check_nonnegative("optimization.lambda_effort", self.lambda_effort)
        _check_nonnegative("optimization.mu_rate", self.mu_rate)
        _check(
            self.w_tracking + self.lambda_effort > 0,
            "optimization.w_tracking + lambda_effort must be > 0 (otherwise the problem is not strictly convex)",
        )
        if self.torque_scale_Nm is not None:
            _check_positive("optimization.torque_scale_Nm", self.torque_scale_Nm)
        _check(
            self.method in ("SLSQP", "trust-constr", "L-BFGS-B"),
            f"optimization.method must be 'SLSQP', 'trust-constr' or 'L-BFGS-B', got {self.method!r}",
        )
        _check(isinstance(self.max_iter, int) and self.max_iter > 0, "optimization.max_iter must be a positive integer")
        _check_positive("optimization.ftol", self.ftol)


@dataclass(frozen=True)
class CycleDemoConfig:
    """Optional flexion-extension cycle used to show hysteresis and negative work."""

    enabled: bool = True
    duration_s: float = 1.0

    def __post_init__(self) -> None:
        _check_positive("cycle_demo.duration_s", self.duration_s)


@dataclass(frozen=True)
class SensitivityConfig:
    """Illustrative parameter ranges for the one-at-a-time sensitivity study."""

    enabled: bool = True
    suit_stiffness_scale: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.5, 2.0])
    max_torque_Nm: List[float] = field(default_factory=lambda: [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0])
    moment_arm_m: List[float] = field(default_factory=lambda: [0.02, 0.03, 0.04, 0.05, 0.07, 0.10])
    lambda_effort: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0])
    duration_s: List[float] = field(default_factory=lambda: [0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0])

    def __post_init__(self) -> None:
        for name in ("suit_stiffness_scale", "max_torque_Nm", "moment_arm_m", "duration_s"):
            values = getattr(self, name)
            _check(isinstance(values, list) and len(values) > 0, f"sensitivity.{name} must be a non-empty list")
            for v in values:
                _check_positive(f"sensitivity.{name}[...]", v)
        _check(len(self.lambda_effort) > 0, "sensitivity.lambda_effort must be a non-empty list")
        for v in self.lambda_effort:
            _check_nonnegative("sensitivity.lambda_effort[...]", v)


@dataclass(frozen=True)
class OutputConfig:
    """Where and how results are written."""

    directory: str = "outputs"
    save_pdf: bool = True
    animation: bool = True
    dpi: int = 200

    def __post_init__(self) -> None:
        _check(isinstance(self.dpi, int) and self.dpi >= 50, "output.dpi must be an integer >= 50")


@dataclass(frozen=True)
class SimulationConfig:
    """Complete, validated simulation configuration."""

    description: str = ""
    random_seed: int = 42
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    anthropometry: AnthropometryConfig = field(default_factory=AnthropometryConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    baseline_torque: BaselineConfig = field(default_factory=BaselineConfig)
    spacesuit: SuitConfig = field(default_factory=SuitConfig)
    actuator: ActuatorConfig = field(default_factory=ActuatorConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    cycle_demo: CycleDemoConfig = field(default_factory=CycleDemoConfig)
    sensitivity: SensitivityConfig = field(default_factory=SensitivityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def with_section(self, section: str, **changes: Any) -> "SimulationConfig":
        """Return a copy with fields of one section replaced (validation re-runs)."""
        current = getattr(self, section)
        return replace(self, **{section: replace(current, **changes)})


_SECTIONS: Dict[str, Type[Any]] = {
    "environment": EnvironmentConfig,
    "anthropometry": AnthropometryConfig,
    "trajectory": TrajectoryConfig,
    "baseline_torque": BaselineConfig,
    "spacesuit": SuitConfig,
    "actuator": ActuatorConfig,
    "optimization": OptimizationConfig,
    "cycle_demo": CycleDemoConfig,
    "sensitivity": SensitivityConfig,
    "output": OutputConfig,
}


def _strip_comments(data: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def _section_from_dict(cls: Type[T], data: Any, section: str) -> T:
    _check(isinstance(data, Mapping), f"Section '{section}' must be a JSON object")
    clean = _strip_comments(data)
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(clean) - known)
    _check(not unknown, f"Unknown key(s) in section '{section}': {unknown}. Allowed keys: {sorted(known)}")
    try:
        return cls(**clean)
    except TypeError as exc:  # wrong argument structure
        raise ConfigError(f"Invalid section '{section}': {exc}") from exc


def config_from_dict(data: Mapping[str, Any]) -> SimulationConfig:
    """Build a validated :class:`SimulationConfig` from a (parsed JSON) mapping.

    Missing sections and keys fall back to the documented defaults; unknown keys
    raise :class:`ConfigError` so that typos never silently change a result.
    """
    _check(isinstance(data, Mapping), "Configuration root must be a JSON object")
    clean = _strip_comments(data)
    allowed = set(_SECTIONS) | {"description", "random_seed"}
    unknown = sorted(set(clean) - allowed)
    _check(not unknown, f"Unknown top-level key(s): {unknown}. Allowed keys: {sorted(allowed)}")
    kwargs: Dict[str, Any] = {}
    for name, cls in _SECTIONS.items():
        if name in clean:
            kwargs[name] = _section_from_dict(cls, clean[name], name)
    if "description" in clean:
        kwargs["description"] = str(clean["description"])
    if "random_seed" in clean:
        seed = clean["random_seed"]
        _check(isinstance(seed, int) and not isinstance(seed, bool), "random_seed must be an integer")
        kwargs["random_seed"] = seed
    return SimulationConfig(**kwargs)


def load_config(path: "str | Path") -> SimulationConfig:
    """Load and validate a JSON configuration file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Configuration file {path} is not valid JSON: {exc}") from exc
    return config_from_dict(data)
