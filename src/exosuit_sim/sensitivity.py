"""One-at-a-time parameter sensitivity study (illustrative ranges, docs/MODEL.md).

Each sweep changes a single parameter of the primary scenario and re-runs the
full pipeline, recording the optimised profile and its metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from .config import SimulationConfig
from .simulation import run_scenario

FloatArray = NDArray[np.float64]


@dataclass
class SweepPoint:
    value: float
    metrics: Dict[str, float]
    knee_deg: FloatArray
    time_s: FloatArray
    actuator_torque_Nm: FloatArray
    suit_torque_Nm: FloatArray
    success: bool


@dataclass
class SensitivitySweep:
    key: str
    label: str
    unit: str
    log_scale: bool
    focus_metric: str
    focus_label: str
    points: List[SweepPoint] = field(default_factory=list)


def _suit_scale(cfg: SimulationConfig, factor: float) -> SimulationConfig:
    s = cfg.spacesuit
    return cfg.with_section(
        "spacesuit",
        k_linear_Nm_per_rad=s.k_linear_Nm_per_rad * factor,
        k_quadratic_Nm_per_rad2=s.k_quadratic_Nm_per_rad2 * factor,
        friction_Nm=s.friction_Nm * factor,
    )


#: key → (label, unit, log axis, config modifier, focus metric, focus label)
SWEEPS: Dict[str, tuple] = {
    "suit_stiffness_scale": (
        "Suit stiffness scale (k₁, k₂, τ_f)", "×", False, _suit_scale,
        "actuator_positive_work_J", "Actuator positive work (J)",
    ),
    "max_torque_Nm": (
        "Actuator torque rating τ_max", "N·m", False,
        lambda c, v: c.with_section("actuator", max_torque_Nm=float(v)),
        "rms_additional_human_torque_assisted_Nm", "RMS residual added human torque (N·m)",
    ),
    "moment_arm_m": (
        "Moment arm r", "m", False,
        lambda c, v: c.with_section("actuator", moment_arm_m=float(v)),
        "peak_actuator_force_N", "Peak actuator force (N)",
    ),
    "lambda_effort": (
        "Effort weight λ", "–", True,
        lambda c, v: c.with_section("optimization", lambda_effort=float(v)),
        "rms_actuator_torque_Nm", "RMS actuator torque (N·m)",
    ),
    "duration_s": (
        "Movement duration T (speed)", "s", False,
        lambda c, v: c.with_section("trajectory", duration_s=float(v)),
        "peak_actuator_power_W", "Peak actuator power (W)",
    ),
}


def run_sensitivity(
    cfg: SimulationConfig,
    base_dir: Optional[Path] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> List[SensitivitySweep]:
    """Run every configured sweep on the primary (flexion) scenario."""
    sweeps: List[SensitivitySweep] = []
    for key, (label, unit, log_scale, modify, focus, focus_label) in SWEEPS.items():
        values = getattr(cfg.sensitivity, key)
        sweep = SensitivitySweep(key=key, label=label, unit=unit, log_scale=log_scale, focus_metric=focus,
                                 focus_label=focus_label)
        for value in values:
            if progress:
                progress(f"sensitivity {key} = {value}")
            res = run_scenario(modify(cfg, value), label=f"{key}={value}", base_dir=base_dir)
            sweep.points.append(
                SweepPoint(
                    value=float(value),
                    metrics=res.metrics,
                    knee_deg=res.trajectory.knee_deg,
                    time_s=res.trajectory.time_s,
                    actuator_torque_Nm=res.actuator_torque_Nm,
                    suit_torque_Nm=res.suit_torque_Nm,
                    success=res.optimized.success,
                )
            )
        sweeps.append(sweep)
    return sweeps


def sweep_table(sweeps: List[SensitivitySweep]) -> List[Dict[str, object]]:
    """Flatten sweeps into rows for CSV export."""
    rows: List[Dict[str, object]] = []
    for sweep in sweeps:
        for point in sweep.points:
            row: Dict[str, object] = {"parameter": sweep.key, "value": point.value, "solver_success": point.success}
            row.update(point.metrics)
            rows.append(row)
    return rows
