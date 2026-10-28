"""Command-line entry point: run the full simulation and write all outputs.

Usage::

    python -m exosuit_sim.main --config config/default_parameters.json --output outputs

Outputs (under the output directory)::

    figures/   PNG + PDF figures and leg_animation.gif
    data/      time_series_primary.csv, time_series_cycle.csv, sensitivity.csv
    summary.json   all metrics, solver status and validation checks
    report.txt     human-readable run report
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from . import __version__
from .config import ConfigError, SimulationConfig, load_config
from .sensitivity import SensitivitySweep, run_sensitivity, sweep_table
from .simulation import ScenarioResult, SolverComparison, compare_solvers, run_scenario, validation_checks


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def write_time_series(result: ScenarioResult, path: Path) -> Path:
    """Write the per-sample time series of one scenario to CSV."""
    traj = result.trajectory
    columns = {
        "time_s": traj.time_s,
        "hip_flexion_deg": traj.hip_deg,
        "knee_flexion_deg": traj.knee_deg,
        "knee_velocity_deg_s": np.degrees(traj.knee_vel_rad_s),
        "knee_acceleration_deg_s2": np.degrees(traj.knee_acc_rad_s2),
        "ankle_x_m": result.ankle_xy[:, 0],
        "ankle_y_m": result.ankle_xy[:, 1],
        "baseline_knee_torque_Nm": result.baseline_knee_torque_Nm,
        "suit_branch_state": result.suit_branch_state,
        "suit_resistance_torque_Nm": result.suit_torque_Nm,
        "actuator_torque_Nm": result.actuator_torque_Nm,
        "actuator_force_N": result.actuator_force_N,
        "human_torque_suited_unassisted_Nm": result.human_unassisted_Nm,
        "human_torque_suited_assisted_Nm": result.human_assisted_Nm,
        "actuator_power_W": result.actuator_power_W,
        "actuator_cumulative_work_J": result.actuator_cumulative_work_J,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# SIMULATED DATA from an illustrative model (exosuit_sim); not experimental measurements.\n")
        writer = csv.writer(handle)
        writer.writerow(columns.keys())
        for row in zip(*columns.values()):
            writer.writerow([f"{v:.6g}" for v in row])
    return path


def write_sensitivity_csv(sweeps: Sequence[SensitivitySweep], path: Path) -> Path:
    rows = sweep_table(list(sweeps))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("# SIMULATED sensitivity study; parameter ranges are illustrative.\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    return path


def _sweep_trends(sweeps: Sequence[SensitivitySweep]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for s in sweeps:
        out[s.key] = {
            "values": [p.value for p in s.points],
            "reduction_rms_additional_torque_pct": [p.metrics["reduction_rms_additional_torque_pct"] for p in s.points],
            s.focus_metric: [p.metrics[s.focus_metric] for p in s.points],
            "solver_success": [float(p.success) for p in s.points],
        }
    return out


def build_summary(
    cfg: SimulationConfig,
    primary: ScenarioResult,
    checks: Dict[str, Any],
    comparison: SolverComparison,
    cycle: Optional[ScenarioResult],
    sweeps: Sequence[SensitivitySweep],
    runtime_s: float,
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "disclaimer": "Simulated results from an illustrative model. Not experimental data; "
        "no claim about astronaut metabolic expenditure is made.",
        "software": {
            "exosuit_sim": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
        "description": cfg.description,
        "baseline_source": primary.baseline_source,
        "primary": {
            "metrics": primary.metrics,
            "reference_metrics": primary.reference_metrics,
            "optimizer": primary.optimized.summary(),
            "objective_history": primary.optimized.history,
        },
        "validation_checks": checks,
        "solver_comparison": {
            "note": "bounds-only problem with mu_rate = 1e-3; J* = best objective found",
            "J_star": comparison.exact_objective,
            "runs": {k: v.summary() for k, v in comparison.results.items()},
        },
        "runtime_s": runtime_s,
    }
    if cycle is not None:
        summary["cycle"] = {"metrics": cycle.metrics, "optimizer": cycle.optimized.summary()}
    if sweeps:
        summary["sensitivity"] = _sweep_trends(sweeps)
    return summary


def format_report(summary: Dict[str, Any]) -> str:
    m = summary["primary"]["metrics"]
    opt = summary["primary"]["optimizer"]
    chk = summary["validation_checks"]
    lines = [
        "EXOSUIT SIMULATION REPORT",
        "=" * 72,
        summary["disclaimer"],
        "",
        f"Baseline torque source : {summary['baseline_source']}",
        f"Software               : {summary['software']}",
        "",
        "OPTIMIZER (primary 0->90 deg flexion)",
        f"  method={opt['method']}  success={opt['success']}  iterations={opt['n_iterations']}",
        f"  message: {opt['message']}",
        f"  objective J = {opt['objective']:.6g}   rate limit enforced: {opt['rate_limit_enforced']}",
        f"  max limit violations: {opt['max_violations']}",
        f"  warnings: {opt['warnings'] or 'none'}",
        "",
        "PERFORMANCE (primary)",
    ]
    lines += [f"  {k:<48s} {v:12.5g}" for k, v in m.items()]
    lines += ["", "VALIDATION CHECKS"]
    lines += [f"  {k:<58s} {v}" for k, v in chk.items()]
    lines += ["", "SOLVER COMPARISON (bounds only, mu = 1e-3)"]
    for name, run in summary["solver_comparison"]["runs"].items():
        gap = run["objective"] - summary["solver_comparison"]["J_star"]
        lines.append(
            f"  {name:<20s} success={run['success']!s:<5} it={run['n_iterations']:<6d} J-J*={gap:.3e}  "
            f"t={run['wall_time_s']:.3f}s"
        )
    if "cycle" in summary:
        c = summary["cycle"]["metrics"]
        lines += [
            "",
            "FLEXION-EXTENSION CYCLE",
            f"  actuator W+ = {c['actuator_positive_work_J']:.3f} J, W- = {c['actuator_negative_work_J']:.3f} J, "
            f"W_net = {c['actuator_net_work_J']:.3f} J",
            f"  work done against suit over cycle (hysteresis loss) = {c['work_done_against_suit_J']:.3f} J",
            f"  reduction of RMS added human torque = {c['reduction_rms_additional_torque_pct']:.2f} %",
        ]
    if "sensitivity" in summary:
        lines += ["", "SENSITIVITY (reduction of RMS added human torque, %)"]
        for key, data in summary["sensitivity"].items():
            pairs = ", ".join(f"{v:g}->{r:.1f}" for v, r in zip(data["values"], data["reduction_rms_additional_torque_pct"]))
            lines.append(f"  {key}: {pairs}")
    lines += ["", f"Total runtime: {summary['runtime_s']:.1f} s"]
    return "\n".join(lines) + "\n"


def run(
    config_path: Path,
    output_dir: Optional[Path] = None,
    sensitivity: Optional[bool] = None,
    animation: Optional[bool] = None,
    copy_figures_to: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the complete study and write all outputs. Returns the summary dict."""
    start = time.perf_counter()

    def log(msg: str) -> None:
        if verbose:
            print(f"[exosuit_sim] {msg}", flush=True)

    cfg = load_config(config_path)
    base_dir = Path.cwd()  # relative paths in the config (output dir, baseline CSV) resolve from here
    out = Path(output_dir) if output_dir is not None else base_dir / cfg.output.directory
    fig_dir, data_dir = out / "figures", out / "data"

    log(f"config: {config_path}")
    primary = run_scenario(cfg, label="primary", base_dir=base_dir)
    log(f"primary optimisation: success={primary.optimized.success}, J={primary.optimized.objective:.6g}, "
        f"reduction={primary.metrics['reduction_rms_additional_torque_pct']:.2f}%")
    checks = validation_checks(primary)
    comparison = compare_solvers(primary)
    cycle = None
    if cfg.cycle_demo.enabled:
        cycle_cfg = cfg.with_section("trajectory", duration_s=cfg.cycle_demo.duration_s)
        cycle = run_scenario(cycle_cfg, mode="flexion_extension", label="cycle", base_dir=base_dir)
        log(f"cycle optimisation: success={cycle.optimized.success}")
    do_sens = cfg.sensitivity.enabled if sensitivity is None else sensitivity
    sweeps: List[SensitivitySweep] = run_sensitivity(cfg, base_dir=base_dir) if do_sens else []
    if sweeps:
        log(f"sensitivity: {sum(len(s.points) for s in sweeps)} runs")

    from .visualization import generate_all_figures  # deferred: keeps physics importable without a display

    make_anim = cfg.output.animation if animation is None else animation
    figures = generate_all_figures(primary, comparison, fig_dir, cycle=cycle, sweeps=sweeps, dpi=cfg.output.dpi,
                                   save_pdf=cfg.output.save_pdf, make_animation=make_anim)
    write_time_series(primary, data_dir / "time_series_primary.csv")
    if cycle is not None:
        write_time_series(cycle, data_dir / "time_series_cycle.csv")
    if sweeps:
        write_sensitivity_csv(sweeps, data_dir / "sensitivity.csv")

    summary = build_summary(cfg, primary, checks, comparison, cycle, sweeps, time.perf_counter() - start)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    (out / "report.txt").write_text(format_report(summary), encoding="utf-8")
    log(f"wrote {len(figures)} figure files to {fig_dir}")

    if copy_figures_to is not None:
        copy_figures_to.mkdir(parents=True, exist_ok=True)
        for f in figures:
            if f.suffix in (".png", ".gif"):
                shutil.copy2(f, copy_figures_to / f.name)
        log(f"copied PNG/GIF figures to {copy_figures_to}")
    log(f"done in {summary['runtime_s']:.1f} s; report: {out / 'report.txt'}")
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="exosuit-sim",
        description="Optimise an active exoskeleton knee-torque profile against modelled EVA suit resistance.",
    )
    parser.add_argument("--config", type=Path, default=Path("config/default_parameters.json"),
                        help="JSON configuration file (default: %(default)s)")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: from config)")
    parser.add_argument("--no-sensitivity", action="store_true", help="Skip the sensitivity study")
    parser.add_argument("--no-animation", action="store_true", help="Skip the GIF animation")
    parser.add_argument("--copy-figures-to", type=Path, default=None,
                        help="Also copy PNG/GIF figures here (e.g. docs/figures for the README)")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        run(
            args.config,
            output_dir=args.output,
            sensitivity=False if args.no_sensitivity else None,
            animation=False if args.no_animation else None,
            copy_figures_to=args.copy_figures_to,
            verbose=not args.quiet,
        )
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
