"""End-to-end: the CLI runs and writes every expected artefact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from exosuit_sim.main import main, run

EXPECTED_FIGURES = [
    "fig01_torque_comparison_time",
    "fig02_torque_vs_angle",
    "fig03_actuator_performance",
    "fig04_optimization_convergence",
    "fig05_energy_analysis",
    "fig06_leg_kinematics",
    "fig07_flexion_extension_cycle",
    "fig08_analytic_vs_numerical",
    "fig09_sensitivity_suit_stiffness_scale",
    "fig10_sensitivity_max_torque_Nm",
    "fig11_sensitivity_moment_arm_m",
    "fig12_sensitivity_lambda_effort",
    "fig13_sensitivity_duration_s",
]


@pytest.fixture()
def small_config(tmp_path: Path, repo_root: Path) -> Path:
    data = json.loads((repo_root / "config" / "default_parameters.json").read_text())
    data["trajectory"]["n_samples"] = 61
    data["sensitivity"] = {
        "enabled": True,
        "suit_stiffness_scale": [0.5, 1.0],
        "max_torque_Nm": [10.0, 20.0],
        "moment_arm_m": [0.03, 0.05],
        "lambda_effort": [0.01, 1.0],
        "duration_s": [0.5, 1.0],
    }
    data["output"]["dpi"] = 60
    path = tmp_path / "small.json"
    path.write_text(json.dumps(data))
    return path


def test_full_pipeline_writes_all_outputs(small_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = run(small_config, output_dir=out, verbose=False)
    for name in EXPECTED_FIGURES:
        assert (out / "figures" / f"{name}.png").stat().st_size > 1000
        assert (out / "figures" / f"{name}.pdf").stat().st_size > 1000
    assert (out / "figures" / "leg_animation.gif").is_file()
    for f in ("time_series_primary.csv", "time_series_cycle.csv", "sensitivity.csv"):
        assert (out / "data" / f).is_file()
    loaded = json.loads((out / "summary.json").read_text())
    assert loaded["primary"]["optimizer"]["success"] is True
    assert loaded["validation_checks"]["optimized_profile_feasible"] is True
    assert "not experimental" in (out / "report.txt").read_text().lower()
    assert summary["primary"]["metrics"]["reduction_rms_additional_torque_pct"] > 0
    assert summary["cycle"]["metrics"]["actuator_negative_work_J"] < 0  # braking during extension


def test_cli_exit_codes(small_config: Path, tmp_path: Path) -> None:
    out = tmp_path / "cli"
    assert main(["--config", str(small_config), "--output", str(out), "--no-sensitivity", "--no-animation",
                 "--quiet"]) == 0
    assert (out / "summary.json").is_file()
    assert main(["--config", str(tmp_path / "missing.json"), "--quiet"]) == 2
