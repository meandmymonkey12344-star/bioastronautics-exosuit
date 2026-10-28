"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from exosuit_sim.config import SimulationConfig, load_config
from exosuit_sim.simulation import ScenarioResult, run_scenario

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def default_config() -> SimulationConfig:
    return load_config(REPO_ROOT / "config" / "default_parameters.json")


@pytest.fixture(scope="session")
def primary_result(default_config: SimulationConfig) -> ScenarioResult:
    return run_scenario(default_config)
