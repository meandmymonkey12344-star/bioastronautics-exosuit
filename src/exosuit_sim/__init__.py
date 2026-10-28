"""exosuit_sim — active exoskeleton compensation of EVA spacesuit knee resistance.

A simplified, documented engineering model (see docs/MODEL.md). Suit and
actuator parameters are illustrative, and results are simulated rather than
experimentally validated.
"""

from .actuator import ActuatorModel
from .config import ConfigError, SimulationConfig, config_from_dict, load_config
from .kinematics import LegGeometry, homogeneous_chain, planar_forward_kinematics, rotation_2d
from .optimization import ObjectiveWeights, build_problem, optimize_actuator_profile
from .simulation import ScenarioResult, run_scenario
from .spacesuit import SuitResistanceModel
from .trajectory import JointTrajectory, build_trajectory

__version__ = "1.0.0"

__all__ = [
    "ActuatorModel",
    "ConfigError",
    "JointTrajectory",
    "LegGeometry",
    "ObjectiveWeights",
    "ScenarioResult",
    "SimulationConfig",
    "SuitResistanceModel",
    "build_problem",
    "build_trajectory",
    "config_from_dict",
    "homogeneous_chain",
    "load_config",
    "optimize_actuator_profile",
    "planar_forward_kinematics",
    "rotation_2d",
    "run_scenario",
    "__version__",
]
