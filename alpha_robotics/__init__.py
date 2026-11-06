"""
Alpha Robotics: Automated reliability boundary discovery and failure diagnosis for robot policies.
"""

from alpha_robotics.wrapper import ChaosWrapper
from alpha_robotics.runner import SweepRunner
from alpha_robotics.attributor import FailureAttributor, DiagnosisReport, FaultSensitivity
from alpha_robotics.report import ReportGenerator

__version__ = "0.1.0"

__all__ = [
    "ChaosWrapper",
    "SweepRunner",
    "FailureAttributor",
    "DiagnosisReport",
    "FaultSensitivity",
    "ReportGenerator",
]

