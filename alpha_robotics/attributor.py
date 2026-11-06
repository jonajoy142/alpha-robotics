"""
Failure attribution, reliability boundary discovery, and degradation scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class FaultSensitivity:
    """Quantitative vulnerability profile for a single fault dimension."""
    fault_name: str
    nominal_value: Any
    tested_values: list[Any]
    success_rates: list[float]
    failure_threshold: float | None
    degradation_slope: float
    max_success_drop: float
    cliff_detected: bool
    auc_normalized: float
    evidence: str


@dataclass
class DiagnosisReport:
    """Complete diagnostic audit of a robot policy under stress."""
    nominal_success_rate: float
    worst_case_success_rate: float
    overall_robustness_score: float  # 0 to 100
    grade: str                       # 'A', 'B', 'C', 'D', 'F'
    grade_color: str                 # hex code
    most_vulnerable_fault: str | None
    fault_sensitivities: list[FaultSensitivity] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class FailureAttributor:
    """
    Analyzes stress-battery results to find the minimal failure boundary
    and attribute performance drops to specific operational boundaries.
    """

    def __init__(self, records: list[dict[str, Any]]) -> None:
        if not records:
            raise ValueError("FailureAttributor requires at least one episode record.")
        self.records = records

    def analyze(self) -> DiagnosisReport:
        # 1. Identify baseline / nominal episodes (where all fault params are 0 or neutral)
        nominal_records = [
            r for r in self.records if r.get("cell_id", 0) == 0
        ]
        if not nominal_records:
            nominal_records = self.records[:5]

        nominal_success = float(np.mean([r["success"] for r in nominal_records]))
        worst_case_success = float(np.min([r["success"] for r in self.records]))

        # 2. Extract fault parameters tested across cells
        meta_keys = {"cell_id", "episode_id", "seed", "success", "reward", "steps"}
        fault_keys = [k for k in self.records[0].keys() if k not in meta_keys]

        sensitivities: list[FaultSensitivity] = []

        for fault_key in fault_keys:
            # Group records by value of this fault_key
            values_set = sorted(list({r[fault_key] for r in self.records}))
            if len(values_set) <= 1:
                continue

            tested_vals: list[Any] = []
            rates: list[float] = []

            for val in values_set:
                sub = [r for r in self.records if r[fault_key] == val]
                if sub:
                    tested_vals.append(val)
                    rates.append(float(np.mean([r["success"] for r in sub])))

            if not rates:
                continue

            max_drop = float(nominal_success - min(rates))

            # Detect cliff (steepest single-step drop >= 30%)
            diffs = [rates[i] - rates[i + 1] for i in range(len(rates) - 1)]
            max_step_drop = max(diffs) if diffs else 0.0
            cliff_detected = bool(max_step_drop >= 0.30)

            # Failure boundary: value where success drops below 50% of nominal
            threshold: float | None = None
            threshold_cutoff = nominal_success * 0.5
            for v, r in zip(tested_vals, rates):
                if r <= threshold_cutoff:
                    threshold = float(v)
                    break

            # Degradation slope
            if len(tested_vals) >= 2 and float(tested_vals[-1] - tested_vals[0]) != 0.0:
                span = float(tested_vals[-1] - tested_vals[0])
                slope = float((rates[-1] - rates[0]) / span)
            else:
                slope = 0.0

            # Normalized AUC
            auc_norm = float(np.mean(rates)) if rates else 0.0

            # Evidence statement
            if threshold is not None:
                evidence = f"Policy collapses (<=50% nominal) at {fault_key} = {threshold}."
            elif cliff_detected:
                evidence = f"Sharp cliff of -{max_step_drop * 100:.0f}% detected under {fault_key}."
            elif max_drop > 0.05:
                evidence = f"Moderate degradation: success drops from {nominal_success * 100:.0f}% to {min(rates) * 100:.0f}%."
            else:
                evidence = f"Highly resilient: success maintained within 5% of nominal across tested range."

            sensitivities.append(
                FaultSensitivity(
                    fault_name=fault_key,
                    nominal_value=tested_vals[0],
                    tested_values=tested_vals,
                    success_rates=rates,
                    failure_threshold=threshold,
                    degradation_slope=slope,
                    max_success_drop=max_drop,
                    cliff_detected=cliff_detected,
                    auc_normalized=auc_norm,
                    evidence=evidence,
                )
            )

        # Rank sensitivities by severity (max_success_drop descending)
        sensitivities.sort(key=lambda s: s.max_success_drop, reverse=True)
        most_vulnerable = sensitivities[0].fault_name if sensitivities and sensitivities[0].max_success_drop > 0.05 else None

        # Overall robustness score (0 to 100)
        if sensitivities:
            avg_auc = float(np.mean([s.auc_normalized for s in sensitivities]))
            robustness_score = round(avg_auc * 100.0, 1)
        else:
            robustness_score = round(nominal_success * 100.0, 1)

        # Grade calculation
        if robustness_score >= 90.0:
            grade = "A"
            color = "#10B981"  # Emerald green
        elif robustness_score >= 75.0:
            grade = "B"
            color = "#3B82F6"  # Blue
        elif robustness_score >= 60.0:
            grade = "C"
            color = "#F59E0B"  # Amber
        elif robustness_score >= 40.0:
            grade = "D"
            color = "#F97316"  # Orange
        else:
            grade = "F"
            color = "#EF4444"  # Red

        # Actionable recommendations
        recs: list[str] = []
        for s in sensitivities[:3]:
            if s.max_success_drop > 0.15:
                if "latency" in s.fault_name:
                    recs.append(f"Observation latency bottleneck: {s.evidence} Implement action-chunking or predictive latency buffers.")
                elif "slip" in s.fault_name:
                    recs.append(f"Actuation slip sensitivity: {s.evidence} Consider tactile slip-detection or compliant grasp thresholds.")
                elif "noise" in s.fault_name:
                    recs.append(f"Sensor noise sensitivity: {s.evidence} Augment training observations with random Gaussian noise.")
                elif "clamp" in s.fault_name:
                    recs.append(f"Workspace reach fragility: {s.evidence} Validate kinematic IK reachability envelopes prior to trajectory planning.")
                else:
                    recs.append(f"{s.fault_name}: {s.evidence}")

        if not recs:
            recs.append("Policy demonstrated high nominal and perturbed robustness across all evaluated fault conditions.")

        return DiagnosisReport(
            nominal_success_rate=nominal_success,
            worst_case_success_rate=worst_case_success,
            overall_robustness_score=robustness_score,
            grade=grade,
            grade_color=color,
            most_vulnerable_fault=most_vulnerable,
            fault_sensitivities=sensitivities,
            recommendations=recs,
        )

