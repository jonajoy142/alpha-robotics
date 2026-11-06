"""
Self-contained HTML ReportGenerator for Alpha Reliability Scorecards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from jinja2 import Template

from alpha_robotics.attributor import DiagnosisReport


class ReportGenerator:
    """
    Renders diagnostic audit findings into a self-contained, interactive HTML scorecard.
    """

    def __init__(
        self,
        diagnosis: DiagnosisReport,
        records: list[dict[str, Any]] | None = None,
        title: str = "Robot Policy Reliability Audit",
        env_name: str = "Simulation Task",
    ) -> None:
        self.diagnosis = diagnosis
        self.records = records or []
        self.title = title
        self.env_name = env_name

    def render(self) -> str:
        template_path = Path(__file__).parent / "_template.html"
        with open(template_path, "r", encoding="utf-8") as f:
            template_content = f.read()

        # Format chart payloads
        charts: list[dict[str, Any]] = []
        for s in self.diagnosis.fault_sensitivities:
            charts.append({
                "name": s.fault_name,
                "cliff_detected": s.cliff_detected,
                "labels": [str(v) for v in s.tested_values],
                "data": [round(r * 100.0, 1) for r in s.success_rates],
            })

        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        template = Template(template_content)
        return template.render(
            title=self.title,
            env_name=self.env_name,
            timestamp=timestamp,
            report=self.diagnosis,
            charts=charts,
        )

    def save(self, output_path: str | Path) -> Path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        html_content = self.render()
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        return out_file
