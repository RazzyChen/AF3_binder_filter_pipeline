"""Pipeline orchestration and preflight checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from af3_binder_filter.config import PipelineConfig
from af3_binder_filter.csv_input import read_binder_csv


@dataclass
class PreflightReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


class PipelineError(RuntimeError):
    """Raised when orchestration cannot continue."""


def run_preflight(config: PipelineConfig) -> PreflightReport:
    """Validate only the CSV input before launching the pipeline."""

    report = PreflightReport()

    try:
        rows = read_binder_csv(config.csv_path)
        report.info.append(f"CSV OK: {config.csv_path} ({len(rows)} jobs)")
    except Exception as exc:  # noqa: BLE001
        report.errors.append(str(exc))

    return report


def expected_target_data_json(config: PipelineConfig, *, target_name: str = "target_A") -> Path:
    candidates = candidate_target_data_jsons(config, target_name=target_name)
    return next((path for path in candidates if path.exists()), candidates[-1])


def candidate_target_data_jsons(config: PipelineConfig, *, target_name: str = "target_A") -> list[Path]:
    """Return supported AF3 target data JSON locations, in preferred order."""

    return [
        config.output_dir / f"{target_name}_data.json",
        config.output_dir / target_name / f"{target_name}_data.json",
    ]
