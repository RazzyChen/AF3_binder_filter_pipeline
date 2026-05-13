"""Pipeline orchestration and preflight checks."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from af3_binder_filter.config import PipelineConfig
from af3_binder_filter.csv_input import read_binder_csv
from af3_binder_filter.gpu import GPUError, query_gpus


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
    """Validate critical inputs and external tools before launching AF3."""

    report = PreflightReport()

    try:
        rows = read_binder_csv(config.csv_path)
        report.info.append(f"CSV OK: {config.csv_path} ({len(rows)} jobs)")
    except Exception as exc:  # noqa: BLE001
        report.errors.append(str(exc))

    for path, label in (
        (config.af3.model_dir, "AF3 model dir"),
        (config.af3.database_dir, "AF3 database dir"),
        (config.af3.jax_cache_dir, "JAX cache dir"),
        (config.esm.scorer_path, "ESM scorer"),
    ):
        if path.exists():
            report.info.append(f"{label} OK: {path}")
        else:
            report.errors.append(f"{label} does not exist: {path}")

    if shutil.which(config.af3.docker_bin):
        report.info.append(f"Docker OK: {config.af3.docker_bin}")
    else:
        report.errors.append(f"Docker executable not found: {config.af3.docker_bin}")

    if shutil.which(config.esm.conda_bin):
        report.info.append(f"Conda OK: {config.esm.conda_bin}")
    else:
        report.errors.append(f"Conda executable not found: {config.esm.conda_bin}")

    try:
        gpus = query_gpus()
        report.info.append(f"nvidia-smi OK: {len(gpus)} GPUs visible")
    except GPUError as exc:
        report.errors.append(str(exc))

    return report


def expected_target_data_json(config: PipelineConfig, *, target_name: str = "target_A") -> Path:
    return config.output_dir / target_name / f"{target_name}_data.json"
