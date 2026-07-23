"""Stable on-disk layout for Aerith result runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


OUTPUT_SCHEMA_VERSION = 3

STAGE_DIRECTORIES: dict[str, str] = {
    "preflight": "01_preflight",
    "features": "02_features",
    "primary_prediction": "03_primary_prediction",
    "primary_interface": "04_primary_interface",
    "esm": "05_esm",
    "secondary_features": "06_secondary_features",
    "secondary_prediction": "07_secondary_prediction",
    "secondary_interface": "08_secondary_interface",
    "consensus": "09_consensus",
    "clustering": "10_clustering",
}

RUNTIME_STAGE_ALIASES: dict[str, str] = {
    "prediction": "primary_prediction",
    "target_features": "features",
    "esmfold": "esm",
    "esm_if": "esm",
    "foldseek": "clustering",
    "foldseek-binder": "clustering",
    "foldseek-complex": "clustering",
    "interface": "primary_interface",
}


@dataclass(frozen=True, slots=True)
class StageLayout:
    root: Path

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def tables(self) -> Path:
        return self.root / "tables"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    def ensure(self) -> "StageLayout":
        for path in (self.logs, self.tables, self.artifacts):
            path.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True, slots=True)
class RunOutputLayout:
    root: Path

    @property
    def stages_root(self) -> Path:
        return self.root / "stages"

    @property
    def all_results(self) -> Path:
        return self.root / "all_results.csv"

    @property
    def candidates(self) -> Path:
        return self.root / "candidates.csv"

    @property
    def final_shortlist(self) -> Path:
        return self.root / "final_shortlist.csv"

    @property
    def backend_review(self) -> Path:
        return self.root / "backend_review.csv"

    def canonical_stage_name(self, name: str) -> str:
        canonical = RUNTIME_STAGE_ALIASES.get(name, name)
        if canonical not in STAGE_DIRECTORIES:
            raise KeyError(f"unknown Aerith output stage: {name}")
        return canonical

    def stage(self, name: str) -> StageLayout:
        canonical = self.canonical_stage_name(name)
        return StageLayout(self.stages_root / STAGE_DIRECTORIES[canonical]).ensure()

    def ensure(self) -> "RunOutputLayout":
        self.root.mkdir(parents=True, exist_ok=True)
        for name in STAGE_DIRECTORIES:
            self.stage(name)
        return self
