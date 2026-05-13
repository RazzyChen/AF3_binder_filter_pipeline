"""Shared Pydantic models used by the pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class JobState(StrEnum):
    pending = "pending"
    running = "running"
    success = "success"
    missing = "missing"
    error = "error"
    skipped = "skipped"


class ExternalCommand(BaseModel):
    """External command invocation record."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    argv: list[str]
    cwd: Path | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None


class JobStatus(BaseModel):
    """Small resumability/status record for one job."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    job_name: str
    state: JobState
    input_json_path: Path | None = None
    output_dir: Path | None = None
    command: ExternalCommand | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    return_code: int | None = None
    error: str | None = None


class BinderCsvRow(BaseModel):
    """Validated binder design row from the input CSV."""

    model_config = ConfigDict(extra="forbid")

    sample_no: str
    run_name: str
    binder_sequence: str
    target_seq: str
    source_row_number: int = Field(ge=2)

    @field_validator("sample_no", "run_name")
    @classmethod
    def nonempty_text(cls, value: str, info: Any) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @field_validator("binder_sequence", "target_seq")
    @classmethod
    def normalize_sequence(cls, value: str, info: Any) -> str:
        sequence = "".join(str(value).split()).upper()
        if not sequence:
            raise ValueError(f"{info.field_name} must not be empty")
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = sorted(set(sequence) - allowed)
        if invalid:
            raise ValueError(
                f"{info.field_name} contains unsupported amino-acid letters: {''.join(invalid)}"
            )
        return sequence


class ProteinSequence(BaseModel):
    """AF3 protein sequence object."""

    model_config = ConfigDict(extra="forbid")

    id: str
    sequence: str
    modifications: list[dict[str, Any]] = Field(default_factory=list)
    templates: list[dict[str, Any]] = Field(default_factory=list)
    unpairedMsa: str | None = None
    pairedMsa: str | None = None
    unpairedMsaPath: str | None = None
    pairedMsaPath: str | None = None


class AF3Input(BaseModel):
    """Single AlphaFold 3 input object."""

    model_config = ConfigDict(extra="forbid")

    dialect: str = "alphafold3"
    version: int = 4
    name: str
    modelSeeds: list[int] = Field(default_factory=lambda: [42])
    sequences: list[dict[str, ProteinSequence]]


class AF3Metrics(BaseModel):
    """Aggregated AF3 metrics for one job."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    job_name: str
    job_status: JobState | str
    ranking_score: float | None = None
    iptm: float | None = None
    ptm: float | None = None
    fraction_disordered: float | None = None
    has_clash: float | None = None
    plddt_global_mean: float | None = None
    plddt_global_min: float | None = None
    plddt_chain_A_mean: float | None = None
    plddt_chain_A_min: float | None = None
    plddt_chain_B_mean: float | None = None
    plddt_chain_B_min: float | None = None
    ipae_A_to_B_mean: float | None = None
    ipae_A_to_B_min: float | None = None
    ipae_B_to_A_mean: float | None = None
    ipae_B_to_A_min: float | None = None
    sasa_status: JobState | str | None = None
    sasa_target_chain: float | None = None
    sasa_binder_chain: float | None = None
    sasa_target_free: float | None = None
    sasa_binder_free: float | None = None
    dsasa_target: float | None = None
    dsasa_binder: float | None = None
    dsasa: float | None = None
    dsasa_interface: float | None = None
    sasa_error: str | None = None


class ESMScoreMetrics(BaseModel):
    """ESM inverse folding metrics for one job."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    job_name: str
    esm_score_status: JobState | str
    esm_log_likelihood: float | None = None
    esm_perplexity: float | None = None
    esm_fasta_path: Path | str | None = None
    esm_score_csv: Path | str | None = None
    esm_error: str | None = None


class IPSAEScoreMetrics(BaseModel):
    """ipSAE-family metrics for one job."""

    model_config = ConfigDict(extra="allow")

    job_name: str
    ipsae_score_status: JobState | str
    ipSAE_A_to_B: float | None = None
    ipSAE_B_to_A: float | None = None
    ipSAE_max: float | None = None
    pDockQ_A_to_B: float | None = None
    pDockQ_B_to_A: float | None = None
    pDockQ_max: float | None = None
    pDockQ2_A_to_B: float | None = None
    pDockQ2_B_to_A: float | None = None
    pDockQ2_max: float | None = None
    LIS_A_to_B: float | None = None
    LIS_B_to_A: float | None = None
    LIS_max: float | None = None
    ipsae_error: str | None = None
