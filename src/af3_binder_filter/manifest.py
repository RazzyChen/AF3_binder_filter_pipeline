"""Run/job manifests and strict output adoption rules."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from af3_binder_filter.io_utils import atomic_write_json
from af3_binder_filter.jobs import JobPlan, JobSpec
from af3_binder_filter.output_layout import OUTPUT_SCHEMA_VERSION


MANIFEST_VERSION = 3
JOB_MANIFEST_NAME = ".aerith_manifest.json"


@dataclass
class RunManifest:
    run_id: str
    fingerprint: str
    backend: str
    model: str
    source_csv: str
    target_sequence_sha256: str
    job_fingerprints: dict[str, str]
    secondary_backend: str = "none"
    secondary_model: str = "none"
    primary_image_id: str | None = None
    secondary_image_id: str | None = None
    feature_fingerprint: str | None = None
    status: str = "running"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    stage_status: dict[str, str] = field(default_factory=dict)
    gpu_assignments: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    output_schema_version: int = OUTPUT_SCHEMA_VERSION
    version: int = MANIFEST_VERSION

    def write(self, path: Path) -> None:
        self.updated_at = datetime.now(UTC).isoformat()
        atomic_write_json(path, asdict(self))


def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def output_reusable(
    output_dir: Path,
    *,
    expected_fingerprint: str,
    parse_output: Callable[[Path], Any],
    adopt_legacy: bool = False,
    validate_legacy: Callable[[Path], bool] | None = None,
) -> bool:
    """Return true only for fingerprint-matched, parseable output.

    Results predating Aerith manifests are rejected by default.  Explicit
    adoption still requires both structural/input validation and successful
    output parsing; adoption writes no metadata by itself.
    """

    manifest = load_manifest(output_dir / JOB_MANIFEST_NAME)
    if manifest is None:
        if not adopt_legacy or validate_legacy is None or not validate_legacy(output_dir):
            return False
    elif manifest.get("fingerprint") != expected_fingerprint:
        return False
    try:
        parsed = parse_output(output_dir)
    except Exception:
        return False
    return parsed is not None and getattr(parsed, "status", "success") == "success"


def write_job_manifest(
    output_dir: Path,
    *,
    job: JobSpec,
    fingerprint: str,
    backend: str,
    artifacts: dict[str, str | None],
) -> None:
    atomic_write_json(
        output_dir / JOB_MANIFEST_NAME,
        {
            "version": MANIFEST_VERSION,
            "job_id": job.job_id,
            "fingerprint": fingerprint,
            "backend": backend,
            "target_chain": job.target_chain,
            "binder_chain": job.binder_chain,
            "target_sequence": job.target_sequence,
            "binder_sequence": job.binder_sequence,
            "artifacts": artifacts,
        },
    )


def validate_legacy_input(
    input_json: Path,
    job: JobSpec,
    *,
    structure_validator: Callable[[Path, str, str], bool],
    structure_path: Path,
) -> bool:
    """Validate sequences/chains before explicitly adopting a legacy output."""

    try:
        payload = json.loads(input_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            payload = next(item for item in payload if item.get("name") == job.job_id)
        sequences: dict[str, str] = {}
        for entry in payload.get("sequences", []):
            protein = entry.get("protein") or entry.get("proteinChain") or {}
            chain = protein.get("id")
            if chain:
                sequences[str(chain)] = str(protein.get("sequence", ""))
        # Protenix/OpenDDE do not encode chain IDs: sequence order is A then B.
        if not sequences:
            chains = [
                (entry.get("proteinChain") or {}).get("sequence")
                for entry in payload.get("sequences", [])
                if entry.get("proteinChain")
            ]
            if len(chains) >= 2:
                sequences = {job.target_chain: chains[0], job.binder_chain: chains[1]}
        return (
            sequences.get(job.target_chain) == job.target_sequence
            and sequences.get(job.binder_chain) == job.binder_sequence
            and structure_validator(structure_path, job.target_chain, job.binder_chain)
        )
    except (OSError, ValueError, KeyError, StopIteration, json.JSONDecodeError):
        return False
