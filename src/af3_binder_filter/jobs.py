"""Immutable job planning and run fingerprinting."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from af3_binder_filter.af3_json import format_job_name
from af3_binder_filter.config import AerithConfig
from af3_binder_filter.csv_input import CsvInputError, read_binder_csv
from af3_binder_filter.models import BinderCsvRow


@dataclass(frozen=True, slots=True)
class JobSpec:
    """One immutable binder/target prediction job shared by all stages."""

    job_id: str
    sample_no: str
    run_name: str
    target_sequence: str
    binder_sequence: str
    target_chain: str
    binder_chain: str
    source_row_number: int
    seed: int
    backend: str
    model: str


@dataclass(frozen=True, slots=True)
class JobPlan:
    jobs: tuple[JobSpec, ...]
    target_sequence: str
    source_csv: Path
    total_csv_jobs: int


def parse_epitope_residues(value: str | None, *, target_length: int) -> frozenset[int]:
    """Parse comma-separated 1-based residues and inclusive ranges."""

    if value is None or not str(value).strip():
        return frozenset()
    residues: set[int] = set()
    for raw_part in str(value).split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError("epitope_residues contains an empty item")
        try:
            if "-" in part:
                pieces = part.split("-")
                if len(pieces) != 2:
                    raise ValueError
                start, end = (int(piece.strip()) for piece in pieces)
                if start > end:
                    raise ValueError
                residues.update(range(start, end + 1))
            else:
                residues.add(int(part))
        except ValueError as exc:
            raise ValueError(f"invalid epitope residue expression: {part!r}") from exc
    invalid = sorted(residue for residue in residues if residue < 1 or residue > target_length)
    if invalid:
        raise ValueError(
            f"epitope residues outside target sequence range 1-{target_length}: "
            + ",".join(str(value) for value in invalid)
        )
    return frozenset(residues)


def build_job_plan_from_rows(rows: Sequence[BinderCsvRow], config: AerithConfig) -> JobPlan:
    if not rows:
        raise CsvInputError("CSV has no binder rows")
    target_sequences = {row.target_seq for row in rows}
    if len(target_sequences) != 1:
        details = sorted((row.source_row_number, row.target_seq) for row in rows)
        preview = ", ".join(f"row {number}" for number, _ in details[:8])
        raise CsvInputError(f"all jobs in one run must share one target sequence; mismatch at {preview}")
    if not config.project.target_chain.strip() or not config.project.binder_chain.strip():
        raise CsvInputError("target and binder chain IDs must be non-empty")
    if config.project.target_chain == config.project.binder_chain:
        raise CsvInputError("target and binder chain IDs must be different")

    target_sequence = next(iter(target_sequences))
    parse_epitope_residues(
        config.interface.epitope_residues,
        target_length=len(target_sequence),
    )

    all_jobs: list[JobSpec] = []
    seen: dict[str, int] = {}
    for row in rows:
        job_id = format_job_name(row, config.project.job_name_template)
        if job_id in seen:
            raise CsvInputError(
                f"duplicate sanitized job name {job_id!r} from CSV rows "
                f"{seen[job_id]} and {row.source_row_number}"
            )
        seen[job_id] = row.source_row_number
        all_jobs.append(
            JobSpec(
                job_id=job_id,
                sample_no=row.sample_no,
                run_name=row.run_name,
                target_sequence=row.target_seq,
                binder_sequence=row.binder_sequence,
                target_chain=config.project.target_chain,
                binder_chain=config.project.binder_chain,
                source_row_number=row.source_row_number,
                seed=config.project.seed,
                backend=config.backend.name,
                model=config.backend.model,
            )
        )
    limit = config.project.limit
    selected = all_jobs if limit is None else all_jobs[:limit]
    return JobPlan(tuple(selected), target_sequence, Path(config.project.csv_path), len(all_jobs))


def build_job_plan(config: AerithConfig) -> JobPlan:
    """Parse the project CSV exactly once and freeze the complete run plan."""

    rows = read_binder_csv(Path(config.project.csv_path))
    return build_job_plan_from_rows(rows, config)


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_identity(value: str | None) -> dict[str, str | int] | None:
    """Return a cheap identity that changes when a selected checkpoint changes."""

    if not value:
        return None
    path = Path(value).expanduser().resolve()
    identity: dict[str, str | int] = {"path": str(path)}
    try:
        stat = path.stat()
    except OSError:
        return identity
    identity.update({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    return identity


def job_fingerprint(
    job: JobSpec,
    config: AerithConfig,
    *,
    feature_fingerprint: str | None = None,
) -> str:
    return _canonical_digest(
        {
            "job": asdict(job),
            "backend": config.backend.name,
            "model": config.backend.model,
            "image": config.backend.image,
            "image_id": config.backend.image_id,
            "source_commit": config.backend.source_commit,
            "checkpoint": checkpoint_identity(config.backend.checkpoint_path),
            "secondary_backend": config.secondary_backend.name,
            "secondary_model": config.secondary_backend.model,
            "secondary_image": config.secondary_backend.image,
            "secondary_image_id": config.secondary_backend.image_id,
            "secondary_source_commit": config.secondary_backend.source_commit,
            "secondary_checkpoint": checkpoint_identity(
                config.secondary_backend.checkpoint_path
            ),
            "secondary_minimum_primary_iptm": config.secondary_backend.minimum_primary_iptm,
            "feature_mode": config.features.name,
            "feature_image": config.features.image,
            "feature_fingerprint": feature_fingerprint,
            "esm": asdict(config.scoring.esm),
            "consensus": asdict(config.consensus),
        }
    )


def run_fingerprint(
    plan: JobPlan,
    config: AerithConfig,
    *,
    feature_fingerprint: str | None = None,
) -> str:
    return _canonical_digest(
        {
            "source_csv": str(plan.source_csv),
            "jobs": [job_fingerprint(job, config, feature_fingerprint=feature_fingerprint) for job in plan.jobs],
            "target_sequence": plan.target_sequence,
            "target_chain": config.project.target_chain,
            "binder_chain": config.project.binder_chain,
        }
    )


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()
