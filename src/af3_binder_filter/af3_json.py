"""AlphaFold 3 input JSON generation."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from af3_binder_filter.config import DEFAULT_COMPLEX_JOB_NAME_TEMPLATE
from af3_binder_filter.io_utils import atomic_write_json
from af3_binder_filter.models import BinderCsvRow


@dataclass(frozen=True)
class TargetFeatures:
    """Externalized target MSA/template references for AF3 chain A."""

    unpaired_msa_path: str | None = None
    paired_msa_path: str | None = None
    templates: list[dict[str, Any]] = field(default_factory=list)


def sanitize_job_name(value: str) -> str:
    """Make a deterministic filesystem-safe AF3 job name."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.-")
    return cleaned or "af3_job"


def format_job_name(row: BinderCsvRow, template: str) -> str:
    """Format and sanitize a job name from a CSV row."""

    name = template.format(
        sample_no=row.sample_no,
        run_name=row.run_name,
        source_row_number=row.source_row_number,
    )
    return sanitize_job_name(name)


def _protein(
    *,
    chain_id: str | list[str],
    sequence: str,
    templates: list[dict[str, Any]] | None = None,
    unpaired_msa_path: str | None = None,
    paired_msa_path: str | None = None,
    include_modifications: bool = True,
    include_templates: bool = True,
) -> dict[str, dict[str, Any]]:
    protein: dict[str, Any] = {
        "id": chain_id,
        "sequence": sequence,
    }
    if include_modifications:
        protein["modifications"] = []
    if include_templates:
        protein["templates"] = templates or []
    if unpaired_msa_path:
        protein["unpairedMsaPath"] = unpaired_msa_path
    if paired_msa_path:
        protein["pairedMsaPath"] = paired_msa_path
    return {"protein": protein}


def make_target_input(
    *,
    name: str,
    target_sequence: str,
    target_chain: str = "A",
    seed: int = 42,
) -> dict[str, Any]:
    """Build target-only AF3 input used to obtain target MSA/template data.

    The target pre-run intentionally uses AF3 v1/minimal JSON so AF3 computes
    MSA/template features for the target chain.
    """

    return {
        "name": sanitize_job_name(name),
        "sequences": [
            _protein(
                chain_id=[target_chain],
                sequence=target_sequence,
                include_modifications=False,
                include_templates=False,
            )
        ],
        "modelSeeds": [seed],
        "dialect": "alphafold3",
        "version": 1,
    }


def make_complex_input(
    row: BinderCsvRow,
    *,
    target_features: TargetFeatures | None = None,
    job_name_template: str = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE,
    target_chain: str = "A",
    binder_chain: str = "B",
    seed: int = 42,
) -> dict[str, Any]:
    """Build one binder-target complex AF3 input object."""

    features = target_features or TargetFeatures()
    return {
        "dialect": "alphafold3",
        "version": 4,
        "name": format_job_name(row, job_name_template),
        "modelSeeds": [seed],
        "sequences": [
            _protein(
                chain_id=target_chain,
                sequence=row.target_seq,
                templates=features.templates,
                unpaired_msa_path=features.unpaired_msa_path,
                paired_msa_path=features.paired_msa_path,
            ),
            _protein(
                chain_id=binder_chain,
                sequence=row.binder_sequence,
                templates=[],
            ),
        ],
    }


def write_af3_input(payload: Any, output_path: Path, *, force: bool = False) -> Path:
    """Write one AF3 input object to JSON."""

    if output_path.exists() and not force:
        return output_path
    data = (
        payload.model_dump(mode="json", exclude_none=True)
        if hasattr(payload, "model_dump")
        else payload
    )
    atomic_write_json(output_path, data)
    return output_path


def write_target_input(
    *,
    target_sequence: str,
    output_dir: Path,
    name: str = "target_A",
    target_chain: str = "A",
    seed: int = 42,
    force: bool = False,
) -> Path:
    payload = make_target_input(
        name=name,
        target_sequence=target_sequence,
        target_chain=target_chain,
        seed=seed,
    )
    return write_af3_input(payload, output_dir / f"{payload['name']}.json", force=force)


def write_complex_inputs(
    rows: Iterable[BinderCsvRow],
    output_dir: Path,
    *,
    target_features: TargetFeatures | None = None,
    job_name_template: str = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE,
    target_chain: str = "A",
    binder_chain: str = "B",
    seed: int = 42,
    force: bool = False,
    clean_stale: bool = False,
) -> list[Path]:
    """Write one complex JSON per CSV row."""

    written: list[Path] = []
    for row in rows:
        payload = make_complex_input(
            row,
            target_features=target_features,
            job_name_template=job_name_template,
            target_chain=target_chain,
            binder_chain=binder_chain,
            seed=seed,
        )
        written.append(
            write_af3_input(payload, output_dir / f"{payload['name']}.json", force=force)
        )
    if clean_stale:
        current_paths = set(written)
        for old_json in output_dir.glob("*.json"):
            if old_json not in current_paths:
                old_json.unlink()
    return written


def load_af3_input(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a single AF3 job object, not a top-level array")
    return payload


def get_chain_sequence(payload: dict[str, Any], chain_id: str) -> str:
    for sequence_entry in payload.get("sequences", []):
        protein = sequence_entry.get("protein", {})
        protein_id = protein.get("id")
        if protein_id == chain_id or (isinstance(protein_id, list) and chain_id in protein_id):
            return str(protein["sequence"])
    raise KeyError(f"chain {chain_id!r} not found in AF3 input {payload.get('name', '<unknown>')}")


def copy_relative_file(
    source: Path, destination_root: Path, relative_path: str, *, force: bool
) -> str:
    """Copy a source file under destination_root/relative_path and return the relative path."""

    destination = destination_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not destination.exists():
        shutil.copy2(source, destination)
    return relative_path
