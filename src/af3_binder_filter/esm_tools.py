"""Containerized ESMFold/ESM-IF stage contracts and result parsing."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.config import AerithConfig
from af3_binder_filter.derived_structures import (
    DerivedStructureArtifacts,
    file_sha256,
    row_job_identifier,
    row_structure_is_eligible,
    validated_artifacts_from_row,
)
from af3_binder_filter.esmfold_score import parse_esmfold_plddt
from af3_binder_filter.io_utils import atomic_write_json, atomic_write_text
from af3_binder_filter.jobs import JobSpec


def write_esm_inputs(
    jobs: Sequence[JobSpec],
    predictions: Sequence[UnifiedPrediction],
    input_dir: Path,
    *,
    structure_rows: Sequence[Mapping[str, Any]] = (),
    structure_prefix: str = "effective",
) -> tuple[Path, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    fasta = input_dir / "binders.fasta"
    atomic_write_text(
        fasta,
        "".join(f">{job.job_id}_chain_{job.binder_chain}\n{job.binder_sequence}\n" for job in jobs),
    )
    prediction_by_job = {prediction.job_id: prediction for prediction in predictions}
    derived_by_job, binding_by_job = _structure_context_by_job(
        structure_rows,
        prefix=structure_prefix,
    )
    blocked_job_ids = _blocked_structure_job_ids(
        structure_rows,
        prefix=structure_prefix,
    )
    staged_paths: dict[str, str] = {}
    for job in jobs:
        derived = derived_by_job.get(job.job_id)
        if derived is None:
            continue
        destination = input_dir / "derived_structures" / f"{job.job_id}.pdb"
        _atomic_copy(derived.binder_pdb, destination)
        staged_paths[job.job_id] = f"/inputs/derived_structures/{job.job_id}.pdb"
    manifest = input_dir / "esm_if_jobs.json"
    bindings = {
        job.job_id: (
            None
            if job.job_id in blocked_job_ids
            else binding_by_job.get(job.job_id)
            or _prediction_binding(prediction_by_job[job.job_id])
        )
        for job in jobs
    }
    atomic_write_json(
        manifest,
        [
            {
                "job_name": job.job_id,
                "chain_id": job.binder_chain,
                "structure_path": staged_paths.get(
                    job.job_id,
                    str(prediction_by_job[job.job_id].best_model_path),
                ),
                "effective_backend": (
                    bindings[job.job_id].backend
                    if bindings[job.job_id] is not None
                    else prediction_by_job[job.job_id].backend
                ),
                "derived_structure_id": (
                    bindings[job.job_id].derived_structure_id
                    if bindings[job.job_id] is not None
                    else None
                ),
                "source_model_sha256": (
                    bindings[job.job_id].source_model_sha256
                    if bindings[job.job_id] is not None
                    else None
                ),
            }
            for job in jobs
            if prediction_by_job[job.job_id].status == "success"
            and prediction_by_job[job.job_id].best_model_path is not None
            and job.job_id not in blocked_job_ids
        ],
    )
    return fasta, manifest


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=destination.suffix,
        dir=destination.parent,
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copy2(source, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class _ESMStructureBinding:
    backend: str
    derived_structure_id: str
    source_model_sha256: str


def _structure_context_by_job(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> tuple[
    dict[str, DerivedStructureArtifacts],
    dict[str, _ESMStructureBinding],
]:
    derived: dict[str, DerivedStructureArtifacts] = {}
    bindings: dict[str, _ESMStructureBinding] = {}
    for row in rows:
        job_id = row_job_identifier(row)
        if job_id is None:
            continue
        if prefix and f"{prefix}_derived_structure_status" not in row:
            # A merged row's unprefixed fields describe the primary backend.
            # Never reinterpret them as an absent secondary/effective backend.
            continue
        artifacts = validated_artifacts_from_row(
            row,
            prefix=prefix,
            require_declared=True,
        )
        if job_id and artifacts is not None:
            derived[job_id] = artifacts
            bindings[job_id] = _ESMStructureBinding(
                backend=artifacts.backend,
                derived_structure_id=artifacts.content_id,
                source_model_sha256=artifacts.source_model_sha256,
            )
    return derived, bindings


def _blocked_structure_job_ids(
    rows: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
) -> set[str]:
    blocked: set[str] = set()
    for row in rows:
        job_id = row_job_identifier(row)
        status_key = f"{prefix}_interface_status" if prefix else "interface_status"
        has_declared_status = status_key in row or (
            bool(prefix) and "interface_status" in row
        )
        if (
            job_id is not None
            and has_declared_status
            and not row_structure_is_eligible(
                row,
                prefix=prefix,
            )
        ):
            blocked.add(job_id)
    return blocked


def _prediction_binding(
    prediction: UnifiedPrediction,
) -> _ESMStructureBinding | None:
    path = prediction.best_model_path
    if prediction.status != "success" or path is None or not path.is_file():
        return None
    return _ESMStructureBinding(
        backend=prediction.backend,
        derived_structure_id="",
        source_model_sha256=file_sha256(path),
    )


def _binding_fields(
    label: str,
    binding: _ESMStructureBinding | None,
) -> dict[str, str]:
    prefix = f"esm_{label}"
    return {
        f"{prefix}_backend": binding.backend if binding is not None else "",
        f"{prefix}_derived_structure_id": (
            binding.derived_structure_id if binding is not None else ""
        ),
        f"{prefix}_source_model_sha256": (
            binding.source_model_sha256 if binding is not None else ""
        ),
    }


def _binding_matches(
    row: Mapping[str, Any],
    label: str,
    binding: _ESMStructureBinding | None,
) -> bool:
    return all(
        str(row.get(field) or "") == expected
        for field, expected in _binding_fields(label, binding).items()
    )


def _container_base(
    config: AerithConfig,
    *,
    gpu_index: int,
    inputs: Path,
    outputs: Path,
    container_name: str | None = None,
) -> list[str]:
    cache = Path(config.scoring.esm.model_cache).expanduser().resolve()
    command = [
        config.backend.docker_bin,
        "run",
        "--rm",
    ]
    if container_name:
        command.extend(["--name", container_name])
    command.extend([
        "--network",
        "none",
        "--gpus",
        f"device={gpu_index}",
        "--shm-size",
        "4g",
        "--volume",
        f"{inputs.resolve()}:/inputs:ro",
        "--volume",
        f"{outputs.resolve()}:/outputs",
        "--volume",
        f"{cache}:/root/.cache/torch/hub/checkpoints:ro",
        config.backend.image,
    ])
    return command


def build_esmfold_container_command(
    config: AerithConfig,
    *,
    input_dir: Path,
    output_dir: Path,
    gpu_index: int,
    container_name: str | None = None,
) -> list[str]:
    return _container_base(
        config,
        gpu_index=gpu_index,
        inputs=input_dir,
        outputs=output_dir,
        container_name=container_name,
    ) + [
        config.scoring.esm.runtime_entry_fold,
        "-i",
        "/inputs/binders.fasta",
        "-o",
        "/outputs/esmfold",
    ]


def build_esm_if_container_command(
    config: AerithConfig,
    *,
    input_dir: Path,
    output_dir: Path,
    prediction_output_dir: Path,
    gpu_index: int,
    container_name: str | None = None,
) -> list[str]:
    command = _container_base(
        config,
        gpu_index=gpu_index,
        inputs=input_dir,
        outputs=output_dir,
        container_name=container_name,
    )
    prediction_root = prediction_output_dir.resolve()
    command[-1:-1] = ["--volume", f"{prediction_root}:{prediction_root}:ro"]
    return command + [
        config.scoring.esm.runtime_entry_if,
        "--manifest",
        "/inputs/esm_if_jobs.json",
        "--output",
        "/outputs/esm_if.csv",
    ]


def _read_csv(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {str(row["job_name"]): dict(row) for row in csv.DictReader(handle)}


def _finite_number(row: dict[str, Any], field: str) -> bool:
    try:
        return math.isfinite(float(row.get(field)))
    except (TypeError, ValueError):
        return False


def load_cached_esm_rows(
    path: Path,
    jobs: Sequence[JobSpec],
    predictions: Sequence[UnifiedPrediction],
    *,
    require_esmfold: bool,
    require_inverse_folding: bool,
    structure_rows: Sequence[Mapping[str, Any]] = (),
    structure_prefix: str = "effective",
    primary_predictions: Sequence[UnifiedPrediction] = (),
    secondary_predictions: Sequence[UnifiedPrediction] = (),
) -> list[dict[str, Any]] | None:
    """Return a complete, parseable same-run ESM table or ``None``."""

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    except (OSError, csv.Error, KeyError):
        return None
    expected = [job.job_id for job in jobs]
    identifiers = [str(row.get("job_name", "")) for row in rows]
    if (
        len(rows) != len(expected)
        or len(set(identifiers)) != len(identifiers)
        or set(identifiers) != set(expected)
    ):
        return None
    cached = {str(row["job_name"]): row for row in rows}
    prediction_by_job = {prediction.job_id: prediction for prediction in predictions}
    _, binding_by_job = _structure_context_by_job(
        structure_rows,
        prefix=structure_prefix,
    )
    blocked_job_ids = _blocked_structure_job_ids(
        structure_rows,
        prefix=structure_prefix,
    )
    comparison_bindings: dict[str, tuple[
        dict[str, UnifiedPrediction],
        dict[str, _ESMStructureBinding],
        set[str],
    ]] = {}
    for label, comparison_predictions in (
        ("primary", primary_predictions),
        ("secondary", secondary_predictions),
    ):
        if not comparison_predictions:
            continue
        _, derived_bindings = _structure_context_by_job(
            structure_rows,
            prefix=label,
        )
        comparison_bindings[label] = (
            {
                prediction.job_id: prediction
                for prediction in comparison_predictions
            },
            derived_bindings,
            _blocked_structure_job_ids(structure_rows, prefix=label),
        )
    for job in jobs:
        row = cached[job.job_id]
        if require_esmfold:
            model_path = Path(str(row.get("esmfold_model_path", "")))
            if (
                row.get("esmfold_status") != "success"
                or not model_path.is_file()
                or not _finite_number(row, "esmfold_plddt")
            ):
                return None
        prediction = prediction_by_job.get(job.job_id)
        inverse_expected = (
            prediction is not None
            and prediction.status == "success"
            and prediction.best_model_path is not None
            and job.job_id not in blocked_job_ids
        )
        binding = (
            None
            if job.job_id in blocked_job_ids
            else binding_by_job.get(job.job_id)
        )
        if (
            binding is None
            and prediction is not None
            and job.job_id not in blocked_job_ids
        ):
            binding = _prediction_binding(prediction)
        if not _binding_matches(row, "effective", binding):
            return None
        for label, (
            comparison_by_job,
            derived_bindings,
            comparison_blocked,
        ) in comparison_bindings.items():
            comparison_binding = (
                None
                if job.job_id in comparison_blocked
                else derived_bindings.get(job.job_id)
            )
            comparison_prediction = comparison_by_job.get(job.job_id)
            if (
                comparison_binding is None
                and comparison_prediction is not None
                and job.job_id not in comparison_blocked
            ):
                comparison_binding = _prediction_binding(comparison_prediction)
            if not _binding_matches(row, label, comparison_binding):
                return None
        if require_inverse_folding and inverse_expected:
            if row.get("esm_if_status") != "success":
                return None
            if not all(
                _finite_number(row, field)
                for field in (
                    "esm_if_log_likelihood",
                    "esm_if_log_likelihood_with_coord",
                    "esm_if_perplexity",
                )
            ):
                return None
    return [cached[job_id] for job_id in expected]


def _chain_ca(path: Path, chain_id: str | None = None) -> dict[int, np.ndarray]:
    import biotite.structure as struc
    import biotite.structure.io as strucio

    array = strucio.load_structure(str(path), model=1)
    if getattr(array, "stack_depth", lambda: 1)() > 1:
        array = array[0]
    present = {str(value) for value in array.chain_id.tolist()}
    if chain_id is not None:
        if chain_id not in present:
            raise ValueError(f"chain {chain_id!r} is absent from {path}")
        array = array[array.chain_id == chain_id]
    elif len(present) != 1:
        raise ValueError(
            f"a single unambiguous chain is required in {path}; found {sorted(present)}"
        )
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    categories = set(array.get_annotation_categories())
    result: dict[int, np.ndarray] = {}
    ordered_positions: list[int] = []
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        residue = array[start:stop]
        position = int(residue.res_id[0])
        if position <= 0 or position in result:
            raise ValueError(
                f"residue IDs in {path} must be unique positive sequence positions"
            )
        if "ins_code" in categories and any(
            str(value).strip() for value in residue.ins_code.tolist()
        ):
            raise ValueError(
                f"insertion-coded residue {position} in {path} is not normalized"
            )
        ca = residue[residue.atom_name == "CA"]
        coordinate = np.asarray(
            ca.coord[0] if len(ca) else residue.coord.mean(axis=0),
            dtype=float,
        )
        if coordinate.shape != (3,) or not np.all(np.isfinite(coordinate)):
            raise ValueError(f"invalid residue coordinates in {path}")
        result[position] = coordinate
        ordered_positions.append(position)
    if not result:
        raise ValueError(f"no residue coordinates are available in {path}")
    if ordered_positions != sorted(ordered_positions):
        raise ValueError(f"residue IDs in {path} are not monotonically increasing")
    return result


def _fold_comparison(
    first: Mapping[int, np.ndarray],
    second: Mapping[int, np.ndarray],
) -> tuple[float, float]:
    common_positions = sorted(set(first) & set(second))
    count = len(common_positions)
    if count < 3:
        raise ValueError(
            "fewer than three common normalized residue positions are available"
        )
    first_coordinates = np.asarray(
        [first[position] for position in common_positions],
        dtype=float,
    )
    second_coordinates = np.asarray(
        [second[position] for position in common_positions],
        dtype=float,
    )
    first_center = first_coordinates.mean(axis=0)
    second_center = second_coordinates.mean(axis=0)
    left, _values, right = np.linalg.svd(
        (second_coordinates - second_center).T
        @ (first_coordinates - first_center)
    )
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    aligned = (second_coordinates - second_center) @ rotation + first_center
    distances = np.linalg.norm(first_coordinates - aligned, axis=1)
    rmsd = float(np.sqrt(np.mean(distances**2)))
    d0 = max(0.5, 1.24 * max(count - 15, 1) ** (1 / 3) - 1.8)
    tm = float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))
    return rmsd, tm


def collect_esm_rows(
    jobs: Sequence[JobSpec],
    predictions: Sequence[UnifiedPrediction],
    output_dir: Path,
    *,
    comparison_label: str = "effective",
    structure_rows: Sequence[Mapping[str, Any]] = (),
    structure_prefix: str = "effective",
) -> list[dict[str, Any]]:
    inverse = _read_csv(output_dir / "esm_if.csv")
    by_prediction = {prediction.job_id: prediction for prediction in predictions}
    derived_by_job, binding_by_job = _structure_context_by_job(
        structure_rows,
        prefix=structure_prefix,
    )
    blocked_job_ids = _blocked_structure_job_ids(
        structure_rows,
        prefix=structure_prefix,
    )
    rows: list[dict[str, Any]] = []
    for job in jobs:
        header = f"{job.job_id}_chain_{job.binder_chain}"
        pdb_candidates = [
            output_dir / "esmfold" / f"{header}.pdb",
            output_dir / "esmfold" / f"{job.job_id}.pdb",
        ]
        pdb = next((path for path in pdb_candidates if path.is_file()), None)
        row: dict[str, Any] = {
            "job_name": job.job_id,
            "esmfold_status": "missing" if pdb is None else "success",
            "esmfold_model_path": str(pdb) if pdb else None,
            "esmfold_plddt": parse_esmfold_plddt(pdb) if pdb else None,
            "esm_if_status": "not_available",
        }
        row.update(inverse.get(job.job_id, {}))
        prediction = by_prediction[job.job_id]
        binding = (
            None
            if job.job_id in blocked_job_ids
            else binding_by_job.get(job.job_id)
            or _prediction_binding(prediction)
        )
        # Identity fields are controller-owned and cannot be overridden by a
        # stale or malformed runtime CSV row.
        row.update(_binding_fields("effective", binding))
        comparison_path = (
            None
            if job.job_id in blocked_job_ids
            else (
                derived_by_job[job.job_id].binder_pdb
                if job.job_id in derived_by_job
                else prediction.best_model_path
            )
        )
        if pdb and comparison_path:
            try:
                rmsd, tm = _fold_comparison(
                    _chain_ca(comparison_path, job.binder_chain),
                    _chain_ca(pdb),
                )
                row[f"esmfold_{comparison_label}_binder_rmsd"] = rmsd
                row[f"esmfold_{comparison_label}_binder_tm"] = tm
            except Exception as exc:
                row["esmfold_comparison_error"] = str(exc)
        rows.append(row)
    return rows


def add_esmfold_backend_comparison(
    rows: Sequence[dict[str, Any]],
    jobs: Sequence[JobSpec],
    predictions: Sequence[UnifiedPrediction],
    output_dir: Path,
    *,
    label: str,
    structure_rows: Sequence[Mapping[str, Any]] = (),
    structure_prefix: str | None = None,
) -> list[dict[str, Any]]:
    """Add a cheap ESMFold-to-backend fold comparison to existing ESM rows."""

    by_row = {str(row.get("job_name")): dict(row) for row in rows}
    by_prediction = {prediction.job_id: prediction for prediction in predictions}
    derived_by_job, binding_by_job = _structure_context_by_job(
        structure_rows,
        prefix=structure_prefix or label,
    )
    blocked_job_ids = _blocked_structure_job_ids(
        structure_rows,
        prefix=structure_prefix or label,
    )
    for job in jobs:
        row = by_row[job.job_id]
        header = f"{job.job_id}_chain_{job.binder_chain}"
        pdb = next(
            (
                path
                for path in (
                    output_dir / "esmfold" / f"{header}.pdb",
                    output_dir / "esmfold" / f"{job.job_id}.pdb",
                )
                if path.is_file()
            ),
            None,
        )
        prediction = by_prediction.get(job.job_id)
        binding = (
            None
            if job.job_id in blocked_job_ids
            else binding_by_job.get(job.job_id)
        )
        if (
            binding is None
            and prediction is not None
            and job.job_id not in blocked_job_ids
        ):
            binding = _prediction_binding(prediction)
        row.update(_binding_fields(label, binding))
        comparison_path = (
            None
            if job.job_id in blocked_job_ids
            else (
                derived_by_job[job.job_id].binder_pdb
                if job.job_id in derived_by_job
                else (prediction.best_model_path if prediction is not None else None)
            )
        )
        if (
            pdb is None
            or prediction is None
            or prediction.status != "success"
            or comparison_path is None
        ):
            continue
        try:
            rmsd, tm = _fold_comparison(
                _chain_ca(comparison_path, job.binder_chain),
                _chain_ca(pdb),
            )
            row[f"esmfold_{label}_binder_rmsd"] = rmsd
            row[f"esmfold_{label}_binder_tm"] = tm
        except Exception as exc:
            row[f"esmfold_{label}_comparison_error"] = str(exc)
    return [by_row[job.job_id] for job in jobs]
