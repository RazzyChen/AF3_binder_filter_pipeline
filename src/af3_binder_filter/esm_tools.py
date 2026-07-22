"""Containerized ESMFold/ESM-IF stage contracts and result parsing."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.config import AerithConfig
from af3_binder_filter.esmfold_score import parse_esmfold_plddt
from af3_binder_filter.io_utils import atomic_write_json, atomic_write_text
from af3_binder_filter.jobs import JobSpec


def write_esm_inputs(
    jobs: Sequence[JobSpec],
    predictions: Sequence[UnifiedPrediction],
    input_dir: Path,
) -> tuple[Path, Path]:
    input_dir.mkdir(parents=True, exist_ok=True)
    fasta = input_dir / "binders.fasta"
    atomic_write_text(
        fasta,
        "".join(f">{job.job_id}_chain_{job.binder_chain}\n{job.binder_sequence}\n" for job in jobs),
    )
    prediction_by_job = {prediction.job_id: prediction for prediction in predictions}
    manifest = input_dir / "esm_if_jobs.json"
    atomic_write_json(
        manifest,
        [
            {
                "job_name": job.job_id,
                "chain_id": job.binder_chain,
                "structure_path": str(prediction_by_job[job.job_id].best_model_path),
            }
            for job in jobs
            if prediction_by_job[job.job_id].status == "success"
            and prediction_by_job[job.job_id].best_model_path is not None
        ],
    )
    return fasta, manifest


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
        )
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


def _chain_ca(path: Path, chain_id: str | None = None) -> np.ndarray:
    import biotite.structure as struc
    import biotite.structure.io as strucio

    array = strucio.load_structure(str(path), model=1)
    if getattr(array, "stack_depth", lambda: 1)() > 1:
        array = array[0]
    if chain_id is not None and chain_id in set(array.chain_id.tolist()):
        array = array[array.chain_id == chain_id]
    starts = struc.get_residue_starts(array, add_exclusive_stop=True)
    result = []
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        residue = array[start:stop]
        ca = residue[residue.atom_name == "CA"]
        result.append(ca.coord[0] if len(ca) else residue.coord.mean(axis=0))
    return np.asarray(result, dtype=float)


def _fold_comparison(first: np.ndarray, second: np.ndarray) -> tuple[float | None, float | None]:
    count = min(len(first), len(second))
    if count < 3:
        return None, None
    first, second = first[:count], second[:count]
    first_center, second_center = first.mean(axis=0), second.mean(axis=0)
    left, _values, right = np.linalg.svd(
        (second - second_center).T @ (first - first_center)
    )
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    aligned = (second - second_center) @ rotation + first_center
    distances = np.linalg.norm(first - aligned, axis=1)
    rmsd = float(np.sqrt(np.mean(distances**2)))
    d0 = max(0.5, 1.24 * max(count - 15, 1) ** (1 / 3) - 1.8)
    tm = float(np.mean(1.0 / (1.0 + (distances / d0) ** 2)))
    return rmsd, tm


def collect_esm_rows(
    jobs: Sequence[JobSpec],
    predictions: Sequence[UnifiedPrediction],
    output_dir: Path,
) -> list[dict[str, Any]]:
    inverse = _read_csv(output_dir / "esm_if.csv")
    by_prediction = {prediction.job_id: prediction for prediction in predictions}
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
        if pdb and prediction.best_model_path:
            try:
                rmsd, tm = _fold_comparison(
                    _chain_ca(prediction.best_model_path, job.binder_chain),
                    _chain_ca(pdb),
                )
                row["esmfold_af3_binder_rmsd"] = rmsd
                row["esmfold_af3_binder_tm"] = tm
            except Exception as exc:
                row["esmfold_comparison_error"] = str(exc)
        rows.append(row)
    return rows
