"""Three-layer structural/pose/epitope clustering and quality representatives."""

from __future__ import annotations

import csv
import math
import os
import shutil
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from af3_binder_filter.config import ClusteringSettings
from af3_binder_filter.derived_structures import (
    row_job_identifier,
    row_structure_is_eligible,
    validated_artifacts_from_row,
)
from af3_binder_filter.interface import load_protein_complex
from af3_binder_filter.io_utils import atomic_write_csv, atomic_write_text
from af3_binder_filter.jobs import JobSpec
from af3_binder_filter.residue_format import parse_residue_positions


class ClusteringError(RuntimeError):
    """Raised when a required clustering input or Foldseek output is invalid."""


@dataclass(frozen=True, slots=True)
class FoldseekRun:
    layer: str
    command: tuple[str, ...]
    cluster_tsv: Path
    status: str
    error: str | None = None


def build_foldseek_container_command(
    settings: ClusteringSettings,
    *,
    layer: str,
    docker_bin: str,
    image: str,
    gpu_index: int,
    input_dir: Path,
    execution_dir: Path,
    container_name: str,
) -> list[str]:
    """Build an offline GPU Foldseek command using only in-image binaries."""

    if settings.foldseek_binary not in {"foldseek", "/usr/local/bin/foldseek"}:
        raise ClusteringError(
            "clustering.foldseek_binary is an in-image command and must be "
            "'foldseek' or '/usr/local/bin/foldseek'; host binaries are disabled"
        )
    if layer == "binder":
        foldseek_args = [
            "easy-cluster",
            "/input",
            "/work/binder",
            "/work/binder_tmp",
            "--tmscore-threshold",
            str(settings.binder_tm_threshold),
            "-c",
            str(settings.binder_coverage),
            "--cov-mode",
            "0",
            "--gpu",
            "1",
        ]
    elif layer == "complex":
        foldseek_args = [
            "easy-multimercluster",
            "/input",
            "/work/complex",
            "/work/complex_tmp",
            "--multimer-tm-threshold",
            str(settings.multimer_tm_threshold),
            "--chain-tm-threshold",
            str(settings.chain_tm_threshold),
            "--interface-lddt-threshold",
            str(settings.interface_lddt_threshold),
            "--gpu",
            "1",
        ]
    else:
        raise ClusteringError(f"unsupported Foldseek layer: {layer}")
    return [
        docker_bin,
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--gpus",
        f"device={gpu_index}",
        "--volume",
        f"{input_dir.resolve()}:/input:ro",
        "--volume",
        f"{execution_dir.resolve()}:/work",
        image,
        settings.foldseek_binary,
        *foldseek_args,
    ]


def _atomic_save_structure(path: Path, array: Any) -> None:
    import biotite.structure.io as strucio

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        strucio.save_structure(str(temporary_path), array)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


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


def extract_binder_structure(
    complex_path: Path,
    output_path: Path,
    *,
    binder_chain: str,
) -> Path:
    import biotite.structure.io as strucio

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        array = strucio.load_structure(str(complex_path), model=1)
    binder = array[array.chain_id == binder_chain]
    if binder.array_length() == 0:
        raise ClusteringError(f"binder chain {binder_chain!r} not found in {complex_path}")
    _atomic_save_structure(output_path, binder)
    return output_path


def prepare_foldseek_inputs(
    jobs: Sequence[JobSpec],
    model_paths: Mapping[str, Path],
    *,
    work_dir: Path,
    rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[Path, Path]:
    binder_dir = work_dir / "binder_structures"
    complex_dir = work_dir / "complex_structures"
    binder_dir.mkdir(parents=True, exist_ok=True)
    complex_dir.mkdir(parents=True, exist_ok=True)
    row_by_job: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        job_id = row_job_identifier(row)
        if job_id is not None:
            row_by_job[job_id] = row
    for job in jobs:
        source_row = row_by_job.get(job.job_id)
        if source_row is not None and not row_structure_is_eligible(
            source_row,
            prefix="effective",
        ):
            continue
        destination = complex_dir / f"{job.job_id}.pdb"
        binder_destination = binder_dir / f"{job.job_id}.pdb"
        derived = validated_artifacts_from_row(
            source_row or {},
            prefix="effective",
        )
        if derived is not None:
            # Both files were written from the interface stage's single parse
            # and checksum-validated together.  Staging is now file I/O only.
            _atomic_copy(derived.complex_pdb, destination)
            _atomic_copy(derived.binder_pdb, binder_destination)
            continue
        model_path = model_paths.get(job.job_id)
        if model_path is None or not model_path.is_file():
            continue
        complex_array = load_protein_complex(
            model_path,
            target_chain=job.target_chain,
            binder_chain=job.binder_chain,
        )
        # PDB is intentionally used for Foldseek staging: a Biotite-written
        # minimal mmCIF lacks entity/polymer categories required by createdb.
        _atomic_save_structure(destination, complex_array)
        extract_binder_structure(
            destination,
            binder_destination,
            binder_chain=job.binder_chain,
        )
    return binder_dir, complex_dir


def run_foldseek_command(
    layer: str,
    command: Sequence[str],
    cluster_tsv: Path,
    *,
    dry_run: bool = False,
    log_dir: Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> FoldseekRun:
    if log_dir is not None:
        atomic_write_text(
            log_dir / f"foldseek_{layer}.command.txt",
            " ".join(command) + "\n",
        )
    if dry_run:
        return FoldseekRun(layer, tuple(command), cluster_tsv, "dry_run")
    try:
        completed = runner(
            list(command),
            capture_output=True,
            text=True,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return FoldseekRun(layer, tuple(command), cluster_tsv, "error", str(exc))
    if log_dir is not None:
        atomic_write_text(
            log_dir / f"foldseek_{layer}.stdout.log",
            completed.stdout or "",
        )
        atomic_write_text(
            log_dir / f"foldseek_{layer}.stderr.log",
            completed.stderr or "",
        )
    if completed.returncode != 0:
        return FoldseekRun(
            layer,
            tuple(command),
            cluster_tsv,
            "error",
            f"Foldseek returned {completed.returncode}: {completed.stderr.strip()}",
        )
    if not cluster_tsv.is_file():
        return FoldseekRun(
            layer,
            tuple(command),
            cluster_tsv,
            "error",
            f"Foldseek did not create {cluster_tsv}",
        )
    return FoldseekRun(layer, tuple(command), cluster_tsv, "success")


def _normalize_member(value: str) -> str:
    return Path(value.strip()).stem


def parse_foldseek_clusters(
    path: Path,
    *,
    all_job_ids: Iterable[str] = (),
    prefix: str,
) -> tuple[dict[str, str], dict[str, str]]:
    membership: dict[str, str] = {}
    raw_representatives: dict[str, str] = {}
    representative_to_cluster: dict[str, str] = {}
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 2:
                    continue
                representative = _normalize_member(fields[0])
                member = _normalize_member(fields[1])
                cluster_id = representative_to_cluster.setdefault(
                    representative,
                    f"{prefix}_{len(representative_to_cluster) + 1:04d}",
                )
                membership[member] = cluster_id
                raw_representatives[cluster_id] = representative
    # Do not manufacture singleton clusters for structures that never reached
    # Foldseek or disappeared from its output.  Callers compare ``all_job_ids``
    # with the returned membership and propagate the missing jobs as errors.
    _ = all_job_ids
    return membership, raw_representatives


def parse_residue_set(value: Any) -> frozenset[int]:
    """Compatibility wrapper accepting schema-v2 and legacy residue lists."""

    return parse_residue_positions(value)


def jaccard(left: frozenset[int], right: frozenset[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def greedy_epitope_clusters(
    contact_residues: Mapping[str, frozenset[int] | set[int] | Sequence[int] | str],
    *,
    threshold: float = 0.50,
) -> tuple[dict[str, str], dict[str, str]]:
    """Deterministic representative-based greedy clustering."""

    representatives: list[tuple[str, frozenset[int], str]] = []
    membership: dict[str, str] = {}
    raw_representatives: dict[str, str] = {}
    for job_id in sorted(contact_residues):
        contacts = parse_residue_set(contact_residues[job_id])
        assigned: str | None = None
        for representative_id, representative_contacts, cluster_id in representatives:
            if jaccard(contacts, representative_contacts) >= threshold:
                assigned = cluster_id
                break
        if assigned is None:
            assigned = f"epitope_{len(representatives) + 1:04d}"
            representatives.append((job_id, contacts, assigned))
            raw_representatives[assigned] = job_id
        membership[job_id] = assigned
    return membership, raw_representatives


def _float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _selection_pass(row: Mapping[str, Any]) -> bool:
    """Use the dual-backend gate when present, otherwise the backend-local gate."""

    candidate_pool = row.get("candidate_pool")
    if candidate_pool is not None and candidate_pool != "":
        return _truthy(candidate_pool)
    return _truthy(row.get("final_pass"))


def quality_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Quality ordering shared by every cluster layer."""

    def desc(name: str) -> float:
        value = _float(row.get(name))
        return -value if value is not None else math.inf

    def asc(name: str) -> float:
        value = _float(row.get(name))
        return value if value is not None else math.inf

    return (
        not _selection_pass(row),
        desc("effective_epitope_coverage"),
        asc("effective_interface_pae_mean"),
        asc("effective_rosetta_dG_separated_per_dSASA_x100"),
        desc("effective_rosetta_packstat"),
        desc("effective_iptm"),
        str(row.get("job_name", "")),
    )


def select_quality_representatives(
    rows_by_job: Mapping[str, Mapping[str, Any]],
    membership: Mapping[str, str],
) -> dict[str, str]:
    members: dict[str, list[str]] = {}
    for job_id, cluster_id in membership.items():
        members.setdefault(cluster_id, []).append(job_id)
    return {
        cluster_id: min(
            job_ids,
            key=lambda job_id: quality_key(
                rows_by_job.get(job_id, {"job_name": job_id})
            ),
        )
        for cluster_id, job_ids in members.items()
    }


def write_cluster_outputs(
    *,
    results_dir: Path,
    artifacts_dir: Path | None = None,
    jobs: Sequence[JobSpec],
    rows: Sequence[Mapping[str, Any]],
    binder_membership: Mapping[str, str],
    binder_raw_representatives: Mapping[str, str],
    complex_membership: Mapping[str, str],
    complex_raw_representatives: Mapping[str, str],
    epitope_membership: Mapping[str, str],
    epitope_raw_representatives: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    artifact_root = artifacts_dir or results_dir
    rows_by_job = {str(row.get("job_name")): row for row in rows}
    job_by_id = {job.job_id: job for job in jobs}
    binder_quality = select_quality_representatives(rows_by_job, binder_membership)
    complex_quality = select_quality_representatives(rows_by_job, complex_membership)
    epitope_quality = select_quality_representatives(rows_by_job, epitope_membership)

    member_rows: list[dict[str, Any]] = []
    for job_id in sorted(job_by_id):
        member_rows.append(
            {
                "job_name": job_id,
                "binder_cluster": binder_membership.get(job_id),
                "complex_cluster": complex_membership.get(job_id),
                "epitope_cluster": epitope_membership.get(job_id),
                **dict(rows_by_job.get(job_id, {})),
            }
        )
    atomic_write_csv(results_dir / "cluster_members.csv", member_rows)

    representative_rows: list[dict[str, Any]] = []
    layers = (
        ("binder", binder_membership, binder_raw_representatives, binder_quality),
        ("complex", complex_membership, complex_raw_representatives, complex_quality),
        ("epitope", epitope_membership, epitope_raw_representatives, epitope_quality),
    )
    for layer, membership, raw, quality in layers:
        sizes: dict[str, int] = {}
        for cluster_id in membership.values():
            sizes[cluster_id] = sizes.get(cluster_id, 0) + 1
        for cluster_id in sorted(sizes):
            representative_rows.append(
                {
                    "layer": layer,
                    "cluster_id": cluster_id,
                    "member_count": sizes[cluster_id],
                    "foldseek_representative": raw.get(cluster_id),
                    "quality_representative": quality.get(cluster_id),
                }
            )
    atomic_write_csv(results_dir / "cluster_representatives.csv", representative_rows)

    def write_tsv(name: str, membership: Mapping[str, str], raw: Mapping[str, str]) -> None:
        text = "cluster_id\trepresentative\tmember\n"
        text += "".join(
            f"{cluster_id}\t{raw.get(cluster_id, '')}\t{job_id}\n"
            for job_id, cluster_id in sorted(membership.items(), key=lambda item: (item[1], item[0]))
        )
        atomic_write_text(artifact_root / name, text)

    write_tsv("binder_clusters.tsv", binder_membership, binder_raw_representatives)
    write_tsv("complex_clusters.tsv", complex_membership, complex_raw_representatives)
    write_tsv("epitope_clusters.tsv", epitope_membership, epitope_raw_representatives)
    complex_report = [
        row for row in representative_rows if row["layer"] == "complex"
    ]
    text = "cluster_id\tmember_count\tfoldseek_representative\tquality_representative\n"
    text += "".join(
        f"{row['cluster_id']}\t{row['member_count']}\t"
        f"{row['foldseek_representative']}\t{row['quality_representative']}\n"
        for row in complex_report
    )
    atomic_write_text(artifact_root / "complex_cluster_report.tsv", text)

    fasta = "".join(
        f">{job_id}\n{job_by_id[job_id].binder_sequence}\n"
        for job_id in sorted(set(binder_quality.values()))
        if job_id in job_by_id
    )
    atomic_write_text(artifact_root / "binder_representatives.fasta", fasta)

    # One quality representative per unique three-layer diversity cell.
    cells: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = {}
    for row in member_rows:
        cell = (
            row.get("binder_cluster"),
            row.get("complex_cluster"),
            row.get("epitope_cluster"),
        )
        if any(value in (None, "") for value in cell):
            continue
        cells.setdefault(cell, []).append(row)
    final_shortlist = [
        min(cell_rows, key=quality_key)
        for _cell, cell_rows in sorted(cells.items(), key=lambda item: str(item[0]))
        if any(_selection_pass(row) for row in cell_rows)
    ]
    final_shortlist.sort(key=quality_key)
    atomic_write_csv(results_dir / "final_shortlist.csv", final_shortlist)
    return member_rows, representative_rows, final_shortlist
