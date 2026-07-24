"""ESMFold single-chain structure and pLDDT scoring helpers."""

from __future__ import annotations

import csv
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from af3_binder_filter.af3_json import get_chain_sequence, load_af3_input
from af3_binder_filter.config import ESMFoldConfig
from af3_binder_filter.gpu import GPUInfo, query_gpus, select_free_gpus


class ESMFoldError(RuntimeError):
    """Raised when ESMFold scoring setup or parsing fails."""


@dataclass(frozen=True)
class ESMFoldJob:
    job_name: str
    input_json: Path
    header: str
    sequence: str


@dataclass(frozen=True)
class ESMFoldShard:
    gpu: GPUInfo
    jobs: tuple[ESMFoldJob, ...]
    fasta_path: Path
    pdb_dir: Path
    stdout_path: Path
    stderr_path: Path
    command: tuple[str, ...]


def write_esmfold_fasta(input_json: Path, fasta_path: Path, *, chain_id: str = "B") -> Path:
    """Write a single-chain FASTA for ESMFold from an AF3 complex input JSON."""

    payload = load_af3_input(input_json)
    job_name = str(payload.get("name") or input_json.stem)
    sequence = get_chain_sequence(payload, chain_id)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(f">{job_name}_chain_{chain_id}\n{sequence}\n")
    return fasta_path


def build_esmfold_command(*, fasta_path: Path, pdb_dir: Path, config: ESMFoldConfig) -> list[str]:
    """Build an ESMFold CLI command."""

    command = [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        config.binary,
        "-i",
        str(fasta_path),
        "-o",
        str(pdb_dir),
    ]
    if config.model_dir is not None:
        command.extend(["--model-dir", str(config.model_dir)])
    if config.num_recycles is not None:
        command.extend(["--num-recycles", str(config.num_recycles)])
    if config.max_tokens_per_batch is not None:
        command.extend(["--max-tokens-per-batch", str(config.max_tokens_per_batch)])
    if config.chunk_size is not None:
        command.extend(["--chunk-size", str(config.chunk_size)])
    if config.cpu_only:
        command.append("--cpu-only")
    if config.cpu_offload:
        command.append("--cpu-offload")
    return command


def parse_esmfold_plddt(pdb_path: Path) -> float:
    """Parse mean residue pLDDT from ESMFold PDB B-factor values."""

    if not pdb_path.exists():
        raise ESMFoldError(f"ESMFold PDB does not exist: {pdb_path}")
    atom_values: list[float] = []
    ca_values: list[float] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            raw_value = line[60:66].strip()
            if not raw_value:
                continue
            value = float(raw_value)
            atom_values.append(value)
            if line[12:16].strip() == "CA":
                ca_values.append(value)
    values = ca_values or atom_values
    if not values:
        raise ESMFoldError(f"ESMFold PDB has no ATOM B-factor values: {pdb_path}")
    return float(sum(values) / len(values))


def _expected_pdb_path(pdb_dir: Path, *, job_name: str, chain_id: str) -> Path:
    return pdb_dir / f"{job_name}_chain_{chain_id}.pdb"


def _load_jobs(input_jsons: Sequence[Path], *, chain_id: str) -> list[ESMFoldJob]:
    jobs: list[ESMFoldJob] = []
    for input_json in input_jsons:
        input_json = input_json.resolve()
        payload = load_af3_input(input_json)
        job_name = str(payload.get("name") or input_json.stem)
        jobs.append(
            ESMFoldJob(
                job_name=job_name,
                input_json=input_json,
                header=f"{job_name}_chain_{chain_id}",
                sequence=get_chain_sequence(payload, chain_id),
            )
        )
    return jobs


def _find_existing_pdb(output_root: Path, job: ESMFoldJob, *, chain_id: str) -> Path | None:
    filename = f"{job.job_name}_chain_{chain_id}.pdb"
    candidates = [output_root / job.job_name / "pdb" / filename]
    candidates.extend(sorted((output_root / "shards").glob(f"gpu_*/pdb/{filename}")))
    for candidate in candidates:
        if candidate.exists():
            try:
                parse_esmfold_plddt(candidate)
            except Exception:
                continue
            return candidate
    return None


def _write_shard_fasta(jobs: Sequence[ESMFoldJob], fasta_path: Path) -> None:
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    with fasta_path.open("w") as handle:
        for job in jobs:
            handle.write(f">{job.header}\n{job.sequence}\n")


def _build_shards(
    *,
    jobs: Sequence[ESMFoldJob],
    free_gpus: Sequence[GPUInfo],
    output_root: Path,
    config: ESMFoldConfig,
) -> list[ESMFoldShard]:
    if not jobs:
        return []
    if not free_gpus:
        raise ESMFoldError("ESMFold jobs are pending but no free GPU is available")

    selected_gpus = list(free_gpus)[: min(len(jobs), len(free_gpus))]
    grouped_jobs: list[list[ESMFoldJob]] = [[] for _ in selected_gpus]
    for index, job in enumerate(jobs):
        grouped_jobs[index % len(selected_gpus)].append(job)

    shards: list[ESMFoldShard] = []
    for gpu, shard_jobs in zip(selected_gpus, grouped_jobs, strict=True):
        shard_dir = output_root / "shards" / f"gpu_{gpu.index}"
        fasta_path = shard_dir / "input.fasta"
        pdb_dir = shard_dir / "pdb"
        command = tuple(
            build_esmfold_command(fasta_path=fasta_path, pdb_dir=pdb_dir, config=config)
        )
        _write_shard_fasta(shard_jobs, fasta_path)
        pdb_dir.mkdir(parents=True, exist_ok=True)
        shards.append(
            ESMFoldShard(
                gpu=gpu,
                jobs=tuple(shard_jobs),
                fasta_path=fasta_path,
                pdb_dir=pdb_dir,
                stdout_path=shard_dir / "esmfold.stdout.log",
                stderr_path=shard_dir / "esmfold.stderr.log",
                command=command,
            )
        )
    return shards


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )


def _pdb_count(shards: Sequence[ESMFoldShard]) -> int:
    return sum(1 for shard in shards for _ in shard.pdb_dir.glob("*.pdb"))


def _run_shards(shards: Sequence[ESMFoldShard], *, dry_run: bool) -> dict[int, int | None]:
    return_codes: dict[int, int | None] = {}
    total_jobs = sum(len(shard.jobs) for shard in shards)
    if total_jobs == 0:
        return return_codes
    if dry_run:
        with _progress() as progress:
            task = progress.add_task("Preparing ESMFold shards", total=total_jobs)
            progress.update(task, completed=total_jobs)
        for shard in shards:
            return_codes[shard.gpu.index] = None
        return return_codes

    running = []
    for shard in shards:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(shard.gpu.index)}
        stdout_handle = shard.stdout_path.open("w")
        stderr_handle = shard.stderr_path.open("w")
        process = subprocess.Popen(  # noqa: S603 - command is configured, not shell-expanded
            list(shard.command),
            shell=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=env,
        )
        running.append((shard, process, stdout_handle, stderr_handle))

    with _progress() as progress:
        task = progress.add_task("Running ESMFold shards", total=total_jobs)
        while running:
            completed = min(_pdb_count(shards), total_jobs)
            progress.update(task, completed=completed)
            still_running = []
            for shard, process, stdout_handle, stderr_handle in running:
                return_code = process.poll()
                if return_code is None:
                    still_running.append((shard, process, stdout_handle, stderr_handle))
                    continue
                return_codes[shard.gpu.index] = return_code
                stdout_handle.close()
                stderr_handle.close()
            running = still_running
            if running:
                time.sleep(5)
        progress.update(task, completed=min(_pdb_count(shards), total_jobs))
    return return_codes


def write_esmfold_summary(summary_csv: Path, rows: list[dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _row_for_success(
    job: ESMFoldJob, *, fasta_path: Path, pdb_path: Path, command: str = ""
) -> dict[str, Any]:
    return {
        "job_name": job.job_name,
        "esmfold_status": "success",
        "esmfold_plddt_mean": parse_esmfold_plddt(pdb_path),
        "esmfold_fasta_path": str(fasta_path),
        "esmfold_pdb_path": str(pdb_path),
        "esmfold_error": "",
        "esmfold_command": command,
    }


def score_esmfold_inputs(
    *,
    input_dir: Path,
    input_jsons: list[Path] | None = None,
    score_dir: Path,
    chain_id: str,
    config: ESMFoldConfig,
    dry_run: bool = False,
    force: bool = False,
    gpu_busy_threshold_mib: int = 100,
    gpu_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Run ESMFold on design-chain sequences, sharded over currently free GPUs."""

    score_dir = score_dir.resolve()
    output_root = score_dir / "esmfold"
    input_paths = list(input_jsons) if input_jsons is not None else sorted(input_dir.glob("*.json"))
    jobs = _load_jobs(input_paths, chain_id=chain_id)

    existing: dict[str, Path] = {}
    pending: list[ESMFoldJob] = []
    with _progress() as progress:
        existing_task = progress.add_task("Checking existing ESMFold outputs", total=len(jobs))
        for job in jobs:
            existing_pdb = (
                None if force else _find_existing_pdb(output_root, job, chain_id=chain_id)
            )
            if existing_pdb is None:
                pending.append(job)
            else:
                existing[job.job_name] = existing_pdb
            progress.advance(existing_task)

    free_gpus = []
    if pending:
        free_gpus = select_free_gpus(
            query_gpus(),
            threshold_mib=gpu_busy_threshold_mib,
            allowed_gpu_ids=gpu_ids,
        )
    shards = _build_shards(
        jobs=pending, free_gpus=free_gpus, output_root=output_root, config=config
    )
    shard_by_job = {job.job_name: shard for shard in shards for job in shard.jobs}
    return_codes = _run_shards(shards, dry_run=dry_run)

    rows: list[dict[str, Any]] = []
    with _progress() as progress:
        parse_task = progress.add_task("Parsing ESMFold pLDDT", total=len(jobs))
        for job in jobs:
            if job.job_name in existing:
                rows.append(
                    _row_for_success(
                        job,
                        fasta_path=output_root
                        / job.job_name
                        / f"{job.job_name}_chain_{chain_id}.fasta",
                        pdb_path=existing[job.job_name],
                    )
                )
                progress.advance(parse_task)
                continue

            shard = shard_by_job[job.job_name]
            command_text = f"CUDA_VISIBLE_DEVICES={shard.gpu.index} " + " ".join(shard.command)
            pdb_path = _expected_pdb_path(shard.pdb_dir, job_name=job.job_name, chain_id=chain_id)
            base_row: dict[str, Any] = {
                "job_name": job.job_name,
                "esmfold_fasta_path": str(shard.fasta_path),
                "esmfold_pdb_path": str(pdb_path),
                "esmfold_command": command_text,
            }
            if dry_run:
                rows.append(
                    {
                        **base_row,
                        "esmfold_status": "skipped",
                        "esmfold_error": "dry-run",
                    }
                )
                progress.advance(parse_task)
                continue
            try:
                rows.append(
                    _row_for_success(
                        job,
                        fasta_path=shard.fasta_path,
                        pdb_path=pdb_path,
                        command=command_text,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - per-job failures should stay resumable
                return_code = return_codes.get(shard.gpu.index)
                error = str(exc)
                if return_code not in (None, 0):
                    error = f"ESMFold shard on GPU {shard.gpu.index} failed with code {return_code}: {error}"
                rows.append({**base_row, "esmfold_status": "error", "esmfold_error": error})
            progress.advance(parse_task)

    write_esmfold_summary(score_dir / "esmfold_scores_summary.csv", rows)
    return rows
