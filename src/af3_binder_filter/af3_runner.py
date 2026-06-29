"""AlphaFold 3 Docker command construction and execution."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from af3_binder_filter.config import AF3DockerConfig
from af3_binder_filter.gpu import Shard
from af3_binder_filter.models import ExternalCommand


@dataclass(frozen=True)
class PreparedShard:
    gpu_index: int
    input_dir: Path
    jobs: tuple[Path, ...]
    command: list[str]
    stdout_path: Path
    stderr_path: Path



def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

def af3_job_name(input_json: Path) -> str:
    data = json.loads(input_json.read_text())
    return str(data.get("name") or input_json.stem)


def job_successful(job_name: str, output_dir: Path) -> bool:
    job_dir = output_dir / job_name
    expected = [
        job_dir / f"{job_name}_summary_confidences.json",
        job_dir / f"{job_name}_confidences.json",
        job_dir / f"{job_name}_model.cif",
    ]
    return all(path.exists() for path in expected)


def pending_input_jsons(
    input_dir: Path,
    output_dir: Path,
    *,
    force: bool = False,
    input_jsons: Sequence[Path] | None = None,
) -> list[Path]:
    jobs = list(input_jsons) if input_jsons is not None else sorted(input_dir.glob("*.json"))
    if force:
        return jobs
    return [path for path in jobs if not job_successful(af3_job_name(path), output_dir)]


def build_af3_docker_command(
    *,
    input_dir: Path,
    output_dir: Path,
    gpu_index: int,
    config: AF3DockerConfig,
) -> list[str]:
    """Build the verified cluster AF3 Docker command."""

    tokamax_target = (
        "/alphafold3_venv/lib/python3.12/site-packages/"
        "tokamax/data/autotuning/nvidia_geforce_rtx_3090"
    )
    return [
        config.docker_bin,
        "run",
        "--rm",
        "--volume",
        f"{input_dir.resolve()}:/root/af_input",
        "--volume",
        f"{output_dir.resolve()}:/root/af_output",
        "--volume",
        f"{config.model_dir}:/root/models",
        "--volume",
        f"{config.database_dir}:/root/public_databases",
        "--gpus",
        f"device={gpu_index}",
        "-e",
        f"XLA_CLIENT_MEM_FRACTION={config.xla_client_mem_fraction}",
        "-v",
        f"{config.jax_cache_dir}:/tmp/jax_cache",
        "-e",
        "JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache",
        "-v",
        f"{config.jax_cache_dir / 'triton'}:/tmp/triton_cache",
        "-e",
        "TRITON_CACHE_DIR=/tmp/triton_cache",
        "-v",
        f"{config.jax_cache_dir / 'tokamax'}:{tokamax_target}",
        config.image,
        "python",
        "run_alphafold.py",
        "--input_dir=/root/af_input",
        "--output_dir=/root/af_output",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--gpu_device=0",
    ]


def _copy_input_assets(source_root: Path, destination_root: Path) -> None:
    """Copy non-JSON relative input assets used by AF3 JSONs into a shard directory."""

    for child in source_root.iterdir():
        if child.suffix.lower() == ".json":
            continue
        destination = destination_root / child.name
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if child.is_dir():
            shutil.copytree(child, destination)
        else:
            shutil.copy2(child, destination)


def prepare_shard_dirs(
    shards: Sequence[Shard],
    *,
    shard_root: Path,
    output_dir: Path,
    config: AF3DockerConfig,
) -> list[PreparedShard]:
    """Create shard input dirs and Docker command records."""

    prepared: list[PreparedShard] = []
    log_dir = shard_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    for shard in shards:
        input_dir = shard_root / f"gpu_{shard.gpu.index}"
        input_dir.mkdir(parents=True, exist_ok=True)
        for old_json in input_dir.glob("*.json"):
            old_json.unlink()
        for source_root in sorted({job.parent for job in shard.jobs}):
            _copy_input_assets(source_root, input_dir)
        for job in shard.jobs:
            shutil.copy2(job, input_dir / job.name)
        command = build_af3_docker_command(
            input_dir=input_dir,
            output_dir=output_dir,
            gpu_index=shard.gpu.index,
            config=config,
        )
        prepared.append(
            PreparedShard(
                gpu_index=shard.gpu.index,
                input_dir=input_dir,
                jobs=shard.jobs,
                command=command,
                stdout_path=log_dir / f"gpu_{shard.gpu.index}.stdout.log",
                stderr_path=log_dir / f"gpu_{shard.gpu.index}.stderr.log",
            )
        )
    return prepared


def command_record(prepared: PreparedShard) -> ExternalCommand:
    return ExternalCommand(
        argv=prepared.command,
        cwd=Path.cwd(),
        stdout_path=prepared.stdout_path,
        stderr_path=prepared.stderr_path,
    )


def run_prepared_shards(prepared_shards: Sequence[PreparedShard], *, dry_run: bool = False) -> int:
    """Run one Docker process per prepared shard and return the worst return code."""

    if dry_run:
        return 0

    processes: list[tuple[PreparedShard, subprocess.Popen[bytes], object, object]] = []
    for prepared in prepared_shards:
        prepared.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_handle = prepared.stdout_path.open("wb")
        stderr_handle = prepared.stderr_path.open("wb")
        stdout_handle.write(f"# started {datetime.now().isoformat()}\n".encode())
        stderr_handle.write(f"# started {datetime.now().isoformat()}\n".encode())
        process = subprocess.Popen(
            prepared.command,
            shell=False,
            stdout=stdout_handle,
            stderr=stderr_handle,
        )
        processes.append((prepared, process, stdout_handle, stderr_handle))

    return_codes: list[int] = []
    with _progress() as progress:
        task = progress.add_task("Running AF3 Docker shards", total=len(processes))
        while processes:
            still_running = []
            for prepared, process, stdout_handle, stderr_handle in processes:
                return_code = process.poll()
                if return_code is None:
                    still_running.append((prepared, process, stdout_handle, stderr_handle))
                    continue
                return_codes.append(return_code)
                stdout_handle.close()
                stderr_handle.close()
                progress.advance(task)
            processes = still_running
            if processes:
                time.sleep(5)
    return max(return_codes, default=0)
