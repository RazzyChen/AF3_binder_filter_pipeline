"""GPU discovery and sharding policy."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=index,name,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    memory_used_mib: int
    memory_total_mib: int

    def is_free(self, threshold_mib: int) -> bool:
        return self.memory_used_mib <= threshold_mib


@dataclass(frozen=True)
class Shard:
    gpu: GPUInfo
    jobs: tuple[Path, ...]


class GPUError(RuntimeError):
    """Raised when GPUs cannot be queried or assigned."""


def parse_nvidia_smi_csv(output: str) -> list[GPUInfo]:
    """Parse nvidia-smi CSV output."""

    gpus: list[GPUInfo] = []
    for line_number, line in enumerate(output.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise GPUError(f"invalid nvidia-smi line {line_number}: {line!r}")
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    memory_used_mib=int(parts[2]),
                    memory_total_mib=int(parts[3]),
                )
            )
        except ValueError as exc:
            raise GPUError(
                f"invalid nvidia-smi numeric field on line {line_number}: {line!r}"
            ) from exc
    return gpus


def query_gpus() -> list[GPUInfo]:
    """Query GPUs with nvidia-smi."""

    try:
        result = subprocess.run(
            NVIDIA_SMI_QUERY, shell=False, check=False, text=True, capture_output=True
        )
    except OSError as exc:
        raise GPUError(f"nvidia-smi is unavailable: {exc}") from exc
    if result.returncode != 0:
        raise GPUError(f"nvidia-smi failed with code {result.returncode}: {result.stderr.strip()}")
    return parse_nvidia_smi_csv(result.stdout)


def select_free_gpus(
    gpus: Sequence[GPUInfo],
    *,
    threshold_mib: int = 100,
    allowed_gpu_ids: Sequence[int] | None = None,
) -> list[GPUInfo]:
    """Select free GPUs by physical index ascending."""

    allowed = set(allowed_gpu_ids) if allowed_gpu_ids is not None else None
    candidates = [gpu for gpu in gpus if allowed is None or gpu.index in allowed]
    return sorted(
        (gpu for gpu in candidates if gpu.is_free(threshold_mib)),
        key=lambda gpu: gpu.index,
    )


def shard_jobs(jobs: Sequence[Path], free_gpus: Sequence[GPUInfo]) -> list[Shard]:
    """Shard pending jobs over free GPUs using at most one shard per selected GPU."""

    pending = list(jobs)
    if not pending:
        return []
    if not free_gpus:
        raise GPUError("jobs are pending but no free GPU is available")

    selected_gpus = list(free_gpus)[: min(len(pending), len(free_gpus))]
    shards = [list[Path]() for _ in selected_gpus]
    for index, job in enumerate(pending):
        shards[index % len(selected_gpus)].append(job)
    return [
        Shard(gpu=gpu, jobs=tuple(shard)) for gpu, shard in zip(selected_gpus, shards, strict=True)
    ]
