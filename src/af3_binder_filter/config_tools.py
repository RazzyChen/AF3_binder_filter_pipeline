"""Environment discovery, configuration generation, and doctor checks."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from af3_binder_filter.config import AerithConfig, ConfigValidationReport
from af3_binder_filter.io_utils import atomic_write_yaml


@dataclass(frozen=True)
class EnvironmentDetection:
    gpu_indexes: tuple[int, ...] = ()
    docker: str | None = None
    database_dir: str | None = None
    foldseek: str | None = None
    mmseqs: str | None = None
    rosetta_binary: str | None = None
    rosetta_database: str | None = None
    protenix_source: str | None = None
    opendde_source: str | None = None
    opendde_commit: str | None = None
    notes: tuple[str, ...] = ()


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def detect_environment() -> EnvironmentDetection:
    notes: list[str] = []
    gpu_indexes: list[int] = []
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=index", "--format=csv,noheader,nounits"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            gpu_indexes = [int(line.strip()) for line in result.stdout.splitlines() if line.strip()]
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            notes.append(f"GPU discovery failed: {exc}")

    database_dir = Path("/data/AF3_database")
    rosetta_root = Path("/home/structure/Software/rosetta.source.release-408/main")
    rosetta_binary = rosetta_root / "source/bin/InterfaceAnalyzer.mpiserialization.linuxgccrelease"
    protenix = Path("/home/structure/Software/Protenix-2.0.0")
    opendde = Path("/home/structure/Software/OpenDDE")
    return EnvironmentDetection(
        gpu_indexes=tuple(gpu_indexes),
        docker=shutil.which("docker"),
        database_dir=str(database_dir) if database_dir.is_dir() else None,
        foldseek=shutil.which("foldseek"),
        mmseqs=shutil.which("mmseqs"),
        rosetta_binary=str(rosetta_binary) if rosetta_binary.is_file() else None,
        rosetta_database=str(rosetta_root / "database") if (rosetta_root / "database").is_dir() else None,
        protenix_source=str(protenix) if protenix.is_dir() else None,
        opendde_source=str(opendde) if opendde.is_dir() else None,
        opendde_commit=_git_head(opendde) if opendde.is_dir() else None,
        notes=tuple(notes),
    )


def resolve_docker_image_id(docker_bin: str, image: str) -> str | None:
    """Resolve a mutable image tag to its immutable local Docker ID."""

    try:
        completed = subprocess.run(
            [docker_bin, "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def resolve_executable_id(executable: str | None) -> str | None:
    """Resolve an executable path and version into a stable cache identity."""

    if not executable:
        return None
    resolved = shutil.which(executable) if "/" not in executable else executable
    if not resolved:
        return None
    path = Path(resolved).expanduser().resolve()
    if not path.is_file():
        return None
    version = ""
    try:
        completed = subprocess.run(
            [str(path), "version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode == 0:
            version = (completed.stdout or completed.stderr).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    stat = path.stat()
    return f"{path}:{version}:{stat.st_size}:{stat.st_mtime_ns}"


def initial_config_payload(
    detection: EnvironmentDetection,
    *,
    csv_path: str = "all_seq_PD1_May12.csv",
    backend: str = "alphafold3",
    secondary_backend: str = "none",
) -> dict:
    database = detection.database_dir or "/data/AF3_database"
    rosetta_binary = detection.rosetta_binary or (
        "/home/structure/Software/rosetta.source.release-408/main/source/bin/"
        "InterfaceAnalyzer.mpiserialization.linuxgccrelease"
    )
    rosetta_database = detection.rosetta_database or (
        "/home/structure/Software/rosetta.source.release-408/main/database"
    )
    return {
        "defaults": [
            {"backend": backend},
            {"secondary_backend": secondary_backend},
            {"features": "local_af3_db"},
            {"interface": "biotite_rosetta"},
            {"clustering": "balanced"},
            "_self_",
        ],
        "project": {
            "csv_path": csv_path,
            "work_dir": "work",
            "output_dir": "af_output",
            "results_dir": "results",
            "target_chain": "A",
            "binder_chain": "B",
            "job_name_template": "sample_{sample_no}_binder_candiate_complex_pred",
            "seed": 42,
            "run_id": None,
            "limit": None,
            "prune": False,
            "adopt_legacy": False,
            "allow_partial": False,
        },
        "features": {
            "docker_bin": detection.docker or "docker",
            "database_dir": database,
            "mmseqs_binary": None,
            "use_gpu": True,
        },
        "interface": {
            "epitope_residues": None,
            "rosetta": {"binary": rosetta_binary, "database": rosetta_database},
        },
        "clustering": {
            "foldseek_binary": "foldseek",
        },
        "runtime": {
            "dry_run": False,
            "force": False,
            "gpu_ids": list(detection.gpu_indexes),
            "gpu_busy_threshold_mib": 100,
            "dockerfile": "docker/runtime/Dockerfile",
            "af3_source_dir": "/home/structure/Software/alphafold3-3.0.3",
            "protenix_source_dir": "/home/structure/Software/Protenix-2.0.0",
            "opendde_source_dir": "/home/structure/Software/OpenDDE",
            "opendde_source_commit": "266ce4c49d492ad1077866000d83704999985f46",
            "esm_source_dir": "/home/structure/Software/esm",
            "esm_source_commit": "2b369911bb5b4b0dda914521b9475cad1656b2ac",
            "mmseqs_release": "18-8cc5c",
            "mmseqs_version": "8cc5ce367b5638c4306c2d7cfc652dd099a4643f",
            "mmseqs_archive_sha256": (
                "83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7"
            ),
            "foldseek_release": "10-941cd33",
            "foldseek_version": "941cd33ff0771cd2e3f144e3293e22a2b87e9fda",
            "foldseek_archive_sha256": (
                "af7a688ffd8625b356c380380fb5650ec811262a2d18bdb0faeda95cc4894a55"
            ),
            "minimum_build_free_gib": 45,
            "build_proxy": None,
            "build_add_host": None,
        },
        "hydra": {
            "searchpath": ["pkg://af3_binder_filter.conf"],
            "job": {"chdir": False},
        },
    }


def write_initial_config(
    output: Path,
    detection: EnvironmentDetection,
    *,
    csv_path: str = "all_seq_PD1_May12.csv",
    backend: str = "alphafold3",
    secondary_backend: str = "none",
) -> Path:
    if output.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("configuration output must use .yaml or .yml")
    atomic_write_yaml(
        output,
        initial_config_payload(
            detection,
            csv_path=csv_path,
            backend=backend,
            secondary_backend=secondary_backend,
        ),
    )
    return output


def minimal_production_config_payload(
    *,
    project_root: Path,
    csv_path: Path | None = None,
    secondary_backend: str = "opendde",
    gpu_ids: Sequence[int] = (),
    epitope_residues: str | None = None,
) -> dict:
    """Build the smallest explicit configuration for a production screen.

    Values that are stable pipeline defaults remain in the Structured Config
    schema. The generated YAML records only choices that define a screen plus
    Hydra's package search path, which is required when the file lives outside
    the source tree.
    """

    if secondary_backend not in {"none", "protenix", "opendde"}:
        raise ValueError("secondary_backend must be none, protenix, or opendde")

    normalized_gpu_ids = list(gpu_ids)
    if any(index < 0 for index in normalized_gpu_ids):
        raise ValueError("GPU indexes must be non-negative")
    if len(normalized_gpu_ids) != len(set(normalized_gpu_ids)):
        raise ValueError("GPU indexes must be unique")

    root = project_root.expanduser().resolve()
    input_csv = (
        csv_path.expanduser().resolve()
        if csv_path is not None
        else root / "input" / "screen.csv"
    )
    epitope = epitope_residues.strip() if epitope_residues else None

    return {
        "defaults": [
            {"backend": "alphafold3"},
            {"secondary_backend": secondary_backend},
            {"features": "local_af3_db"},
            {"interface": "biotite_rosetta"},
            {"clustering": "balanced"},
            "_self_",
        ],
        "project": {
            "csv_path": str(input_csv),
            "work_dir": str(root / "work"),
            "output_dir": str(root / "outputs"),
            "results_dir": str(root / "results"),
        },
        "runtime": {"gpu_ids": normalized_gpu_ids},
        "interface": {"epitope_residues": epitope},
        "hydra": {
            "searchpath": ["pkg://af3_binder_filter.conf"],
            "job": {"chdir": False},
        },
    }


def write_minimal_production_config(
    output: Path,
    *,
    project_root: Path,
    csv_path: Path | None = None,
    secondary_backend: str = "opendde",
    gpu_ids: Sequence[int] = (),
    epitope_residues: str | None = None,
) -> Path:
    """Write a non-interactive, production-oriented minimal Hydra YAML."""

    if output.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("configuration output must use .yaml or .yml")
    atomic_write_yaml(
        output,
        minimal_production_config_payload(
            project_root=project_root,
            csv_path=csv_path,
            secondary_backend=secondary_backend,
            gpu_ids=gpu_ids,
            epitope_residues=epitope_residues,
        ),
    )
    return output


def doctor_config(
    config: AerithConfig,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ConfigValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    checks: list[tuple[str, Sequence[str], bool]] = [
        ("Docker", [config.backend.docker_bin, "info", "--format", "{{.ServerVersion}}"], True),
        ("GPU", ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], True),
        (
            "backend image",
            [
                config.backend.docker_bin,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                config.backend.image,
            ],
            True,
        ),
        (
            "runtime environments",
            [
                config.backend.docker_bin,
                "run",
                "--rm",
                "--network",
                "none",
                "--gpus",
                "all",
                config.backend.image,
                "doctor",
            ],
            True,
        ),
    ]
    for label, command, required in checks:
        try:
            timeout = 120 if label == "runtime environments" else 20
            result = runner(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            if result.returncode != 0:
                message = f"{label} check failed ({result.returncode}): {result.stderr.strip()}"
                (errors if required else warnings).append(message)
            else:
                first_line = (result.stdout.strip().splitlines() or ["ok"])[0]
                info.append(f"{label}: {first_line}")
        except (OSError, subprocess.SubprocessError) as exc:
            (errors if required else warnings).append(f"{label} check failed: {exc}")

    for label, executable in (
        ("Rosetta", config.interface.rosetta.binary),
    ):
        resolved = shutil.which(executable) if "/" not in executable else executable
        if not resolved or not Path(resolved).is_file():
            warnings.append(f"{label} executable not found: {executable}")
        else:
            info.append(f"{label}: {resolved}")

    database = Path(config.features.database_dir)
    try:
        next(database.iterdir())
        info.append(f"AF3 database readable: {database}")
    except (OSError, StopIteration) as exc:
        errors.append(f"AF3 database read-only smoke check failed: {database}: {exc}")

    selected_sources = [config.backend]
    if config.secondary_backend.enabled:
        selected_sources.append(config.secondary_backend)
    for backend in selected_sources:
        if not backend.source_commit or not backend.source_dir:
            continue
        actual_commit = _git_head(Path(backend.source_dir))
        if actual_commit is None or not actual_commit.startswith(backend.source_commit):
            errors.append(
                f"{backend.name} source commit mismatch: expected {backend.source_commit}, "
                f"found {actual_commit or 'unavailable'}"
            )
        else:
            info.append(f"{backend.name} source commit: {actual_commit[:12]}")

    return ConfigValidationReport(tuple(errors), tuple(warnings), tuple(info))
