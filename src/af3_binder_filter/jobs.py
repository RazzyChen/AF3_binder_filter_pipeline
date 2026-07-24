"""Immutable job planning and run fingerprinting."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

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
        raise CsvInputError(
            f"all jobs in one run must share one target sequence; mismatch at {preview}"
        )
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


def file_sha256(path: Path) -> str | None:
    """Hash a regular file without making its absolute path part of identity."""

    try:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


_FILE_DIGEST_CACHE: dict[tuple[Any, ...], str] = {}


def file_asset_identity(value: str | Path | None) -> dict[str, Any] | None:
    """Return a path-free, content-complete identity for one file asset."""

    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False}
    if not path.is_file():
        return {"exists": True, "kind": "not_file"}
    cache_key = (
        str(path.resolve()),
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
    )
    digest = _FILE_DIGEST_CACHE.get(cache_key)
    if digest is None:
        digest = file_sha256(path)
        if digest is None:
            return {"exists": False}
        _FILE_DIGEST_CACHE[cache_key] = digest
    return {"exists": True, "size": stat.st_size, "sha256": digest}


def _bounded_sample_sha256(
    path: Path,
    *,
    size: int,
    sample_bytes: int = 1024 * 1024,
) -> str | None:
    """Hash deterministic first/middle/last chunks of a large immutable asset."""

    offsets = sorted(
        {
            0,
            max(0, (size - sample_bytes) // 2),
            max(0, size - sample_bytes),
        }
    )
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for offset in offsets:
                handle.seek(offset)
                chunk = handle.read(sample_bytes)
                digest.update(str(offset).encode("ascii"))
                digest.update(b"\0")
                digest.update(str(len(chunk)).encode("ascii"))
                digest.update(b"\0")
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _bounded_file_identity(
    path: Path,
    *,
    hash_limit: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Return an explicitly labelled bounded identity for a database member.

    Small members receive a complete SHA256. Large members deliberately use
    a fixed first/middle/last content sample so multi-hundred-GB databases do
    not need to be read in full merely to compose a run configuration.
    """

    try:
        stat = path.stat()
    except OSError:
        return {"exists": False}
    result: dict[str, Any] = {"exists": True, "size": stat.st_size}
    if stat.st_size <= hash_limit:
        result["identity_mode"] = "full-sha256-v1"
        result["sha256"] = file_sha256(path)
    else:
        result.update(
            {
                "identity_mode": "bounded-content-sample-v1",
                "sample_bytes": 1024 * 1024,
                "sample_sha256": _bounded_sample_sha256(
                    path,
                    size=stat.st_size,
                ),
            }
        )
    return result


def _directory_listing_identity(
    value: str | Path | None,
    *,
    prefixes: Sequence[str] = (),
    max_entries: int = 256,
) -> dict[str, Any] | None:
    """Return deterministic, bounded metadata for a release/database directory."""

    if value in (None, ""):
        return None
    root = Path(value).expanduser()
    if not root.is_dir():
        return {"exists": False}
    if max_entries < 1:
        raise ValueError("max_entries must be positive")
    try:
        entries = sorted(
            (
                entry
                for entry in root.iterdir()
                if not prefixes or any(entry.name.startswith(prefix) for prefix in prefixes)
            ),
            key=lambda entry: entry.name,
        )
    except OSError:
        return {"exists": False}
    if len(entries) <= max_entries:
        selected = entries
    elif max_entries == 1:
        selected = entries[:1]
    else:
        # Evenly sample the complete sorted name space instead of only its
        # first members. This remains bounded while covering both ends and
        # the interior of large template releases.
        indices = {
            round(index * (len(entries) - 1) / (max_entries - 1)) for index in range(max_entries)
        }
        selected = [entries[index] for index in sorted(indices)]
    metadata: list[dict[str, Any]] = []
    for entry in selected:
        try:
            stat = entry.stat()
        except OSError:
            metadata.append({"name": entry.name, "exists": False})
            continue
        item: dict[str, Any] = {
            "name": entry.name,
            "kind": "directory" if entry.is_dir() else "file",
        }
        if entry.is_file():
            item["content_identity"] = _bounded_file_identity(entry)
        else:
            item["size"] = stat.st_size
        metadata.append(item)
    return {
        "exists": True,
        "identity_mode": "bounded-directory-sample-v1",
        "selection": "lexicographic-evenly-spaced",
        "max_entries": max_entries,
        "entry_count": len(entries),
        "entries": metadata,
        "truncated": len(entries) > max_entries,
    }


def _named_file_assets(
    root_value: str | Path | None,
    names: Sequence[str],
) -> dict[str, Any] | None:
    if root_value in (None, ""):
        return None
    root = Path(root_value).expanduser()
    return {name: file_asset_identity(root / name) for name in names}


def _normalize_command(tokens: Sequence[str]) -> list[Any]:
    normalized: list[Any] = []
    for token in tokens:
        path = Path(token).expanduser()
        if path.is_absolute() and path.is_file():
            normalized.append({"file_asset": file_asset_identity(path)})
        elif path.is_absolute() and path.is_dir():
            normalized.append(
                {"directory_asset": _directory_listing_identity(path, max_entries=64)}
            )
        elif path.is_absolute():
            # Runtime mount points are operational placement, not scientific
            # identity. Keep only the path-free leaf so relocation does not
            # change a run fingerprint.
            normalized.append({"unresolved_absolute_leaf": path.name})
        else:
            normalized.append(token)
    return normalized


@lru_cache(maxsize=1)
def aerith_source_identity() -> dict[str, str | None]:
    """Return version, Git revision, and executable-package source identity."""

    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    source_files = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".yaml", ".yml"}
    )
    for path in source_files:
        try:
            relative = path.relative_to(package_root).as_posix()
            content = path.read_bytes()
        except OSError:
            continue
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
        digest.update(b"\0")

    try:
        version = importlib.metadata.version("aerith")
    except importlib.metadata.PackageNotFoundError:
        from af3_binder_filter import __version__

        version = __version__

    git_commit: str | None = None
    try:
        completed = subprocess.run(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if completed.returncode == 0:
            git_commit = completed.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass

    repository_root: Path | None = next(
        (
            candidate
            for candidate in (package_root, *package_root.parents)
            if (candidate / "pyproject.toml").is_file()
        ),
        None,
    )
    return {
        "package_version": version,
        "git_commit": git_commit,
        "runtime_source_sha256": digest.hexdigest(),
        "pyproject_sha256": (
            file_sha256(repository_root / "pyproject.toml") if repository_root is not None else None
        ),
        "uv_lock_sha256": (
            file_sha256(repository_root / "uv.lock") if repository_root is not None else None
        ),
    }


def output_schema_identity() -> dict[str, Any]:
    """Describe both public CSV schemas without a module-load import cycle."""

    from af3_binder_filter.output_layout import OUTPUT_SCHEMA_VERSION
    from af3_binder_filter.reporting import BACKEND_REVIEW_COLUMNS, DECISION_COLUMNS

    return {
        "version": OUTPUT_SCHEMA_VERSION,
        "decision_columns_sha256": _canonical_digest(list(DECISION_COLUMNS)),
        "backend_review_columns_sha256": _canonical_digest(list(BACKEND_REVIEW_COLUMNS)),
    }


def _without(mapping: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: value for key, value in mapping.items() if key not in keys}


def _backend_identity(settings: Any) -> dict[str, Any]:
    checkpoint_value = settings.checkpoint_path
    if not checkpoint_value and settings.name == "alphafold3":
        checkpoint_value = str(Path(settings.model_dir).expanduser() / "af3.bin")
    identity: dict[str, Any] = {
        "name": settings.name,
        "model": settings.model,
        # The mutable tag is not trusted once docker inspect supplied an ID.
        "runtime_image_id": settings.image_id,
        "runtime_image_reference": None if settings.image_id else settings.image,
        "source_commit": settings.source_commit,
        "runtime_entry": settings.runtime_entry,
        "target_name": settings.target_name,
        "command": _normalize_command(settings.command),
        "checkpoint": file_asset_identity(checkpoint_value),
        "target_data": json_asset_identity(settings.target_data_json),
    }
    if settings.name == "protenix":
        identity["common_assets"] = _named_file_assets(
            settings.common_dir,
            (
                "components.cif",
                "components.cif.rdkit_mol.pkl",
                "obsolete_release_date.csv",
                "clusters-by-entity-40.txt",
            ),
        )
        identity["metadata_assets"] = _named_file_assets(
            settings.metadata_dir,
            ("release_date_cache.json", "obsolete_to_successor.json"),
        )
    elif settings.name == "opendde":
        identity["common_assets"] = _named_file_assets(
            settings.common_dir,
            (
                "components.cif",
                "release_date_cache.json",
                "obsolete_to_successor.json",
            ),
        )
    return identity


def json_asset_identity(value: str | Path | None) -> dict[str, Any] | None:
    """Hash JSON plus every existing file/directory path referenced within it."""

    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    result: dict[str, Any] = {"json": file_asset_identity(path)}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return result
    references: dict[str, Any] = {}

    def walk(item: Any, pointer: str) -> None:
        if isinstance(item, dict):
            for key in sorted(item):
                walk(item[key], f"{pointer}/{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                walk(child, f"{pointer}/{index}")
        elif isinstance(item, str):
            candidate = Path(item).expanduser()
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if candidate.is_file():
                references[pointer] = file_asset_identity(candidate)
            elif candidate.is_dir():
                references[pointer] = _directory_listing_identity(candidate, max_entries=256)

    walk(payload, "")
    result["referenced_assets"] = references
    return result


def feature_database_identity(settings: Any) -> dict[str, Any]:
    """Return the bounded, path-free identity of a local feature database."""

    database_root = Path(settings.database_dir).expanduser()

    def configured_path(value: str, fallback: Path) -> Path:
        return fallback if "${" in value else Path(value).expanduser()

    mmseqs_root = configured_path(settings.mmseqs_dir, database_root / "mmseqs")
    pdb_seqres = configured_path(
        settings.pdb_seqres_fasta,
        database_root / "pdb_seqres_2022_09_28.fasta",
    )
    mmcif_root = configured_path(settings.mmcif_dir, database_root / "mmcif_files")
    database_names = [settings.primary_database, settings.template_database]
    if settings.use_environment_database:
        database_names.append(settings.environment_database)
    return {
        "logical_databases": database_names,
        "mmseqs_prefixes": _directory_listing_identity(
            mmseqs_root,
            prefixes=tuple(database_names),
            max_entries=256,
        ),
        "pdb_seqres": _bounded_file_identity(pdb_seqres),
        "template_mmcif_release": _directory_listing_identity(
            mmcif_root,
            max_entries=96,
        ),
    }


def feature_settings_scientific_identity(
    settings: Any,
    *,
    database_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize settings that can change generated MSA/template content."""

    mmseqs_binary = settings.mmseqs_binary
    return {
        "name": settings.name,
        "runtime_image_id": settings.image_id,
        "runtime_image_reference": None if settings.image_id else settings.image,
        "mmseqs_id": settings.mmseqs_id,
        "mmseqs_binary": (
            file_asset_identity(mmseqs_binary)
            if mmseqs_binary and "/" in mmseqs_binary
            else {"command": Path(mmseqs_binary).name if mmseqs_binary else "runtime:mmseqs"}
        ),
        "use_gpu": settings.use_gpu,
        "split_memory_limit": settings.split_memory_limit,
        "iterations": settings.iterations,
        "primary_database": settings.primary_database,
        "environment_database": settings.environment_database,
        "template_database": settings.template_database,
        "use_environment_database": settings.use_environment_database,
        "timeout_seconds": settings.timeout_seconds,
        "database_release": (
            database_identity
            if database_identity is not None
            else feature_database_identity(settings)
        ),
    }


def local_feature_generation_fingerprint(
    settings: Any,
    target_sequence: str,
    *,
    database_identity: dict[str, Any] | None = None,
) -> str:
    """Fingerprint local MSA/template generation without host placement."""

    return _canonical_digest(
        {
            "mode": "local_target_features_v1",
            "target_sequence_sha256": sequence_sha256(target_sequence),
            "features": feature_settings_scientific_identity(
                settings,
                database_identity=database_identity,
            ),
        }
    )


def feature_generation_fingerprint(
    config: AerithConfig,
    target_sequence: str,
    *,
    scientific_config: dict[str, Any] | None = None,
) -> str:
    """Return the canonical feature-input identity selected by the workflow."""

    if config.backend.name == "alphafold3" and config.backend.target_data_json:
        return _canonical_digest(
            {
                "mode": "alphafold3_target_only",
                "target_sequence_sha256": sequence_sha256(target_sequence),
                "target_chain": config.project.target_chain,
                "seed": config.project.seed,
                "backend_model": config.backend.model,
                "backend_runtime_image_id": config.backend.image_id,
                "backend_runtime_image_reference": (
                    None if config.backend.image_id else config.backend.image
                ),
                "target_data_identity": json_asset_identity(config.backend.target_data_json),
            }
        )
    features = (
        scientific_config["features"]
        if scientific_config is not None
        else feature_settings_scientific_identity(config.features)
    )
    return _canonical_digest(
        {
            "mode": "local_target_features_v1",
            "target_sequence_sha256": sequence_sha256(target_sequence),
            "features": features,
        }
    )


def _rosetta_database_identity(config: AerithConfig) -> dict[str, Any]:
    settings = config.interface.rosetta
    root = Path(settings.database).expanduser()
    score_name = settings.score_function
    return {
        "release": _directory_listing_identity(root, max_entries=64),
        "score_weights": file_asset_identity(root / "scoring" / "weights" / f"{score_name}.wts"),
        "residue_types": file_asset_identity(
            root / "chemical" / "residue_type_sets" / "fa_standard" / "residue_types.txt"
        ),
    }


def scientific_config_identity(config: AerithConfig) -> dict[str, Any]:
    """Normalize every setting that can change a scientific/public result."""

    backend = _backend_identity(config.backend)
    if config.secondary_backend.enabled:
        secondary = _backend_identity(config.secondary_backend)
        secondary["enabled"] = True
        secondary["minimum_primary_iptm"] = config.secondary_backend.minimum_primary_iptm
    else:
        secondary = {"enabled": False, "name": "none"}
    features = feature_settings_scientific_identity(config.features)
    esm_settings = config.scoring.esm
    esm = {
        "enabled": esm_settings.enabled,
        "run_on": esm_settings.run_on,
        "inverse_folding": esm_settings.inverse_folding,
        "esmfold": esm_settings.esmfold,
        "runtime_entry_if": esm_settings.runtime_entry_if,
        "runtime_entry_fold": esm_settings.runtime_entry_fold,
        "timeout_seconds": esm_settings.timeout_seconds,
        "inverse_folding_checkpoint": (
            file_asset_identity(
                Path(esm_settings.model_cache).expanduser()
                / esm_settings.inverse_folding_checkpoint
            )
            if esm_settings.inverse_folding
            else None
        ),
        "esmfold_checkpoint": (
            file_asset_identity(
                Path(esm_settings.model_cache).expanduser() / esm_settings.esmfold_checkpoint
            )
            if esm_settings.esmfold
            else None
        ),
    }
    if not config.scoring.esm.enabled:
        esm = {"enabled": False}
    interface = asdict(config.interface)
    if config.interface.energy_engine == "rosetta_cli":
        interface["rosetta"] = _without(
            dict(interface["rosetta"]), "max_workers", "binary", "database"
        )
        interface["rosetta"]["binary_identity"] = file_asset_identity(
            config.interface.rosetta.binary
        )
        interface["rosetta"]["database_identity"] = _rosetta_database_identity(config)
    else:
        interface["rosetta"] = None
    clustering = _without(asdict(config.clustering), "max_workers", "foldseek_binary")
    clustering["foldseek_binary"] = file_asset_identity(
        config.clustering.foldseek_binary if "/" in config.clustering.foldseek_binary else None
    ) or {"command": Path(config.clustering.foldseek_binary).name}
    return {
        "backend": backend,
        "secondary_backend": secondary,
        "features": features,
        "scoring": {"esm": esm},
        "consensus": asdict(config.consensus),
        "interface": interface,
        "clustering": clustering,
        "runtime_tools": {
            "mmseqs_release": config.runtime.mmseqs_release,
            "mmseqs_version": config.runtime.mmseqs_version,
            "mmseqs_archive_sha256": config.runtime.mmseqs_archive_sha256,
            "foldseek_release": config.runtime.foldseek_release,
            "foldseek_version": config.runtime.foldseek_version,
            "foldseek_archive_sha256": config.runtime.foldseek_archive_sha256,
        },
    }


def run_provenance(
    plan: JobPlan,
    config: AerithConfig,
) -> dict[str, Any]:
    """Build the canonical, path-independent provenance payload for one run."""

    # The shared generation fingerprint covers the bounded database release
    # identity, immutable image ID, target, and every output-affecting feature
    # parameter without host paths or worker counts. The exact generated
    # artifact digest is added to RunManifest after feature preparation.
    scientific_config = scientific_config_identity(config)
    feature_fingerprint = feature_generation_fingerprint(
        config,
        plan.target_sequence,
        scientific_config=scientific_config,
    )
    return {
        "jobs": [asdict(job) for job in plan.jobs],
        "total_csv_jobs": plan.total_csv_jobs,
        "source_csv_sha256": file_sha256(plan.source_csv),
        "target_sequence": plan.target_sequence,
        "target_chain": config.project.target_chain,
        "binder_chain": config.project.binder_chain,
        "scientific_config": scientific_config,
        "feature_generation_identity_sha256": feature_fingerprint,
        "output_schema": output_schema_identity(),
        "aerith": aerith_source_identity(),
    }


def checkpoint_identity(value: str | None) -> dict[str, Any] | None:
    """Return a path-free content identity for a checkpoint-like asset."""

    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        return _directory_listing_identity(path, max_entries=128)
    return file_asset_identity(path)


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
            "runtime_image_id": config.backend.image_id,
            "runtime_image_reference": (None if config.backend.image_id else config.backend.image),
            "source_commit": config.backend.source_commit,
            "checkpoint": checkpoint_identity(config.backend.checkpoint_path),
            "secondary_backend": config.secondary_backend.name,
            "secondary_model": config.secondary_backend.model,
            "secondary_runtime_image_id": config.secondary_backend.image_id,
            "secondary_runtime_image_reference": (
                None if config.secondary_backend.image_id else config.secondary_backend.image
            ),
            "secondary_source_commit": config.secondary_backend.source_commit,
            "secondary_checkpoint": checkpoint_identity(config.secondary_backend.checkpoint_path),
            "secondary_minimum_primary_iptm": config.secondary_backend.minimum_primary_iptm,
            "feature_mode": config.features.name,
            "feature_runtime_image_id": config.features.image_id,
            "feature_runtime_image_reference": (
                None if config.features.image_id else config.features.image
            ),
            "feature_fingerprint": feature_fingerprint,
            "esm": asdict(config.scoring.esm),
            "consensus": asdict(config.consensus),
        }
    )


def run_fingerprint(
    plan: JobPlan,
    config: AerithConfig,
    *,
    provenance: dict[str, Any] | None = None,
) -> str:
    return _canonical_digest(
        provenance
        if provenance is not None
        else run_provenance(
            plan,
            config,
        )
    )


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()
