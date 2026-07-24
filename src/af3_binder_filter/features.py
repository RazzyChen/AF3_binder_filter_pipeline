"""Offline target-feature cache and query-only binder MSA generation."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from af3_binder_filter.af3_json import TargetFeatures
from af3_binder_filter.config import FeatureSettings
from af3_binder_filter.execution import CommandSpec, LocalDockerExecutor
from af3_binder_filter.io_utils import atomic_write_json, atomic_write_text
from af3_binder_filter.jobs import (
    feature_database_identity,
    local_feature_generation_fingerprint,
    sequence_sha256,
)

FEATURE_MANIFEST_VERSION = 3


class FeatureError(RuntimeError):
    """Raised when offline feature preparation is incomplete or inconsistent."""


@contextmanager
def _feature_cache_lock(cache_dir: Path):
    """Serialize expensive feature publication for one target cache key."""

    lock_path = cache_dir / ".prepare.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_bundle_artifact_identity(bundle: "FeatureBundle") -> dict[str, object]:
    """Return path-free, complete identities for a prepared feature bundle."""

    files = {
        "pairing_a3m": bundle.pairing_a3m,
        "non_pairing_a3m": bundle.non_pairing_a3m,
        "hmmsearch_a3m": bundle.hmmsearch_a3m,
        "af3_templates_json": bundle.af3_templates_json,
    }
    identities: dict[str, object] = {
        name: {"size": path.stat().st_size, "sha256": _artifact_sha256(path)}
        for name, path in files.items()
    }
    identities["template_mmcif_files"] = {
        path.relative_to(bundle.template_mmcif_dir).as_posix(): {
            "size": path.stat().st_size,
            "sha256": _artifact_sha256(path),
        }
        for path in sorted(bundle.template_mmcif_dir.rglob("*"))
        if path.is_file()
    }
    return identities


def af3_feature_bundle_artifact_identity(
    bundle: "AF3FeatureBundle",
) -> dict[str, object]:
    """Return path-free, complete identities for externalized AF3 features."""

    templates: list[dict[str, object]] = []
    for template in bundle.features.templates:
        mmcif_value = template.get("mmcifPath")
        mmcif = Path(str(mmcif_value)) if mmcif_value not in (None, "") else None
        templates.append(
            {
                "mmcif": (
                    {
                        "size": mmcif.stat().st_size,
                        "sha256": _artifact_sha256(mmcif),
                    }
                    if mmcif is not None and mmcif.is_file()
                    else None
                ),
                "query_indices": list(template.get("queryIndices") or []),
                "template_indices": list(template.get("templateIndices") or []),
            }
        )

    def identity(value: str | Path | None) -> dict[str, object] | None:
        if value in (None, ""):
            return None
        path = Path(str(value))
        if not path.is_file():
            return None
        return {"size": path.stat().st_size, "sha256": _artifact_sha256(path)}

    return {
        "target_data": identity(bundle.target_data_json),
        "unpaired_msa": identity(bundle.features.unpaired_msa_path),
        "paired_msa": identity(bundle.features.paired_msa_path),
        "templates": templates,
    }


def feature_bundle_content_sha256(
    bundle: "FeatureBundle | AF3FeatureBundle",
) -> str:
    """Hash the exact generated feature artifacts, independent of placement."""

    identity = (
        af3_feature_bundle_artifact_identity(bundle)
        if isinstance(bundle, AF3FeatureBundle)
        else feature_bundle_artifact_identity(bundle)
    )
    return sequence_sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    )


@dataclass(frozen=True, slots=True)
class FeatureBundle:
    sequence_sha256: str
    cache_dir: Path
    pairing_a3m: Path
    non_pairing_a3m: Path
    hmmsearch_a3m: Path
    fingerprint: str
    af3_templates_json: Path
    template_mmcif_dir: Path
    source_mmcif_dir: Path | None = None

    def validate(self) -> None:
        for path in (self.pairing_a3m, self.non_pairing_a3m, self.hmmsearch_a3m):
            if not path.is_file() or path.stat().st_size == 0:
                raise FeatureError(f"feature output is missing or empty: {path}")
        if not self.af3_templates_json.is_file():
            raise FeatureError(f"AF3 template manifest is missing: {self.af3_templates_json}")
        if not self.template_mmcif_dir.is_dir():
            raise FeatureError(f"AF3 template directory is missing: {self.template_mmcif_dir}")
        try:
            payload = json.loads(self.af3_templates_json.read_text(encoding="utf-8"))
            if (
                payload.get("templates")
                and self.source_mmcif_dir is not None
                and not self.source_mmcif_dir.is_dir()
            ):
                raise FeatureError(
                    f"source template mmCIF directory is missing: {self.source_mmcif_dir}"
                )
            for template in payload.get("templates", []):
                filename = Path(str(template["mmcifFile"])).name
                if not (self.template_mmcif_dir / filename).is_file():
                    raise FeatureError(f"AF3 template mmCIF is missing: {filename}")
                query_indices = template.get("queryIndices") or []
                template_indices = template.get("templateIndices") or []
                if not query_indices or len(query_indices) != len(template_indices):
                    raise FeatureError(f"AF3 template has an invalid residue mapping: {filename}")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise FeatureError(
                f"AF3 template manifest is invalid: {self.af3_templates_json}: {exc}"
            ) from exc

    def af3_templates(self) -> list[dict[str, object]]:
        self.validate()
        payload = json.loads(self.af3_templates_json.read_text(encoding="utf-8"))
        return [
            {
                "mmcifPath": str(
                    (self.template_mmcif_dir / Path(str(template["mmcifFile"])).name).resolve()
                ),
                "queryIndices": [int(value) for value in template["queryIndices"]],
                "templateIndices": [int(value) for value in template["templateIndices"]],
            }
            for template in payload.get("templates", [])
        ]


@dataclass(frozen=True, slots=True)
class AF3FeatureBundle:
    """Externalized features created by an AF3 target-only preprocessing run."""

    sequence_sha256: str
    cache_dir: Path
    target_data_json: Path
    features: TargetFeatures
    fingerprint: str

    def validate(self) -> None:
        if not self.target_data_json.is_file():
            raise FeatureError(f"AF3 target data JSON is missing: {self.target_data_json}")
        if not self.features.unpaired_msa_path:
            raise FeatureError("AF3 target data has no unpaired target MSA")
        for value in (
            self.features.unpaired_msa_path,
            self.features.paired_msa_path,
        ):
            if value and not Path(value).is_file():
                raise FeatureError(f"AF3 target MSA is missing: {value}")
        for template in self.features.templates:
            path = template.get("mmcifPath")
            if not path or not Path(str(path)).is_file():
                raise FeatureError(f"AF3 template mmCIF is missing: {path}")


@dataclass(frozen=True, slots=True)
class FeaturePreparation:
    bundle: FeatureBundle | AF3FeatureBundle | None
    command: tuple[str, ...] | None
    reused: bool = False


def query_only_a3m(sequence: str) -> str:
    normalized = "".join(sequence.split()).upper()
    if not normalized:
        raise FeatureError("cannot create a query-only MSA for an empty sequence")
    return f">query\n{normalized}\n"


def _msa_query_matches(path: Path, sequence: str) -> bool:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return len(lines) >= 2 and lines[0] == ">query" and lines[1] == sequence


def write_query_only_msa(sequence: str, output_path: Path, *, force: bool = False) -> Path:
    expected = query_only_a3m(sequence)
    if output_path.exists() and not force:
        if output_path.read_text(encoding="utf-8") != expected:
            raise FeatureError(f"existing query-only MSA has a different sequence: {output_path}")
        return output_path
    atomic_write_text(output_path, expected)
    return output_path


def target_feature_dir(settings: FeatureSettings, target_sequence: str) -> Path:
    return Path(settings.cache_dir).expanduser() / sequence_sha256(target_sequence)


def _bundle(
    settings: FeatureSettings,
    target_sequence: str,
    *,
    database_identity: dict[str, object] | None = None,
) -> FeatureBundle:
    digest = sequence_sha256(target_sequence)
    root = target_feature_dir(settings, target_sequence)
    fingerprint = local_feature_generation_fingerprint(
        settings,
        target_sequence,
        database_identity=database_identity,
    )
    return FeatureBundle(
        sequence_sha256=digest,
        cache_dir=root,
        pairing_a3m=root / "pairing.a3m",
        non_pairing_a3m=root / "non_pairing.a3m",
        hmmsearch_a3m=root / "hmmsearch.a3m",
        fingerprint=fingerprint,
        af3_templates_json=root / "af3_templates.json",
        template_mmcif_dir=root / "templates",
        source_mmcif_dir=(Path(settings.database_dir).expanduser().resolve() / "mmcif_files"),
    )


def cached_target_features(
    settings: FeatureSettings,
    target_sequence: str,
    *,
    database_identity: dict[str, object] | None = None,
) -> FeatureBundle | None:
    bundle = _bundle(
        settings,
        target_sequence,
        database_identity=database_identity,
    )
    manifest_path = bundle.cache_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("version") != FEATURE_MANIFEST_VERSION
            or manifest.get("sequence_sha256") != bundle.sequence_sha256
            or manifest.get("fingerprint") != bundle.fingerprint
        ):
            return None
        bundle.validate()
        if manifest.get("artifact_identities") != feature_bundle_artifact_identity(bundle):
            return None
        # The query must be the first sequence in both MSA files.
        expected_query = "".join(target_sequence.split()).upper()
        for msa_path in (
            bundle.pairing_a3m,
            bundle.non_pairing_a3m,
            bundle.hmmsearch_a3m,
        ):
            if not _msa_query_matches(msa_path, expected_query):
                return None
        return bundle
    except (OSError, ValueError, json.JSONDecodeError, FeatureError):
        return None


def build_feature_builder_command(
    settings: FeatureSettings,
    *,
    target_sequence: str,
    output_dir: Path,
    gpu_index: int = 0,
    container_name: str | None = None,
) -> tuple[list[str], Path]:
    database = Path(settings.database_dir).expanduser().resolve()
    output = output_dir.expanduser().resolve()
    query_fasta = output / "query.fasta"
    atomic_write_text(query_fasta, f">query\n{''.join(target_sequence.split()).upper()}\n")
    command = [
        settings.docker_bin,
        "run",
        "--rm",
        "--network",
        "none",
    ]
    if container_name:
        command.extend(["--name", container_name])
    if settings.use_gpu:
        command.extend(["--gpus", f"device={gpu_index}"])
    command.extend(
        [
            "--volume",
            f"{database}:/db:ro",
            "--volume",
            f"{output}:/output",
        ]
    )
    if settings.mmseqs_binary is not None:
        raise FeatureError(
            "features.mmseqs_binary host overrides are disabled; "
            "the pinned GPU MMseqs2 binary must come from the fold-runtime image"
        )
    command.extend(
        [
            settings.image,
            "prepare-features",
            "--query",
            "/output/query.fasta",
            "--output",
            "/output",
            "--mmseqs-db",
            "/db/mmseqs",
            "--pdb-seqres",
            "/db/pdb_seqres_2022_09_28.fasta",
            "--mmcif-dir",
            "/db/mmcif_files",
            "--mmseqs-binary",
            "mmseqs",
            "--use-gpu",
            "1" if settings.use_gpu else "0",
            "--threads",
            str(settings.threads),
            "--split-memory-limit",
            settings.split_memory_limit,
            "--iterations",
            str(settings.iterations),
            "--primary-database",
            settings.primary_database,
            "--environment-database",
            settings.environment_database,
            "--template-database",
            settings.template_database,
            "--use-environment-database",
            "1" if settings.use_environment_database else "0",
        ]
    )
    return command, query_fasta


def prepare_target_features(
    settings: FeatureSettings,
    target_sequence: str,
    *,
    dry_run: bool = False,
    force: bool = False,
    gpu_index: int = 0,
    log_dir: Path | None = None,
    database_identity: dict[str, object] | None = None,
    container_name: str | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    _lock_acquired: bool = False,
) -> FeaturePreparation:
    selected_database_identity = (
        database_identity if database_identity is not None else feature_database_identity(settings)
    )
    cached = (
        None
        if force
        else cached_target_features(
            settings,
            target_sequence,
            database_identity=selected_database_identity,
        )
    )
    if cached is not None:
        return FeaturePreparation(cached, None, reused=True)

    bundle = _bundle(
        settings,
        target_sequence,
        database_identity=selected_database_identity,
    )
    bundle.cache_dir.mkdir(parents=True, exist_ok=True)
    if not dry_run and not _lock_acquired:
        with _feature_cache_lock(bundle.cache_dir):
            # A concurrent run may have published the exact same feature set
            # while this process waited. Re-enter once under the lock so the
            # cache is validated again before spending GPU time.
            return prepare_target_features(
                settings,
                target_sequence,
                dry_run=dry_run,
                force=force,
                gpu_index=gpu_index,
                log_dir=log_dir,
                database_identity=selected_database_identity,
                container_name=container_name,
                runner=runner,
                _lock_acquired=True,
            )
    if dry_run:
        command, _query_fasta = build_feature_builder_command(
            settings,
            target_sequence=target_sequence,
            output_dir=bundle.cache_dir,
            gpu_index=gpu_index,
            container_name=container_name,
        )
        if log_dir is not None:
            atomic_write_text(
                log_dir / "prepare_features.command.txt",
                " ".join(command) + "\n",
            )
        return FeaturePreparation(None, tuple(command), reused=False)

    build_dir = Path(tempfile.mkdtemp(prefix=".feature-build-", dir=bundle.cache_dir))
    command, _query_fasta = build_feature_builder_command(
        settings,
        target_sequence=target_sequence,
        output_dir=build_dir,
        gpu_index=gpu_index,
        container_name=container_name,
    )
    if log_dir is not None:
        atomic_write_text(
            log_dir / "prepare_features.command.txt",
            " ".join(command) + "\n",
        )
    try:
        if runner is None:
            execution_log_dir = log_dir or (build_dir / ".execution_logs")
            outcome = LocalDockerExecutor(
                docker_executable=settings.docker_bin,
            ).run(
                CommandSpec.logged(
                    command,
                    log_dir=execution_log_dir,
                    name="prepare_features",
                    timeout_seconds=settings.timeout_seconds,
                    stage="features",
                    shard_id=gpu_index,
                )
            )
            returncode = outcome.returncode
            stderr_text = (
                outcome.command.stderr_path.read_text(
                    encoding="utf-8",
                    errors="replace",
                )
                if outcome.command.stderr_path.is_file()
                else ""
            )
            if outcome.error is not None:
                raise FeatureError(f"feature-builder execution failed: {outcome.error}")
        else:
            try:
                completed = runner(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=settings.timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise FeatureError(f"feature-builder execution failed: {exc}") from exc
            returncode = completed.returncode
            stderr_text = completed.stderr or ""
            if log_dir is not None:
                atomic_write_text(
                    log_dir / "prepare_features.stdout.log",
                    completed.stdout or "",
                )
                atomic_write_text(
                    log_dir / "prepare_features.stderr.log",
                    stderr_text,
                )
        if returncode != 0:
            raise FeatureError(
                f"feature-builder failed with return code {returncode}: {stderr_text.strip()}"
            )
        built_paths = {
            bundle.pairing_a3m: build_dir / "pairing.a3m",
            bundle.non_pairing_a3m: build_dir / "non_pairing.a3m",
            bundle.hmmsearch_a3m: build_dir / "hmmsearch.a3m",
            bundle.af3_templates_json: build_dir / "af3_templates.json",
        }
        for source in (
            build_dir / "pairing.a3m",
            build_dir / "non_pairing.a3m",
            build_dir / "hmmsearch.a3m",
        ):
            if not source.is_file() or source.stat().st_size == 0:
                raise FeatureError(f"feature-builder output is missing or empty: {source}")
            if not _msa_query_matches(
                source,
                "".join(target_sequence.split()).upper(),
            ):
                raise FeatureError(f"feature-builder output query does not match target: {source}")
        for destination, source in built_paths.items():
            if not source.is_file() or source.stat().st_size == 0:
                raise FeatureError(f"feature-builder output is missing or empty: {source}")
            os.replace(source, destination)
        built_template_dir = build_dir / "templates"
        if not built_template_dir.is_dir():
            raise FeatureError(
                f"feature-builder template directory is missing: {built_template_dir}"
            )
        shutil.rmtree(bundle.template_mmcif_dir, ignore_errors=True)
        os.replace(built_template_dir, bundle.template_mmcif_dir)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)
    bundle.validate()
    atomic_write_json(
        bundle.cache_dir / "manifest.json",
        {
            "version": FEATURE_MANIFEST_VERSION,
            "sequence_sha256": bundle.sequence_sha256,
            "sequence": "".join(target_sequence.split()).upper(),
            "fingerprint": bundle.fingerprint,
            "feature_mode": settings.name,
            "image": settings.image,
            "image_id": settings.image_id,
            "docker_bin": settings.docker_bin,
            "database_dir": str(Path(settings.database_dir).expanduser()),
            "mmseqs_binary": settings.mmseqs_binary,
            "mmseqs_id": settings.mmseqs_id,
            "threads": settings.threads,
            "split_memory_limit": settings.split_memory_limit,
            "iterations": settings.iterations,
            "primary_database": settings.primary_database,
            "environment_database": settings.environment_database,
            "template_database": settings.template_database,
            "use_environment_database": settings.use_environment_database,
            "artifacts": {
                key: str(value) for key, value in asdict(bundle).items() if isinstance(value, Path)
            },
            "artifact_identities": feature_bundle_artifact_identity(bundle),
        },
    )
    return FeaturePreparation(bundle, tuple(command), reused=False)
