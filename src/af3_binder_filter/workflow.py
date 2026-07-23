"""Hydra-driven end-to-end orchestration with partial-result preservation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from omegaconf import OmegaConf

from af3_binder_filter.backends import (
    UnifiedPrediction,
    build_backend_command,
    output_adapter,
    write_backend_inputs,
)
from af3_binder_filter.af3_json import TargetFeatures, write_target_input
from af3_binder_filter.clustering import (
    build_foldseek_container_command,
    greedy_epitope_clusters,
    parse_foldseek_clusters,
    prepare_foldseek_inputs,
    run_foldseek_command,
    write_cluster_outputs,
)
from af3_binder_filter.config import (
    AerithConfig,
    BackendSettings,
    compose_hydra_config,
    validate_hydra_config,
)
from af3_binder_filter.config_tools import resolve_docker_image_id
from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeatureBundle,
    FeaturePreparation,
    _bundle as expected_feature_bundle,
    af3_feature_bundle_artifact_identity,
    cached_target_features,
    feature_bundle_content_sha256,
    prepare_target_features,
)
from af3_binder_filter.interface import (
    analyze_interface_geometry,
    apply_balanced_shortlist,
    structure_has_chains,
)
from af3_binder_filter.io_utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    atomic_write_yaml,
)
from af3_binder_filter.jobs import (
    JobPlan,
    JobSpec,
    build_job_plan,
    checkpoint_identity,
    feature_generation_fingerprint,
    file_sha256,
    job_fingerprint,
    run_fingerprint,
    run_provenance,
    sequence_sha256,
)
from af3_binder_filter.gpu import GPUError, GPUInfo, query_gpus, select_free_gpus
from af3_binder_filter.manifest import (
    JOB_MANIFEST_NAME,
    MANIFEST_VERSION,
    RunManifest,
    load_manifest,
    validate_legacy_input,
    write_job_manifest,
)
from af3_binder_filter.rosetta import RosettaCliEngine
from af3_binder_filter.output_layout import (
    OUTPUT_SCHEMA_VERSION,
    RunOutputLayout,
    STAGE_DIRECTORIES,
)
from af3_binder_filter.progress import (
    NullProgressReporter,
    PipelineProgressReporter,
    PipelineRunInfo,
    StageSpec,
)
from af3_binder_filter.reporting import write_public_reports
from af3_binder_filter.target_data import extract_target_features
from af3_binder_filter.secondary_features import (
    SecondaryFeatureBundle,
    secondary_feature_bundle_content_sha256,
)
from af3_binder_filter.secondary_features import adapt_af3_features_for_secondary
from af3_binder_filter.secondary_features import adapt_local_features_for_secondary
from af3_binder_filter.consensus import consensus_rows
from af3_binder_filter.effective import apply_effective_backend
from af3_binder_filter.esm_tools import (
    add_esmfold_backend_comparison,
    build_esm_if_container_command,
    build_esmfold_container_command,
    collect_esm_rows,
    load_cached_esm_rows,
    write_esm_inputs,
)


class PipelineExecutionError(RuntimeError):
    """Raised after partial artifacts are preserved for a required-stage failure."""


@dataclass(frozen=True, slots=True)
class RunContext:
    config: AerithConfig
    resolved_config: Any
    plan: JobPlan
    fingerprint: str
    run_id: str
    results_dir: Path
    manifest_path: Path
    provenance: dict[str, Any] | None = None
    feature_fingerprint: str | None = None
    feature_database_identity: dict[str, Any] | None = None

    @property
    def layout(self) -> RunOutputLayout:
        return RunOutputLayout(self.results_dir).ensure()


@dataclass(frozen=True, slots=True)
class ClusteringOutcome:
    failed: bool
    member_rows: tuple[dict[str, Any], ...] = ()
    representative_rows: tuple[dict[str, Any], ...] = ()
    final_rows: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True, slots=True)
class GpuJobShard:
    """A deterministic set of jobs assigned to one physical host GPU."""

    gpu: GPUInfo
    jobs: tuple[JobSpec, ...]

    @property
    def estimated_cost(self) -> int:
        return sum(_job_estimated_cost(job) for job in self.jobs)


def _job_estimated_cost(job: JobSpec) -> int:
    """A stable, model-agnostic proxy for fold inference work."""

    total_residues = len(job.target_sequence) + len(job.binder_sequence)
    return total_residues * total_residues


def plan_gpu_job_shards(
    jobs: Sequence[JobSpec],
    gpus: Sequence[GPUInfo],
) -> list[GpuJobShard]:
    """Balance jobs using deterministic longest-processing-time assignment."""

    if not jobs:
        return []
    selected = list(gpus[: min(len(jobs), len(gpus))])
    if not selected:
        raise PipelineExecutionError("jobs are pending but no free GPU is available")
    buckets: list[list[JobSpec]] = [[] for _ in selected]
    loads = [0] * len(selected)
    ordered = sorted(jobs, key=lambda job: (-_job_estimated_cost(job), job.job_id))
    for job in ordered:
        index = min(range(len(selected)), key=lambda value: (loads[value], selected[value].index))
        buckets[index].append(job)
        loads[index] += _job_estimated_cost(job)
    return [
        GpuJobShard(gpu, tuple(bucket))
        for gpu, bucket in zip(selected, buckets, strict=True)
    ]


def _runtime_gpus(context: RunContext, *, job_count: int, stage_name: str) -> list[GPUInfo]:
    """Return free configured GPUs, or deterministic placeholders for dry-run."""

    if job_count < 1:
        return []
    allowed = context.config.runtime.gpu_ids or None
    if context.config.runtime.dry_run:
        if allowed:
            ids = list(allowed)
        else:
            try:
                ids = [gpu.index for gpu in query_gpus()]
            except GPUError:
                ids = [0]
        return [
            GPUInfo(index, "dry-run", 0, 0)
            for index in ids[:job_count]
        ]
    try:
        free = select_free_gpus(
            query_gpus(),
            threshold_mib=context.config.runtime.gpu_busy_threshold_mib,
            allowed_gpu_ids=allowed,
        )
    except GPUError as exc:
        raise PipelineExecutionError(
            f"{stage_name} cannot query available GPUs: {exc}"
        ) from exc
    if not free:
        allowed_text = (
            ",".join(str(index) for index in allowed)
            if allowed is not None
            else "all"
        )
        raise PipelineExecutionError(
            f"{stage_name} has pending jobs but no free GPU is available "
            f"(allowed={allowed_text}, threshold="
            f"{context.config.runtime.gpu_busy_threshold_mib} MiB)"
        )
    return free[:job_count]


def _container_name(context: RunContext, stage_name: str, gpu_index: int) -> str:
    raw = f"aerith-{context.run_id}-{stage_name}-gpu{gpu_index}"
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-.")
    return cleaned[:120]


def _record_gpu_assignments(
    manifest: RunManifest,
    path: Path,
    stage_name: str,
    shards: Sequence[GpuJobShard],
) -> None:
    manifest.gpu_assignments[stage_name] = [
        {
            "gpu_index": shard.gpu.index,
            "jobs": [job.job_id for job in shard.jobs],
            "estimated_cost": shard.estimated_cost,
        }
        for shard in shards
    ]
    manifest.write(path)


def _overrides_with_runtime(
    overrides: Iterable[str],
    *,
    limit: int | None = None,
    dry_run: bool | None = None,
) -> list[str]:
    result = list(overrides)
    if limit is not None:
        result.append(f"project.limit={limit}")
    if dry_run is not None:
        result.append(f"runtime.dry_run={'true' if dry_run else 'false'}")
    return result


def _af3_feature_fingerprint(config: AerithConfig, target_sequence: str) -> str:
    return feature_generation_fingerprint(config, target_sequence)


def _expected_feature_fingerprint(
    config: AerithConfig,
    target_sequence: str,
) -> str:
    if config.backend.name == "alphafold3" and config.backend.target_data_json:
        return _af3_feature_fingerprint(config, target_sequence)
    return expected_feature_bundle(config.features, target_sequence).fingerprint


def _expected_feature_cache_dir(
    config: AerithConfig,
    target_sequence: str,
    *,
    fingerprint: str | None = None,
) -> Path:
    if config.backend.name == "alphafold3" and config.backend.target_data_json:
        digest = sequence_sha256(target_sequence)
        selected = fingerprint or _af3_feature_fingerprint(config, target_sequence)
        return (
            Path(config.features.cache_dir).expanduser()
            / digest
            / "af3"
            / selected[:16]
        )
    return (
        Path(config.features.cache_dir).expanduser()
        / sequence_sha256(target_sequence)
    )


def _target_feature_cache_hit(context: RunContext) -> bool:
    """Return whether the exact target feature bundle is reusable."""

    if context.config.runtime.force:
        return False
    if (
        context.config.backend.name == "alphafold3"
        and context.config.backend.target_data_json
    ):
        fingerprint = _context_feature_fingerprint(context)
        return (
            _af3_bundle_from_manifest(
                _expected_feature_cache_dir(
                    context.config,
                    context.plan.target_sequence,
                    fingerprint=fingerprint,
                ),
                target_sequence=context.plan.target_sequence,
                fingerprint=fingerprint,
            )
            is not None
        )
    return (
        cached_target_features(
            context.config.features,
            context.plan.target_sequence,
            database_identity=context.feature_database_identity,
        )
        is not None
    )


def _pipeline_stage_specs(config: AerithConfig) -> tuple[StageSpec, ...]:
    stages = [
        StageSpec("features", "MSA/template searching"),
        StageSpec("primary_prediction", "Primary prediction"),
        StageSpec("primary_interface", "Primary interface analysis"),
    ]
    if config.secondary_backend.enabled:
        stages.extend(
            (
                StageSpec("secondary_features", "Secondary feature adaptation"),
                StageSpec("secondary_prediction", "Secondary prediction"),
                StageSpec("secondary_interface", "Secondary interface analysis"),
            )
        )
    stages.extend(
        (StageSpec("consensus", "Backend consensus / effective selection"),)
    )
    if config.scoring.esm.enabled:
        stages.append(StageSpec("esm", "Effective ESMFold / ESM-IF scoring"))
    stages.append(StageSpec("clustering", "Foldseek / epitope clustering"))
    return tuple(stages)


def create_run_context(
    config_path: Path,
    *,
    backend: str | None = None,
    secondary_backend: str | None = None,
    overrides: Iterable[str] = (),
    limit: int | None = None,
    dry_run: bool | None = None,
    initialize_run: bool = True,
) -> RunContext:
    config, resolved = compose_hydra_config(
        config_path,
        backend=backend,
        secondary_backend=secondary_backend,
        overrides=_overrides_with_runtime(overrides, limit=limit, dry_run=dry_run),
    )

    inspected_images: dict[tuple[str, str], str | None] = {}

    def verified_image_id(settings: Any, label: str) -> str:
        image_key = (settings.docker_bin, settings.image)
        if image_key not in inspected_images:
            inspected_images[image_key] = resolve_docker_image_id(*image_key)
        actual = inspected_images[image_key]
        if not actual:
            raise PipelineExecutionError(
                f"cannot resolve the actual Docker image ID for {label}: "
                f"{settings.image}"
            )
        if settings.image_id is not None and settings.image_id != actual:
            raise PipelineExecutionError(
                f"configured {label} image_id {settings.image_id!r} does not "
                f"match actual image ID {actual!r} for {settings.image}"
            )
        return actual

    config.backend.image_id = verified_image_id(config.backend, "primary backend")
    config.backend.image = config.backend.image_id
    OmegaConf.update(
        resolved,
        "backend.image_id",
        config.backend.image_id,
        merge=False,
    )
    OmegaConf.update(resolved, "backend.image", config.backend.image, merge=False)
    if config.secondary_backend.enabled:
        config.secondary_backend.image_id = verified_image_id(
            config.secondary_backend, "secondary backend"
        )
        config.secondary_backend.image = config.secondary_backend.image_id
        OmegaConf.update(
            resolved,
            "secondary_backend.image_id",
            config.secondary_backend.image_id,
            merge=False,
        )
        OmegaConf.update(
            resolved,
            "secondary_backend.image",
            config.secondary_backend.image,
            merge=False,
        )
    config.features.image_id = verified_image_id(config.features, "feature builder")
    config.features.image = config.features.image_id
    OmegaConf.update(
        resolved,
        "features.image_id",
        config.features.image_id,
        merge=False,
    )
    OmegaConf.update(resolved, "features.image", config.features.image, merge=False)
    if config.features.mmseqs_id is None:
        config.features.mmseqs_id = (
            f"{config.features.image_id}:mmseqs:{config.runtime.mmseqs_version}"
        )
        OmegaConf.update(
            resolved,
            "features.mmseqs_id",
            config.features.mmseqs_id,
            merge=False,
        )
    plan = build_job_plan(config)
    provenance = run_provenance(plan, config)
    feature_fingerprint = str(
        provenance["feature_generation_identity_sha256"]
    )
    feature_database_release = provenance["scientific_config"]["features"].get(
        "database_release"
    )
    fingerprint = run_fingerprint(
        plan,
        config,
        provenance=provenance,
    )
    run_id = config.project.run_id or f"run-{fingerprint[:12]}"
    results_dir = Path(config.project.results_dir) / run_id
    context = RunContext(
        config=config,
        resolved_config=resolved,
        plan=plan,
        fingerprint=fingerprint,
        run_id=run_id,
        results_dir=results_dir,
        manifest_path=results_dir / "manifest.json",
        provenance=provenance,
        feature_fingerprint=feature_fingerprint,
        feature_database_identity=(
            dict(feature_database_release)
            if isinstance(feature_database_release, dict)
            else None
        ),
    )
    existing_manifest: RunManifest | None = None
    if results_dir.exists():
        if not results_dir.is_dir():
            raise PipelineExecutionError(
                f"run output path exists and is not a directory: {results_dir}"
            )
        try:
            nonempty = next(results_dir.iterdir(), None) is not None
        except OSError as exc:
            raise PipelineExecutionError(
                f"cannot inspect existing run directory {results_dir}: {exc}"
            ) from exc
        if nonempty:
            payload = load_manifest(context.manifest_path)
            if payload is None:
                raise PipelineExecutionError(
                    "existing non-empty run directory has no valid manifest; "
                    f"refusing to overwrite {results_dir}"
                )
            existing_manifest = _manifest_from_payload(
                context,
                feature_fingerprint,
                payload,
                verify_resolved_config=True,
            )
        elif not initialize_run:
            raise PipelineExecutionError(
                f"standalone run directory is empty: {results_dir}"
            )
    elif not initialize_run:
        raise PipelineExecutionError(
            f"standalone run directory does not exist: {results_dir}"
        )
    if not initialize_run:
        if existing_manifest is None:
            raise PipelineExecutionError(
                f"standalone run has no validated manifest: {results_dir}"
            )
        return context
    RunOutputLayout(results_dir).ensure()
    resolved_container = OmegaConf.to_container(resolved, resolve=True)
    resolved_config_path = results_dir / "resolved_config.yaml"
    atomic_write_yaml(resolved_config_path, resolved_container)
    manifest = existing_manifest or _new_manifest(context, feature_fingerprint)
    manifest.feature_fingerprint = feature_fingerprint
    manifest.job_fingerprints = {
        job.job_id: job_fingerprint(
            job,
            config,
            feature_fingerprint=feature_fingerprint,
        )
        for job in plan.jobs
    }
    manifest.resolved_config_sha256 = file_sha256(resolved_config_path)
    manifest.provenance = _context_provenance(
        context,
        feature_fingerprint,
    )
    manifest.write(context.manifest_path)
    return context


def _new_manifest(context: RunContext, feature_fingerprint: str) -> RunManifest:
    provenance = _context_provenance(context, feature_fingerprint)
    return RunManifest(
        run_id=context.run_id,
        fingerprint=context.fingerprint,
        backend=context.config.backend.name,
        model=context.config.backend.model,
        source_csv=str(context.plan.source_csv),
        target_sequence_sha256=sequence_sha256(context.plan.target_sequence),
        job_fingerprints={
            job.job_id: job_fingerprint(
                job,
                context.config,
                feature_fingerprint=feature_fingerprint,
            )
            for job in context.plan.jobs
        },
        secondary_backend=context.config.secondary_backend.name,
        secondary_model=context.config.secondary_backend.model,
        primary_image_id=context.config.backend.image_id,
        secondary_image_id=context.config.secondary_backend.image_id,
        feature_fingerprint=feature_fingerprint,
        source_csv_sha256=provenance.get("source_csv_sha256"),
        resolved_config_sha256=file_sha256(
            context.results_dir / "resolved_config.yaml"
        ),
        provenance=provenance,
    )


def _context_provenance(
    context: RunContext,
    feature_fingerprint: str,
) -> dict[str, Any]:
    if context.provenance is not None:
        return context.provenance
    return run_provenance(
        context.plan,
        context.config,
    )


def _context_feature_fingerprint(context: RunContext) -> str:
    return getattr(context, "feature_fingerprint", None) or _expected_feature_fingerprint(
        context.config,
        context.plan.target_sequence,
    )


def _existing_or_new_manifest(
    context: RunContext,
    feature_fingerprint: str,
) -> RunManifest:
    payload = load_manifest(context.manifest_path)
    if payload is None:
        if context.manifest_path.exists():
            raise PipelineExecutionError(
                f"run manifest is invalid: {context.manifest_path}"
            )
        return _new_manifest(context, feature_fingerprint)
    return _manifest_from_payload(
        context,
        feature_fingerprint,
        payload,
        verify_resolved_config=True,
    )


def _manifest_from_payload(
    context: RunContext,
    feature_fingerprint: str,
    payload: dict[str, Any],
    *,
    verify_resolved_config: bool,
) -> RunManifest:
    """Fully validate a persisted run manifest before any resume-side write."""

    if payload.get("fingerprint") != context.fingerprint:
        raise PipelineExecutionError(
            f"run_id {context.run_id!r} already exists with a different fingerprint; "
            "refusing to mix artifacts from different configurations"
        )

    def string(name: str, *, optional: bool = False) -> str | None:
        value = payload.get(name)
        if optional and value is None:
            return None
        if not isinstance(value, str) or not value:
            raise TypeError(f"{name} must be a non-empty string")
        return value

    def string_mapping(name: str) -> dict[str, str]:
        value = payload.get(name)
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value.items()
        ):
            raise TypeError(f"{name} must be a string mapping")
        return dict(value)

    try:
        job_fingerprints = string_mapping("job_fingerprints")
        artifact_sha256 = string_mapping("artifact_sha256")
        effective_model_sha256 = string_mapping("effective_model_sha256")
        provenance = payload.get("provenance")
        stage_status = payload.get("stage_status")
        gpu_assignments = payload.get("gpu_assignments")
        errors = payload.get("errors")
        if not isinstance(provenance, dict):
            raise TypeError("provenance must be a mapping")
        if not isinstance(stage_status, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in stage_status.items()
        ):
            raise TypeError("stage_status must be a string mapping")
        if not isinstance(gpu_assignments, dict):
            raise TypeError("gpu_assignments must be a mapping")
        if not isinstance(errors, list) or not all(
            isinstance(error, str) for error in errors
        ):
            raise TypeError("errors must be a string list")
        manifest = RunManifest(
            run_id=string("run_id") or "",
            fingerprint=string("fingerprint") or "",
            backend=string("backend") or "",
            model=string("model") or "",
            source_csv=string("source_csv") or "",
            target_sequence_sha256=string("target_sequence_sha256") or "",
            job_fingerprints=job_fingerprints,
            secondary_backend=string("secondary_backend") or "",
            secondary_model=string("secondary_model") or "",
            primary_image_id=string("primary_image_id", optional=True),
            secondary_image_id=string("secondary_image_id", optional=True),
            feature_fingerprint=string("feature_fingerprint", optional=True),
            feature_content_sha256=string(
                "feature_content_sha256", optional=True
            ),
            source_csv_sha256=string("source_csv_sha256") or "",
            resolved_config_sha256=string("resolved_config_sha256") or "",
            provenance=provenance,
            artifact_sha256=artifact_sha256,
            effective_model_sha256=effective_model_sha256,
            status=string("status") or "",
            created_at=string("created_at") or "",
            updated_at=string("updated_at") or "",
            stage_status=dict(stage_status),
            gpu_assignments=dict(gpu_assignments),
            errors=list(errors),
            output_schema_version=payload["output_schema_version"],
            version=payload["version"],
        )
        if type(manifest.output_schema_version) is not int or (
            manifest.output_schema_version != OUTPUT_SCHEMA_VERSION
        ):
            raise TypeError("output_schema_version is invalid")
        if type(manifest.version) is not int or manifest.version != MANIFEST_VERSION:
            raise TypeError("manifest version is invalid")
        if manifest.run_id != context.run_id:
            raise TypeError("manifest run_id does not match the run directory")
        if manifest.target_sequence_sha256 != sequence_sha256(
            context.plan.target_sequence
        ):
            raise TypeError("target sequence identity does not match")
        expected_job_fingerprints = {
            job.job_id: job_fingerprint(
                job,
                context.config,
                feature_fingerprint=feature_fingerprint,
            )
            for job in context.plan.jobs
        }
        if manifest.job_fingerprints != expected_job_fingerprints:
            raise TypeError("job_fingerprints do not match the immutable job plan")
        if manifest.feature_fingerprint != feature_fingerprint:
            raise TypeError("feature fingerprint does not match the run plan")
        if (
            manifest.backend != context.config.backend.name
            or manifest.model != context.config.backend.model
            or manifest.secondary_backend != context.config.secondary_backend.name
            or manifest.secondary_model != context.config.secondary_backend.model
        ):
            raise TypeError("backend/model fields do not match the run plan")
        expected_provenance = _context_provenance(context, feature_fingerprint)
        if manifest.provenance != expected_provenance:
            raise TypeError("stored provenance does not match the run fingerprint")
        if manifest.source_csv_sha256 != expected_provenance.get(
            "source_csv_sha256"
        ):
            raise TypeError("source CSV identity does not match")
        if manifest.primary_image_id != context.config.backend.image_id:
            raise TypeError("primary image identity does not match")
        expected_secondary_image = (
            context.config.secondary_backend.image_id
            if context.config.secondary_backend.enabled
            else None
        )
        if manifest.secondary_image_id != expected_secondary_image:
            raise TypeError("secondary image identity does not match")
        bound_feature_content = manifest.artifact_sha256.get("target_features")
        if manifest.feature_content_sha256 != bound_feature_content:
            raise TypeError("prepared feature content binding is inconsistent")
        if (
            manifest.stage_status.get("features") == "success"
            and not manifest.feature_content_sha256
        ):
            raise TypeError("successful feature stage has no content binding")
        if verify_resolved_config and manifest.resolved_config_sha256 != file_sha256(
            context.results_dir / "resolved_config.yaml"
        ):
            raise TypeError("resolved_config.yaml does not match its manifest SHA256")
    except (KeyError, TypeError, ValueError) as exc:
        raise PipelineExecutionError(
            f"run manifest is invalid: {context.manifest_path}"
        ) from exc
    return manifest


def prepare_features_stage(context: RunContext) -> FeaturePreparation:
    log_dir = context.layout.stage("features").logs
    if (
        context.config.backend.name == "alphafold3"
        and context.config.backend.target_data_json
    ):
        preparation = _prepare_af3_target_features(context)
        if preparation.command:
            atomic_write_text(
                log_dir / "prepare_features.command.txt",
                " ".join(preparation.command) + "\n",
            )
        return preparation
    preparation = prepare_target_features(
        context.config.features,
        context.plan.target_sequence,
        dry_run=context.config.runtime.dry_run,
        force=context.config.runtime.force,
        gpu_index=_runtime_gpus(
            context,
            job_count=1,
            stage_name="features",
        )[0].index,
        log_dir=log_dir,
        database_identity=context.feature_database_identity,
    )
    if preparation.command:
        atomic_write_text(
            log_dir / "prepare_features.command.txt",
            " ".join(preparation.command) + "\n",
        )
    return preparation


def run_prepare_features_only(context: RunContext) -> FeaturePreparation:
    feature_fingerprint = _context_feature_fingerprint(context)
    manifest = _existing_or_new_manifest(context, feature_fingerprint)
    manifest.stage_status["features"] = "running"
    manifest.status = "running"
    manifest.write(context.manifest_path)
    try:
        preparation = prepare_features_stage(context)
        if preparation.bundle is not None:
            _bind_feature_content(manifest, preparation.bundle)
        manifest.stage_status["features"] = (
            "dry_run" if context.config.runtime.dry_run else "success"
        )
        manifest.status = (
            "dry_run" if context.config.runtime.dry_run else "success"
        )
        manifest.write(context.manifest_path)
        return preparation
    except Exception as exc:
        manifest.stage_status["features"] = "error"
        manifest.status = "error"
        manifest.errors.append(str(exc))
        manifest.write(context.manifest_path)
        raise


def _absolute_target_features(
    features: TargetFeatures,
    root: Path,
) -> TargetFeatures:
    def absolute(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        return str(path if path.is_absolute() else (root / path).resolve())

    templates: list[dict[str, Any]] = []
    for template in features.templates:
        converted = dict(template)
        converted["mmcifPath"] = absolute(str(template.get("mmcifPath") or ""))
        templates.append(converted)
    return TargetFeatures(
        unpaired_msa_path=absolute(features.unpaired_msa_path),
        paired_msa_path=absolute(features.paired_msa_path),
        templates=templates,
    )


def _af3_bundle_from_manifest(
    cache_dir: Path,
    *,
    target_sequence: str,
    fingerprint: str,
) -> AF3FeatureBundle | None:
    payload = load_manifest(cache_dir / "manifest.json")
    if (
        payload is None
        or payload.get("fingerprint") != fingerprint
        or payload.get("sequence_sha256") != sequence_sha256(target_sequence)
    ):
        return None
    feature_payload = payload.get("features")
    if not isinstance(feature_payload, dict):
        return None
    bundle = AF3FeatureBundle(
        sequence_sha256=sequence_sha256(target_sequence),
        cache_dir=cache_dir,
        target_data_json=cache_dir / "target_data.json",
        features=TargetFeatures(
            unpaired_msa_path=feature_payload.get("unpaired_msa_path"),
            paired_msa_path=feature_payload.get("paired_msa_path"),
            templates=list(feature_payload.get("templates") or []),
        ),
        fingerprint=fingerprint,
    )
    try:
        bundle.validate()
    except Exception:
        return None
    if payload.get("artifact_identity") != _af3_bundle_artifact_identity(bundle):
        return None
    return bundle


def _af3_bundle_artifact_identity(bundle: AF3FeatureBundle) -> dict[str, Any]:
    return dict(af3_feature_bundle_artifact_identity(bundle))


def _bind_feature_content(
    manifest: RunManifest,
    bundle: FeatureBundle | AF3FeatureBundle,
) -> str:
    """Bind the exact prepared MSA/template bytes to the run manifest."""

    digest = feature_bundle_content_sha256(bundle)
    manifest.feature_content_sha256 = digest
    manifest.artifact_sha256["target_features"] = digest
    return digest


def _prediction_feature_identity(
    bundle: FeatureBundle | AF3FeatureBundle | SecondaryFeatureBundle,
) -> str:
    """Bind prediction reuse to generation settings and exact feature bytes."""

    content_sha256 = (
        secondary_feature_bundle_content_sha256(bundle)
        if isinstance(bundle, SecondaryFeatureBundle)
        else feature_bundle_content_sha256(bundle)
    )
    return sequence_sha256(
        json.dumps(
            {
                "generation_fingerprint": bundle.fingerprint,
                "content_sha256": content_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def _primary_prediction_feature_identity(context: RunContext) -> str:
    """Recover the exact primary feature identity for standalone validation."""

    feature_fingerprint = _context_feature_fingerprint(context)
    manifest_path = getattr(context, "manifest_path", None)
    payload = load_manifest(manifest_path) if isinstance(manifest_path, Path) else None
    content_sha256 = (
        payload.get("feature_content_sha256")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(content_sha256, str) or not content_sha256:
        # Compatibility for direct library callers that predate/run outside a
        # validated RunContext. Production standalone CLI manifests require
        # this field whenever the feature stage succeeded.
        return feature_fingerprint
    return sequence_sha256(
        json.dumps(
            {
                "generation_fingerprint": feature_fingerprint,
                "content_sha256": content_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def _prepare_af3_target_features(context: RunContext) -> FeaturePreparation:
    config = context.config
    target_sequence = context.plan.target_sequence
    fingerprint = _context_feature_fingerprint(context)
    if fingerprint != _af3_feature_fingerprint(config, target_sequence):
        raise PipelineExecutionError("AF3 target feature fingerprint changed during setup")
    cache_dir = _expected_feature_cache_dir(
        config,
        target_sequence,
        fingerprint=fingerprint,
    )
    cached = None if config.runtime.force else _af3_bundle_from_manifest(
        cache_dir,
        target_sequence=target_sequence,
        fingerprint=fingerprint,
    )
    if cached is not None:
        return FeaturePreparation(cached, None, reused=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    configured_data = config.backend.target_data_json
    source_data: Path
    command: list[str] | None = None
    build_root: Path | None = None
    if configured_data:
        source_data = Path(configured_data).expanduser().resolve()
        if not source_data.is_file():
            raise PipelineExecutionError(
                f"configured AF3 target data JSON does not exist: {source_data}"
            )
    else:
        if config.runtime.dry_run:
            build_root = cache_dir / ".target-only-dry-run"
            build_root.mkdir(parents=True, exist_ok=True)
        else:
            build_root = Path(
                tempfile.mkdtemp(prefix=".target-only-", dir=cache_dir)
            )
        input_dir = build_root / "input"
        output_dir = build_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_input = write_target_input(
            target_sequence=target_sequence,
            output_dir=input_dir,
            name=config.backend.target_name,
            target_chain=config.project.target_chain,
            seed=config.project.seed,
            force=True,
        )
        gpu_index = _runtime_gpus(
            context,
            job_count=1,
            stage_name="features",
        )[0].index
        command = build_backend_command(
            config,
            input_dir=input_dir,
            output_dir=output_dir,
            gpu_index=gpu_index,
        )
        if config.runtime.dry_run:
            return FeaturePreparation(None, tuple(command), reused=False)
        return_code = _run_prediction_command(
            context,
            command,
            name="target_features",
        )
        if return_code != 0:
            shutil.rmtree(build_root, ignore_errors=True)
            raise PipelineExecutionError(
                f"AF3 target-only feature command returned {return_code}"
            )
        target_name = target_input.stem
        candidates = [
            output_dir / f"{target_name}_data.json",
            output_dir / target_name / f"{target_name}_data.json",
            *sorted(output_dir.rglob(f"{target_name}_data.json")),
        ]
        source_data = next((path for path in candidates if path.is_file()), candidates[0])
        if not source_data.is_file():
            shutil.rmtree(build_root, ignore_errors=True)
            raise PipelineExecutionError(
                f"AF3 target-only run did not produce {target_name}_data.json"
            )

    try:
        extracted = extract_target_features(
            source_data,
            cache_dir,
            chain_id=config.project.target_chain,
            prefix=config.backend.target_name,
            expected_sequence=target_sequence,
            force=True,
        )
        absolute_features = _absolute_target_features(extracted, cache_dir)
        target_payload = json.loads(source_data.read_text(encoding="utf-8"))
        atomic_write_json(cache_dir / "target_data.json", target_payload)
        bundle = AF3FeatureBundle(
            sequence_sha256=sequence_sha256(target_sequence),
            cache_dir=cache_dir,
            target_data_json=cache_dir / "target_data.json",
            features=absolute_features,
            fingerprint=fingerprint,
        )
        bundle.validate()
        atomic_write_json(
            cache_dir / "manifest.json",
            {
                "version": 1,
                "mode": "alphafold3_target_only",
                "fingerprint": fingerprint,
                "sequence_sha256": sequence_sha256(target_sequence),
                "source_target_data_json": str(source_data),
                "backend_image": config.backend.image,
                "backend_image_id": config.backend.image_id,
                "features": {
                    "unpaired_msa_path": absolute_features.unpaired_msa_path,
                    "paired_msa_path": absolute_features.paired_msa_path,
                    "templates": absolute_features.templates,
                },
                "artifact_identity": _af3_bundle_artifact_identity(bundle),
            },
        )
        return FeaturePreparation(
            bundle,
            tuple(command) if command is not None else None,
            reused=False,
        )
    finally:
        if build_root is not None:
            shutil.rmtree(build_root, ignore_errors=True)


def _input_for_job(input_paths: Sequence[Path], job: JobSpec, backend: str) -> Path:
    if backend == "alphafold3":
        return next(path for path in input_paths if path.stem == job.job_id)
    return input_paths[0]


def _legacy_output_valid(
    job: JobSpec,
    input_path: Path,
    prediction: UnifiedPrediction,
) -> bool:
    if prediction.best_model_path is None:
        return False
    return validate_legacy_input(
        input_path,
        job,
        structure_validator=structure_has_chains,
        structure_path=prediction.best_model_path,
    )


def _backend_job_fingerprint(
    context: RunContext,
    job: JobSpec,
    feature_fingerprint: str,
    backend: BackendSettings,
) -> str:
    """Fingerprint one backend job using that backend's actual feature bundle."""

    return sequence_sha256(
        json.dumps(
            {
                "job": job_fingerprint(
                    job,
                    context.config,
                    feature_fingerprint=feature_fingerprint,
                ),
                "backend": backend.name,
                "model": backend.model,
                "runtime_image_id": backend.image_id,
                "runtime_image_reference": (
                    None if backend.image_id else backend.image
                ),
                "checkpoint": checkpoint_identity(backend.checkpoint_path),
            },
            sort_keys=True,
        )
    )


def _reusable_predictions(
    context: RunContext,
    input_paths: Sequence[Path],
    feature_fingerprint: str,
    *,
    jobs: Sequence[JobSpec],
    backend: BackendSettings,
    output_root: Path,
) -> tuple[dict[str, UnifiedPrediction], list[JobSpec]]:
    adapter = output_adapter(backend.name)
    reusable: dict[str, UnifiedPrediction] = {}
    pending: list[JobSpec] = []
    for job in jobs:
        fingerprint = _backend_job_fingerprint(
            context,
            job,
            feature_fingerprint,
            backend,
        )
        job_manifest = load_manifest(output_root / job.job_id / JOB_MANIFEST_NAME)
        parsed = adapter.parse(job, output_root)
        structure_valid = (
            parsed.best_model_path is not None
            and structure_has_chains(
                parsed.best_model_path,
                job.target_chain,
                job.binder_chain,
            )
        )
        matched = job_manifest is not None and job_manifest.get("fingerprint") == fingerprint
        adopted = (
            job_manifest is None
            and context.config.project.adopt_legacy
            and parsed.status == "success"
            and _legacy_output_valid(
                job,
                _input_for_job(input_paths, job, backend.name),
                parsed,
            )
        )
        if (
            not context.config.runtime.force
            and parsed.status == "success"
            and structure_valid
            and (matched or adopted)
        ):
            reusable[job.job_id] = replace(parsed, fingerprint_valid=True)
        else:
            pending.append(job)
    return reusable, pending


def _command_stage_name(name: str) -> str:
    if name.startswith("secondary_prediction"):
        return "secondary_prediction"
    if name.startswith(("esmfold", "esm_if")):
        return "esm"
    if name.startswith("target_features"):
        return "features"
    if name.startswith(("primary_prediction", "prediction")):
        return "primary_prediction"
    return name


def _run_prediction_command(
    context: RunContext,
    command: Sequence[str],
    *,
    name: str = "prediction",
) -> int:
    log_dir = context.layout.stage(_command_stage_name(name)).logs
    atomic_write_text(
        log_dir / f"{name}.command.txt",
        " ".join(command) + "\n",
    )
    if context.config.runtime.dry_run:
        return 0
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    process: subprocess.Popen[str] | None = None
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(list(command), stdout=stdout, stderr=stderr, text=True)
            return process.wait()
    except BaseException:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        raise


def _run_sharded_commands(
    context: RunContext,
    stage_name: str,
    commands: Sequence[tuple[GpuJobShard, Sequence[str]]],
    *,
    progress_probe: Callable[[], int] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> tuple[dict[int, int], list[str]]:
    """Run one Docker command per GPU concurrently and preserve every log."""

    log_dir = context.layout.stage(_command_stage_name(stage_name)).logs
    atomic_write_text(
        log_dir / f"{stage_name}.command.txt",
        "".join(
            "# gpu={} jobs={}\n".format(
                shard.gpu.index,
                ",".join(job.job_id for job in shard.jobs),
            )
            + " ".join(command)
            + "\n"
            for shard, command in commands
        ),
    )
    if context.config.runtime.dry_run:
        return ({shard.gpu.index: 0 for shard, _command in commands}, [])

    return_codes: dict[int, int] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, len(commands))) as pool:
        futures = {
            pool.submit(
                _run_prediction_command,
                context,
                command,
                name=f"{stage_name}.gpu_{shard.gpu.index}",
            ): shard
            for shard, command in commands
        }
        pending_futures = set(futures)
        while pending_futures:
            done, pending_futures = wait(
                pending_futures,
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            if progress_probe is not None and progress_callback is not None:
                try:
                    progress_callback(progress_probe())
                except Exception:
                    # Progress is observational and must never change execution
                    # or failure semantics.
                    pass
            for future in done:
                shard = futures[future]
                try:
                    return_codes[shard.gpu.index] = future.result()
                except Exception as exc:
                    return_codes[shard.gpu.index] = -1
                    errors.append(
                        f"{stage_name} GPU {shard.gpu.index} raised "
                        f"{type(exc).__name__}: {exc}"
                    )
        if progress_probe is not None and progress_callback is not None:
            try:
                progress_callback(progress_probe())
            except Exception:
                pass
    return return_codes, errors


def _file_signature(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in sorted(set(paths)):
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > 0:
            signature.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def _small_json_is_complete(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, json.JSONDecodeError):
        return False


def _path_belongs_to_job(path: Path, job_id: str) -> bool:
    return (
        job_id in path.parts
        or path.stem == job_id
        or path.stem.startswith(f"{job_id}_")
    )


def _prediction_completion_signature(
    backend_name: str,
    job: JobSpec,
    output_root: Path,
) -> tuple[tuple[str, int, int], ...]:
    """Return a lightweight signature only for a plausibly complete job."""

    job_root = output_root / job.job_id
    roots = (job_root, output_root)
    summaries: list[Path] = []
    models: list[Path] = []
    confidences: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        summaries.extend(root.rglob("*summary_confidence*.json"))
        models.extend(root.rglob("*_model.cif" if backend_name == "alphafold3" else "*.cif"))
        if backend_name == "alphafold3":
            confidences.extend(
                path
                for path in root.rglob("*_confidences.json")
                if "summary" not in path.name
            )
        if summaries and models and (backend_name != "alphafold3" or confidences):
            break
    summaries = [
        path
        for path in set(summaries)
        if _path_belongs_to_job(path, job.job_id)
        and _small_json_is_complete(path)
    ]
    models = [
        path
        for path in set(models)
        if _path_belongs_to_job(path, job.job_id)
        and path.is_file()
        and path.stat().st_size > 0
    ]
    if not summaries or not models:
        return ()
    required = [*summaries, *models]
    if backend_name == "alphafold3":
        confidences = [
            path
            for path in set(confidences)
            if _path_belongs_to_job(path, job.job_id)
            and path.is_file()
            and path.stat().st_size > 0
        ]
        ranking = job_root / f"{job.job_id}_ranking_scores.csv"
        if not confidences or not ranking.is_file() or ranking.stat().st_size == 0:
            return ()
        required.extend(confidences)
        required.append(ranking)
    return _file_signature(required)


def _stable_completion_probe(
    keys: Sequence[str],
    signature: Callable[[str], tuple[tuple[str, int, int], ...]],
) -> Callable[[], int]:
    """Count changed completion signatures after two stable observations."""

    baseline = {key: signature(key) for key in keys}
    observed: dict[str, tuple[tuple[str, int, int], ...]] = {}
    completed: set[str] = set()

    def probe() -> int:
        for key in keys:
            if key in completed:
                continue
            current = signature(key)
            if not current or current == baseline[key]:
                observed.pop(key, None)
                continue
            if observed.get(key) == current:
                completed.add(key)
            else:
                observed[key] = current
        return len(completed)

    return probe


def _prediction_artifact_signature(
    prediction: UnifiedPrediction,
) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in (
        prediction.best_model_path,
        prediction.summary_path,
        prediction.confidence_path,
    ):
        if path is None or not path.is_file():
            continue
        stat = path.stat()
        signature.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def prediction_stage(
    context: RunContext,
    target_features: FeatureBundle | AF3FeatureBundle | SecondaryFeatureBundle,
    manifest: RunManifest,
    *,
    jobs: Sequence[JobSpec] | None = None,
    backend_settings: BackendSettings | None = None,
    stage_name: str = "prediction",
    reporter: PipelineProgressReporter | None = None,
) -> tuple[list[UnifiedPrediction], bool]:
    reporter = reporter or NullProgressReporter()
    backend = backend_settings or context.config.backend
    active_jobs = tuple(jobs or context.plan.jobs)
    input_dir = (
        Path(context.config.project.work_dir)
        / context.run_id
        / "inputs"
        / backend.name
    )
    input_paths = write_backend_inputs(
        active_jobs,
        context.config,
        input_dir=input_dir,
        target_features=target_features,
        backend_settings=backend,
        force=context.config.runtime.force,
    )
    output_root = (
        Path(context.config.project.output_dir) / context.run_id / backend.name
    )
    prediction_feature_identity = _prediction_feature_identity(target_features)
    reusable, pending = _reusable_predictions(
        context,
        input_paths,
        prediction_feature_identity,
        jobs=active_jobs,
        backend=backend,
        output_root=output_root,
    )
    cache_hits = len(reusable)
    cache_misses = len(pending)
    reporter.cache_status(
        stage_name,
        hits=cache_hits,
        misses=cache_misses,
        total=len(active_jobs),
        force=context.config.runtime.force,
    )
    task_name = f"{backend.name} predictions"
    reporter.task_started(
        stage_name,
        task_name,
        total=len(active_jobs),
        completed=cache_hits,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    command_failed = False
    previous_signatures: dict[str, tuple[tuple[str, int, int], ...]] = {}
    if pending:
        adapter = output_adapter(backend.name)
        previous_signatures = {
            job.job_id: _prediction_artifact_signature(adapter.parse(job, output_root))
            for job in pending
        }
        pending_key = sequence_sha256(
            "\n".join(sorted(job.job_id for job in pending))
        )[:12]
        execution_root = input_dir / "pending" / pending_key
        shards = plan_gpu_job_shards(
            pending,
            _runtime_gpus(
                context,
                job_count=len(pending),
                stage_name=stage_name,
            ),
        )
        _record_gpu_assignments(
            manifest,
            context.manifest_path,
            stage_name,
            shards,
        )
        commands: list[tuple[GpuJobShard, Sequence[str]]] = []
        for shard in shards:
            execution_input_dir = execution_root / f"gpu_{shard.gpu.index}"
            write_backend_inputs(
                shard.jobs,
                context.config,
                input_dir=execution_input_dir,
                target_features=target_features,
                backend_settings=backend,
                force=True,
            )
            command = build_backend_command(
                context.config,
                input_dir=execution_input_dir,
                output_dir=output_root,
                gpu_index=shard.gpu.index,
                feature_dir=target_features.cache_dir,
                backend_settings=backend,
                template_mmcif_dir=(
                    target_features.template_mmcif_dir
                    if isinstance(target_features, SecondaryFeatureBundle)
                    and target_features.templates_enabled
                    else None
                ),
                container_name=_container_name(
                    context,
                    stage_name,
                    shard.gpu.index,
                ),
            )
            commands.append((shard, command))
        pending_by_id = {job.job_id: job for job in pending}
        completion_probe = _stable_completion_probe(
            tuple(pending_by_id),
            lambda job_id: _prediction_completion_signature(
                backend.name,
                pending_by_id[job_id],
                output_root,
            ),
        )
        return_codes, command_errors = _run_sharded_commands(
            context,
            stage_name,
            commands,
            progress_probe=completion_probe,
            progress_callback=lambda completed: reporter.task_progress(
                stage_name,
                task_name,
                completed=cache_hits + completed,
                total=len(active_jobs),
                success=cache_hits + completed,
                failed=0,
            ),
        )
        manifest.errors.extend(command_errors)
        for gpu_index, return_code in sorted(return_codes.items()):
            if return_code != 0:
                manifest.errors.append(
                    f"{stage_name} GPU {gpu_index} command returned {return_code}"
                )
        command_failed = bool(command_errors) or any(
            return_code != 0 for return_code in return_codes.values()
        )

    adapter = output_adapter(backend.name)
    predictions: list[UnifiedPrediction] = []
    for job in active_jobs:
        prediction = reusable.get(job.job_id) or adapter.parse(job, output_root)
        refreshed = job.job_id in reusable
        if job.job_id not in reusable:
            refreshed = (
                _prediction_artifact_signature(prediction)
                != previous_signatures.get(job.job_id, ())
            )
            if not refreshed:
                prediction = replace(
                    prediction,
                    status="error",
                    error="backend did not create or refresh this job's output",
                    iptm=None,
                    fingerprint_valid=False,
                )
            else:
                prediction = replace(prediction, fingerprint_valid=True)
            if refreshed and prediction.status == "success" and (
                prediction.best_model_path is None
                or not structure_has_chains(
                    prediction.best_model_path,
                    job.target_chain,
                    job.binder_chain,
                )
            ):
                prediction = replace(
                    prediction,
                    status="error",
                    error="best model is missing required target/binder protein chains",
                )
        predictions.append(prediction)
        if prediction.fingerprint_valid and prediction.iptm is not None and refreshed:
            job_dir = output_root / job.job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            write_job_manifest(
                job_dir,
                job=job,
                fingerprint=_backend_job_fingerprint(
                    context,
                    job,
                    prediction_feature_identity,
                    backend,
                ),
                backend=backend.name,
                artifacts={
                    "best_model_path": str(prediction.best_model_path)
                    if prediction.best_model_path
                    else None,
                    "summary_path": str(prediction.summary_path)
                    if prediction.summary_path
                    else None,
                    "confidence_path": str(prediction.confidence_path)
                    if prediction.confidence_path
                    else None,
                },
            )
    stage_failed = command_failed or any(
        prediction.status != "success" for prediction in predictions
    )
    success_count = sum(
        prediction.status == "success" for prediction in predictions
    )
    failure_count = len(predictions) - success_count
    reporter.task_finished(
        stage_name,
        task_name,
        completed=len(active_jobs),
        total=len(active_jobs),
        success=success_count,
        failed=failure_count,
    )
    atomic_write_csv(
        context.layout.stage(stage_name).tables / "predictions.csv",
        _prediction_rows(active_jobs, predictions),
    )
    return predictions, stage_failed


def _prediction_rows(
    jobs: Sequence[JobSpec],
    predictions: Sequence[UnifiedPrediction],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job, prediction in zip(jobs, predictions, strict=True):
        row = {
            "sample_no": job.sample_no,
            "run_name": job.run_name,
            "source_row_number": job.source_row_number,
            "job_name": job.job_id,
            "job_status": prediction.status,
            "job_error": prediction.error or "",
            "backend": prediction.backend,
            "target_chain": job.target_chain,
            "binder_chain": job.binder_chain,
            "binder_sequence": job.binder_sequence,
            "target_sequence": job.target_sequence,
            "best_model_path": str(prediction.best_model_path)
            if prediction.best_model_path
            else None,
            "confidence_path": str(prediction.confidence_path)
            if prediction.confidence_path
            else None,
            "ranking_score": prediction.ranking_score,
            "iptm": prediction.iptm,
            "ptm": prediction.ptm,
            "plddt_global_mean": prediction.plddt,
            "fingerprint_valid": prediction.fingerprint_valid,
            "normalized_plddt_global_mean": (
                prediction.plddt / 100
                if prediction.plddt is not None and prediction.plddt > 1
                else prediction.plddt
            ),
        }
        rows.append(row)
    return rows


def interface_stage(
    context: RunContext,
    predictions: Sequence[UnifiedPrediction],
    base_rows: Sequence[dict[str, Any]],
    *,
    jobs: Sequence[JobSpec] | None = None,
    label: str = "primary",
    write_outputs: bool = True,
    reporter: PipelineProgressReporter | None = None,
) -> list[dict[str, Any]]:
    reporter = reporter or NullProgressReporter()
    active_jobs = tuple(jobs or context.plan.jobs)
    stage_name = "primary_interface" if label == "primary" else "secondary_interface"
    stage_layout = context.layout.stage(stage_name)
    rosetta_input_dir = stage_layout.artifacts / "rosetta_inputs"
    geometry_by_job: dict[str, dict[str, Any]] = {}
    eligibility = (
        f"{len(active_jobs)} eligible / {len(context.plan.jobs)} total"
        if label == "secondary"
        else ""
    )
    geometry_task = "Biotite geometry"
    reporter.task_started(
        stage_name,
        geometry_task,
        total=len(active_jobs),
        detail=eligibility,
    )
    geometry_success = 0
    geometry_failed = 0

    def run_geometry(job: JobSpec, prediction: UnifiedPrediction) -> tuple[str, dict[str, Any]]:
        return job.job_id, analyze_interface_geometry(
            job,
            prediction,
            distance=context.config.interface.distance,
            epitope_residues=context.config.interface.epitope_residues,
            sasa_point_number=context.config.interface.sasa_point_number,
            rosetta_input_dir=rosetta_input_dir,
        )

    worker_count = min(
        context.config.runtime.geometry_max_workers,
        max(1, len(active_jobs)),
    )
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures = {
            pool.submit(run_geometry, job, prediction): job.job_id
            for job, prediction in zip(active_jobs, predictions, strict=True)
        }
        for future in as_completed(futures):
            job_id, result = future.result()
            geometry_by_job[job_id] = result
            if result.get("interface_status") == "success":
                geometry_success += 1
            else:
                geometry_failed += 1
            reporter.task_progress(
                stage_name,
                geometry_task,
                completed=geometry_success + geometry_failed,
                total=len(active_jobs),
                success=geometry_success,
                failed=geometry_failed,
                detail=eligibility,
            )
    reporter.task_finished(
        stage_name,
        geometry_task,
        completed=len(active_jobs),
        total=len(active_jobs),
        success=geometry_success,
        failed=geometry_failed,
        detail=eligibility,
    )

    rosetta_by_job: dict[str, dict[str, Any]] = {}
    rosetta_task = "Rosetta energy"
    if context.config.interface.energy_engine == "rosetta_cli":
        reporter.task_started(
            stage_name,
            rosetta_task,
            total=len(active_jobs),
            detail=eligibility,
        )
        engine = RosettaCliEngine(context.config.interface.rosetta)

        def run_one(job_id: str, geometry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            pdb_value = geometry.get("rosetta_input_pdb")
            if not pdb_value:
                return job_id, {
                    "rosetta_status": "skipped",
                    "rosetta_error": "geometry/PDB conversion failed",
                }
            return job_id, engine.analyze(
                Path(str(pdb_value)),
                output_dir=stage_layout.artifacts / "rosetta_scores",
                log_dir=stage_layout.logs / "rosetta",
            )

        rosetta_success = 0
        rosetta_failed = 0
        rosetta_skipped = 0
        with ThreadPoolExecutor(max_workers=context.config.interface.rosetta.max_workers) as pool:
            futures = {
                pool.submit(run_one, job_id, geometry): job_id
                for job_id, geometry in geometry_by_job.items()
            }
            for future in as_completed(futures):
                job_id, result = future.result()
                rosetta_by_job[job_id] = result
                status = result.get("rosetta_status")
                if status == "success":
                    rosetta_success += 1
                elif status == "skipped":
                    rosetta_skipped += 1
                else:
                    rosetta_failed += 1
                reporter.task_progress(
                    stage_name,
                    rosetta_task,
                    completed=rosetta_success + rosetta_failed + rosetta_skipped,
                    total=len(active_jobs),
                    success=rosetta_success,
                    failed=rosetta_failed,
                    skipped=rosetta_skipped,
                    detail=eligibility,
                )
        reporter.task_finished(
            stage_name,
            rosetta_task,
            completed=len(active_jobs),
            total=len(active_jobs),
            success=rosetta_success,
            failed=rosetta_failed,
            skipped=rosetta_skipped,
            detail=eligibility,
        )
    else:
        reporter.task_started(
            stage_name,
            rosetta_task,
            total=len(active_jobs),
            completed=len(active_jobs),
            detail="disabled",
        )
        reporter.task_finished(
            stage_name,
            rosetta_task,
            completed=len(active_jobs),
            total=len(active_jobs),
            skipped=len(active_jobs),
            detail="disabled",
        )

    merged: list[dict[str, Any]] = []
    for row in base_rows:
        job_id = str(row["job_name"])
        merged.append(
            {
                **row,
                **geometry_by_job.get(job_id, {}),
                **rosetta_by_job.get(
                    job_id,
                    {"rosetta_status": "disabled", "rosetta_error": ""},
                ),
            }
        )
    ranked = apply_balanced_shortlist(
        merged,
        minimum_contact_pairs=context.config.interface.minimum_contact_pairs,
        epitope_configured=bool(context.config.interface.epitope_residues),
        minimum_epitope_coverage=context.config.interface.minimum_epitope_coverage,
        minimum_epitope_purity=context.config.interface.minimum_epitope_purity,
    )
    if write_outputs:
        atomic_write_csv(stage_layout.tables / "interface_metrics.csv", ranked)
    return ranked


def _interface_stage_failed(
    rows: Sequence[dict[str, Any]], *, energy_engine: str
) -> bool:
    """Propagate both geometry and configured energy-substage failures."""

    for row in rows:
        if row.get("interface_status") != "success":
            return True
        if energy_engine == "rosetta_cli" and row.get("rosetta_status") != "success":
            return True
    return False


def clustering_stage(
    context: RunContext,
    predictions: Sequence[UnifiedPrediction],
    rows: Sequence[dict[str, Any]],
    *,
    jobs: Sequence[JobSpec] | None = None,
    manifest: RunManifest | None = None,
) -> ClusteringOutcome:
    active_jobs = tuple(jobs or context.plan.jobs)
    model_paths = {
        prediction.job_id: prediction.best_model_path
        for prediction in predictions
        if prediction.status == "success" and prediction.best_model_path is not None
    }
    clustering_root = (
        Path(context.config.project.work_dir) / context.run_id / "clustering"
    )
    clustering_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(
        tempfile.mkdtemp(prefix="execution-", dir=clustering_root)
    )
    binder_dir, complex_dir = prepare_foldseek_inputs(
        active_jobs,
        model_paths,
        work_dir=work_dir,
    )
    stage_layout = context.layout.stage("clustering")
    foldseek_root = stage_layout.artifacts / "foldseek"
    foldseek_root.mkdir(parents=True, exist_ok=True)
    foldseek_result_dir = Path(
        tempfile.mkdtemp(prefix="execution-", dir=foldseek_root)
    )
    binder_prefix = foldseek_result_dir / "binder"
    complex_prefix = foldseek_result_dir / "complex"
    binder_cluster_tsv = Path(str(binder_prefix) + "_cluster.tsv")
    complex_cluster_tsv = Path(str(complex_prefix) + "_cluster.tsv")
    requested_workers = min(2, context.config.clustering.max_workers)
    gpus = _runtime_gpus(
        context,
        job_count=requested_workers,
        stage_name="clustering",
    )
    binder_gpu = gpus[0]
    complex_gpu = gpus[1] if len(gpus) > 1 else gpus[0]
    if manifest is not None:
        _record_gpu_assignments(
            manifest,
            context.manifest_path,
            "clustering",
            [
                GpuJobShard(binder_gpu, active_jobs),
                GpuJobShard(complex_gpu, active_jobs),
            ],
        )
    binder_command = build_foldseek_container_command(
        context.config.clustering,
        layer="binder",
        docker_bin=context.config.backend.docker_bin,
        image=context.config.backend.image,
        gpu_index=binder_gpu.index,
        input_dir=binder_dir,
        execution_dir=foldseek_result_dir,
        container_name=_container_name(context, "foldseek-binder", binder_gpu.index),
    )
    complex_command = build_foldseek_container_command(
        context.config.clustering,
        layer="complex",
        docker_bin=context.config.backend.docker_bin,
        image=context.config.backend.image,
        gpu_index=complex_gpu.index,
        input_dir=complex_dir,
        execution_dir=foldseek_result_dir,
        container_name=_container_name(context, "foldseek-complex", complex_gpu.index),
    )

    def execute_foldseek(
        layer: str, command: Sequence[str], cluster_tsv: Path
    ):
        return run_foldseek_command(
            layer,
            command,
            cluster_tsv,
            dry_run=context.config.runtime.dry_run,
            log_dir=stage_layout.logs,
        )

    if len(gpus) > 1 and context.config.clustering.max_workers > 1:
        with ThreadPoolExecutor(max_workers=2) as pool:
            binder_future = pool.submit(
                execute_foldseek, "binder", binder_command, binder_cluster_tsv
            )
            complex_future = pool.submit(
                execute_foldseek, "complex", complex_command, complex_cluster_tsv
            )
            binder_run = binder_future.result()
            complex_run = complex_future.result()
    else:
        binder_run = execute_foldseek("binder", binder_command, binder_cluster_tsv)
        complex_run = execute_foldseek("complex", complex_command, complex_cluster_tsv)
    atomic_write_json(
        stage_layout.logs / "clustering_commands.json",
        {
            "binder": {
                "status": binder_run.status,
                "error": binder_run.error,
                "command": list(binder_run.command),
            },
            "complex": {
                "status": complex_run.status,
                "error": complex_run.error,
                "command": list(complex_run.command),
            },
        },
    )
    all_job_ids = [job.job_id for job in active_jobs]
    binder_membership, binder_raw = parse_foldseek_clusters(
        binder_cluster_tsv if binder_run.status == "success" else Path("/nonexistent"),
        all_job_ids=all_job_ids,
        prefix="binder",
    )
    complex_membership, complex_raw = parse_foldseek_clusters(
        complex_cluster_tsv if complex_run.status == "success" else Path("/nonexistent"),
        all_job_ids=all_job_ids,
        prefix="complex",
    )
    contacts = {
        str(row["job_name"]): row.get("effective_target_interface_residues", "")
        for row in rows
    }
    epitope_membership, epitope_raw = greedy_epitope_clusters(
        contacts,
        threshold=context.config.clustering.epitope_jaccard_threshold,
    )
    missing_ids: set[str] = set()
    if not context.config.runtime.dry_run:
        missing_ids = set(all_job_ids) - (
            set(binder_membership) & set(complex_membership)
        )
    cluster_rows: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        if str(row.get("job_name")) in missing_ids:
            reasons = {
                reason
                for reason in str(row.get("manual_review_reason", "")).split(";")
                if reason
            }
            reasons.add("clustering_input_or_output_missing")
            row["manual_review"] = True
            row["manual_review_reason"] = ";".join(sorted(reasons))
            row["clustering_status"] = "error"
        cluster_rows.append(row)
    member_rows, representative_rows, final_rows = write_cluster_outputs(
        results_dir=stage_layout.tables,
        artifacts_dir=stage_layout.artifacts,
        jobs=active_jobs,
        rows=cluster_rows,
        binder_membership=binder_membership,
        binder_raw_representatives=binder_raw,
        complex_membership=complex_membership,
        complex_raw_representatives=complex_raw,
        epitope_membership=epitope_membership,
        epitope_raw_representatives=epitope_raw,
    )
    return ClusteringOutcome(
        failed=(
            binder_run.status == "error"
            or complex_run.status == "error"
            or bool(missing_ids)
        ),
        member_rows=tuple(member_rows),
        representative_rows=tuple(representative_rows),
        final_rows=tuple(final_rows),
    )


def esm_stage(
    context: RunContext,
    predictions: Sequence[UnifiedPrediction],
    manifest: RunManifest,
    *,
    primary_predictions: Sequence[UnifiedPrediction] = (),
    secondary_predictions: Sequence[UnifiedPrediction] = (),
    reporter: PipelineProgressReporter | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Run ESMFold and score each sequence on its effective backbone."""

    reporter = reporter or NullProgressReporter()
    if not context.config.scoring.esm.enabled:
        return ([{"job_name": job.job_id, "esm_status": "disabled"} for job in context.plan.jobs], False)
    input_dir = Path(context.config.project.work_dir) / context.run_id / "esm_inputs"
    stage_layout = context.layout.stage("esm")
    output_dir = stage_layout.artifacts / "esm"
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_rows = None
    if not context.config.runtime.force:
        cached_rows = load_cached_esm_rows(
            stage_layout.tables / "esm_scores.csv",
            context.plan.jobs,
            predictions,
            require_esmfold=context.config.scoring.esm.esmfold,
            require_inverse_folding=context.config.scoring.esm.inverse_folding,
        )
    cache_hits = len(context.plan.jobs) if cached_rows is not None else 0
    reporter.cache_status(
        "esm",
        hits=cache_hits,
        misses=len(context.plan.jobs) - cache_hits,
        total=len(context.plan.jobs),
        force=context.config.runtime.force,
    )
    if cached_rows is not None:
        if context.config.scoring.esm.esmfold:
            reporter.task_started(
                "esm",
                "ESMFold",
                total=len(context.plan.jobs),
                completed=len(context.plan.jobs),
                detail="cache hit",
            )
            reporter.task_finished(
                "esm",
                "ESMFold",
                completed=len(context.plan.jobs),
                total=len(context.plan.jobs),
                success=len(context.plan.jobs),
                failed=0,
                detail="cache hit",
            )
        if context.config.scoring.esm.inverse_folding:
            prediction_by_job = {
                prediction.job_id: prediction for prediction in predictions
            }
            eligible = sum(
                prediction_by_job[job.job_id].status == "success"
                and prediction_by_job[job.job_id].best_model_path is not None
                for job in context.plan.jobs
            )
            detail = f"{eligible} eligible / {len(context.plan.jobs)} total; cache hit"
            reporter.task_started(
                "esm",
                "ESM-IF",
                total=eligible,
                completed=eligible,
                detail=detail,
            )
            reporter.task_finished(
                "esm",
                "ESM-IF",
                completed=eligible,
                total=eligible,
                success=eligible,
                failed=0,
                detail=detail,
            )
        return cached_rows, False
    write_esm_inputs(context.plan.jobs, predictions, input_dir)
    failed = False

    def shard_commands(
        tool_name: str,
        jobs: Sequence[JobSpec],
    ) -> tuple[list[tuple[GpuJobShard, Sequence[str]]], list[Path]]:
        if not jobs:
            manifest.gpu_assignments[tool_name] = []
            manifest.write(context.manifest_path)
            return [], []
        shards = plan_gpu_job_shards(
            jobs,
            _runtime_gpus(
                context,
                job_count=len(jobs),
                stage_name=tool_name,
            ),
        )
        _record_gpu_assignments(
            manifest,
            context.manifest_path,
            tool_name,
            shards,
        )
        commands: list[tuple[GpuJobShard, Sequence[str]]] = []
        shard_outputs: list[Path] = []
        for shard in shards:
            shard_input = (
                input_dir
                / "shards"
                / tool_name
                / f"gpu_{shard.gpu.index}"
            )
            shard_output = (
                output_dir
                / "shards"
                / tool_name
                / f"gpu_{shard.gpu.index}"
            )
            shard_output.mkdir(parents=True, exist_ok=True)
            write_esm_inputs(shard.jobs, predictions, shard_input)
            if tool_name == "esmfold":
                command = build_esmfold_container_command(
                    context.config,
                    input_dir=shard_input,
                    output_dir=shard_output,
                    gpu_index=shard.gpu.index,
                    container_name=_container_name(
                        context,
                        tool_name,
                        shard.gpu.index,
                    ),
                )
            else:
                command = build_esm_if_container_command(
                    context.config,
                    input_dir=shard_input,
                    output_dir=shard_output,
                    prediction_output_dir=(
                        Path(context.config.project.output_dir)
                        / context.run_id
                    ),
                    gpu_index=shard.gpu.index,
                    container_name=_container_name(
                        context,
                        tool_name,
                        shard.gpu.index,
                    ),
                )
            commands.append((shard, command))
            shard_outputs.append(shard_output)
        return commands, shard_outputs

    if context.config.scoring.esm.esmfold:
        commands, shard_outputs = shard_commands(
            "esmfold",
            context.plan.jobs,
        )
        esmfold_task = "ESMFold"
        reporter.task_started(
            "esm",
            esmfold_task,
            total=len(context.plan.jobs),
        )
        esmfold_probe = _stable_completion_probe(
            tuple(job.job_id for job in context.plan.jobs),
            lambda job_id: _file_signature(
                tuple(
                    path
                    for shard_output in shard_outputs
                    for path in (shard_output / "esmfold").glob(
                        f"{job_id}*.pdb"
                    )
                )
            ),
        )
        return_codes, errors = _run_sharded_commands(
            context,
            "esmfold",
            commands,
            progress_probe=esmfold_probe,
            progress_callback=lambda completed: reporter.task_progress(
                "esm",
                esmfold_task,
                completed=completed,
                total=len(context.plan.jobs),
            ),
        )
        manifest.errors.extend(errors)
        manifest.errors.extend(
            f"esmfold GPU {gpu_index} command returned {code}"
            for gpu_index, code in sorted(return_codes.items())
            if code != 0
        )
        failed |= bool(errors) or any(code != 0 for code in return_codes.values())
        canonical_fold = output_dir / "esmfold"
        canonical_fold.mkdir(parents=True, exist_ok=True)
        for shard_output in shard_outputs:
            for model in sorted((shard_output / "esmfold").glob("*.pdb")):
                shutil.copy2(model, canonical_fold / model.name)
        fold_rows = collect_esm_rows(
            context.plan.jobs,
            predictions,
            output_dir,
        )
        fold_success = sum(
            row.get("esmfold_status") == "success" for row in fold_rows
        )
        reporter.task_finished(
            "esm",
            esmfold_task,
            completed=len(context.plan.jobs),
            total=len(context.plan.jobs),
            success=fold_success,
            failed=len(context.plan.jobs) - fold_success,
        )
    if context.config.scoring.esm.inverse_folding:
        prediction_by_job = {prediction.job_id: prediction for prediction in predictions}
        inverse_jobs = [
            job
            for job in context.plan.jobs
            if prediction_by_job[job.job_id].status == "success"
            and prediction_by_job[job.job_id].best_model_path is not None
        ]
        commands, shard_outputs = shard_commands(
            "esm_if",
            inverse_jobs,
        )
        esm_if_task = "ESM-IF"
        inverse_detail = (
            f"{len(inverse_jobs)} eligible / {len(context.plan.jobs)} total"
        )
        reporter.task_started(
            "esm",
            esm_if_task,
            total=len(inverse_jobs),
            detail=inverse_detail,
        )
        esm_if_probe = _stable_completion_probe(
            tuple(job.job_id for job in inverse_jobs),
            lambda job_id: _file_signature(
                tuple(
                    shard_output
                    / ".aerith_progress"
                    / "esm_if"
                    / f"{sequence_sha256(job_id)}.json"
                    for shard_output in shard_outputs
                )
            ),
        )
        return_codes, errors = _run_sharded_commands(
            context,
            "esm_if",
            commands,
            progress_probe=esm_if_probe,
            progress_callback=lambda completed: reporter.task_progress(
                "esm",
                esm_if_task,
                completed=completed,
                total=len(inverse_jobs),
                detail=inverse_detail,
            ),
        )
        manifest.errors.extend(errors)
        manifest.errors.extend(
            f"esm_if GPU {gpu_index} command returned {code}"
            for gpu_index, code in sorted(return_codes.items())
            if code != 0
        )
        failed |= bool(errors) or any(code != 0 for code in return_codes.values())
        inverse_rows: list[dict[str, Any]] = []
        for shard_output in shard_outputs:
            path = shard_output / "esm_if.csv"
            if not path.is_file():
                continue
            with path.open(encoding="utf-8", newline="") as handle:
                inverse_rows.extend(dict(row) for row in csv.DictReader(handle))
        atomic_write_csv(
            output_dir / "esm_if.csv",
            inverse_rows,
            fieldnames=(
                "job_name",
                "esm_if_status",
                "esm_if_error",
                "esm_if_log_likelihood",
                "esm_if_log_likelihood_with_coord",
                "esm_if_perplexity",
            ),
        )
        inverse_by_job = {
            str(row.get("job_name")): row for row in inverse_rows
        }
        inverse_success = sum(
            inverse_by_job.get(job.job_id, {}).get("esm_if_status") == "success"
            for job in inverse_jobs
        )
        reporter.task_finished(
            "esm",
            esm_if_task,
            completed=len(inverse_jobs),
            total=len(inverse_jobs),
            success=inverse_success,
            failed=len(inverse_jobs) - inverse_success,
            detail=inverse_detail,
        )
    rows = collect_esm_rows(
        context.plan.jobs,
        predictions,
        output_dir,
        comparison_label="effective",
    )
    if primary_predictions:
        rows = add_esmfold_backend_comparison(
            rows,
            context.plan.jobs,
            primary_predictions,
            output_dir,
            label="primary",
        )
    if secondary_predictions:
        rows = add_esmfold_backend_comparison(
            rows,
            context.plan.jobs,
            secondary_predictions,
            output_dir,
            label="secondary",
        )
    if context.config.scoring.esm.esmfold:
        failed |= any(row.get("esmfold_status") != "success" for row in rows)
    expected_if = {
        prediction.job_id for prediction in predictions if prediction.status == "success"
    }
    if context.config.scoring.esm.inverse_folding:
        failed |= any(
            row.get("esm_if_status") != "success"
            for row in rows
            if row["job_name"] in expected_if
        )
    atomic_write_csv(stage_layout.tables / "esm_scores.csv", rows)
    return rows, failed


def _merge_rows_by_job(
    rows: Sequence[dict[str, Any]], additions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_job = {str(row["job_name"]): row for row in additions}
    return [{**row, **by_job.get(str(row["job_name"]), {})} for row in rows]


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _effective_predictions_from_rows(
    jobs: Sequence[JobSpec], rows: Sequence[dict[str, Any]]
) -> list[UnifiedPrediction]:
    """Materialize the selected backend as the one-structure adapter contract."""

    by_job = {str(row["job_name"]): row for row in rows}
    predictions: list[UnifiedPrediction] = []
    for job in jobs:
        row = by_job[job.job_id]
        status = row.get("effective_status")
        path_value = row.get("effective_best_model_path")
        predictions.append(
            UnifiedPrediction(
                job_id=job.job_id,
                backend=str(row.get("effective_backend") or "none"),
                status="success" if status == "success" and path_value else "missing",
                best_model_path=Path(str(path_value)) if path_value else None,
                ranking_score=_optional_float(row.get("effective_ranking_score")),
                iptm=_optional_float(row.get("effective_iptm")),
                ptm=_optional_float(row.get("effective_ptm")),
                plddt=_optional_float(row.get("effective_plddt_global_mean")),
                error=None if path_value else "no eligible effective backend",
                fingerprint_valid=bool(path_value),
            )
        )
    return predictions


def _final_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def number(name: str, default: float) -> float:
        try:
            value = float(row.get(name))
            return value if value == value else default
        except (TypeError, ValueError):
            return default

    return (
        0 if row.get("candidate_pool") else 1,
        -number("effective_epitope_coverage", -1),
        -number("consensus_epitope_jaccard", -1),
        -number("consensus_interface_pair_jaccard", -1),
        -number("consensus_interface_lddt", -1),
        number("consensus_interface_fixed_frame_rmsd", math.inf),
        number("effective_interface_pae_mean", math.inf),
        number("effective_rosetta_dG_separated_per_dSASA_x100", math.inf),
        -number("effective_rosetta_packstat", -math.inf),
        -number("effective_iptm", -math.inf),
        -number("esm_if_log_likelihood", -math.inf),
        -number("esmfold_plddt", -math.inf),
        -number("effective_ranking_score", -math.inf),
        str(row.get("job_name", "")),
    )


def secondary_gate_job_ids(
    predictions: Sequence[UnifiedPrediction], threshold: float
) -> set[str]:
    """Gate on fingerprint-valid AF3 metrics, not AF3 structure/geometry success."""

    return {
        prediction.job_id
        for prediction in predictions
        if prediction.fingerprint_valid
        and prediction.iptm is not None
        and prediction.iptm >= threshold
    }


def run_pipeline(
    context: RunContext,
    *,
    reporter: PipelineProgressReporter | None = None,
) -> list[dict[str, Any]]:
    reporter = reporter or NullProgressReporter()
    stage_specs = _pipeline_stage_specs(context.config)
    reporter.pipeline_started(
        PipelineRunInfo(
            run_id=context.run_id,
            job_count=len(context.plan.jobs),
            primary_backend=context.config.backend.name,
            secondary_backend=(
                context.config.secondary_backend.name
                if context.config.secondary_backend.enabled
                else "none"
            ),
            gpu_ids=tuple(context.config.runtime.gpu_ids),
            results_dir=context.results_dir,
            output_dir=Path(context.config.project.output_dir),
            logs_dir=context.results_dir / "stages",
        ),
        stage_specs,
    )
    active_stage: str | None = None
    pipeline_reported = False

    def start_stage(stage: str) -> None:
        nonlocal active_stage
        active_stage = stage
        reporter.stage_started(
            stage,
            log_dir=context.layout.stage(stage).logs,
        )

    def finish_stage(stage: str, status: str, detail: str = "") -> None:
        nonlocal active_stage
        reporter.stage_finished(stage, status=status, detail=detail)
        if active_stage == stage:
            active_stage = None

    def finish_pipeline(status: str, detail: str = "") -> None:
        nonlocal pipeline_reported
        if pipeline_reported:
            return
        reporter.pipeline_finished(status=status, detail=detail)
        pipeline_reported = True

    expected_feature_fingerprint = _context_feature_fingerprint(context)
    manifest = _existing_or_new_manifest(context, expected_feature_fingerprint)
    manifest.write(context.manifest_path)
    reporter.message("Preflight: validating configuration, images, paths, and GPU policy")
    if not context.config.runtime.dry_run:
        validation = validate_hydra_config(context.config)
        preflight_errors = list(validation.errors)
        if context.config.backend.image_id is None:
            preflight_errors.append(
                f"backend Docker image is not available: {context.config.backend.image}"
            )
        if (
            context.config.secondary_backend.enabled
            and context.config.secondary_backend.image_id is None
        ):
            preflight_errors.append(
                "secondary backend Docker image is not available: "
                f"{context.config.secondary_backend.image}"
            )
        if preflight_errors:
            message = "pipeline preflight failed: " + "; ".join(preflight_errors)
            manifest.stage_status["preflight"] = "error"
            manifest.status = "error"
            manifest.errors.append(message)
            manifest.write(context.manifest_path)
            finish_pipeline("error", message)
            raise PipelineExecutionError(message)
        manifest.stage_status["preflight"] = "success"
        reporter.message("Preflight: SUCCESS", level="success")
    else:
        manifest.stage_status["preflight"] = "dry_run"
        reporter.message("Preflight: DRY-RUN")
    manifest.write(context.manifest_path)
    required_failure = False
    try:
        start_stage("features")
        manifest.stage_status["features"] = "running"
        manifest.write(context.manifest_path)
        feature_cache_hit = _target_feature_cache_hit(context)
        reporter.cache_status(
            "features",
            hits=int(feature_cache_hit),
            misses=int(not feature_cache_hit),
            total=1,
            force=context.config.runtime.force,
        )
        reporter.task_started(
            "features",
            "Target MSA/templates",
            total=1,
            completed=int(feature_cache_hit),
        )
        preparation = prepare_features_stage(context)
        if context.config.runtime.dry_run:
            reporter.task_finished(
                "features",
                "Target MSA/templates",
                completed=1,
                total=1,
                skipped=1,
                detail="planned only",
            )
            finish_stage("features", "dry_run")
            message = (
                "# deferred until GPU MMseqs2 preprocessing produces "
                "validated features\n"
            )
            if preparation.bundle is not None:
                input_root = (
                    Path(context.config.project.work_dir)
                    / context.run_id
                    / "inputs"
                    / context.config.backend.name
                )
                # Preserve the canonical dry-run inputs for compatibility and
                # auditability; shard directories below are execution views.
                write_backend_inputs(
                    context.plan.jobs,
                    context.config,
                    input_dir=input_root,
                    target_features=preparation.bundle,
                    backend_settings=context.config.backend,
                    force=True,
                )
                shards = plan_gpu_job_shards(
                    context.plan.jobs,
                    _runtime_gpus(
                        context,
                        job_count=len(context.plan.jobs),
                        stage_name="primary_prediction",
                    ),
                )
                _record_gpu_assignments(
                    manifest,
                    context.manifest_path,
                    "primary_prediction",
                    shards,
                )
                commands: list[tuple[GpuJobShard, Sequence[str]]] = []
                for shard in shards:
                    input_dir = input_root / "dry_run" / f"gpu_{shard.gpu.index}"
                    write_backend_inputs(
                        shard.jobs,
                        context.config,
                        input_dir=input_dir,
                        target_features=preparation.bundle,
                        backend_settings=context.config.backend,
                        force=True,
                    )
                    commands.append(
                        (
                            shard,
                            build_backend_command(
                                context.config,
                                input_dir=input_dir,
                                output_dir=(
                                    Path(context.config.project.output_dir)
                                    / context.run_id
                                    / context.config.backend.name
                                ),
                                gpu_index=shard.gpu.index,
                                feature_dir=preparation.bundle.cache_dir,
                                backend_settings=context.config.backend,
                                container_name=_container_name(
                                    context,
                                    "primary_prediction",
                                    shard.gpu.index,
                                ),
                            ),
                        )
                    )
                _run_sharded_commands(
                    context,
                    "primary_prediction",
                    commands,
                )
                primary_logs = context.layout.stage("primary_prediction").logs
                shutil.copy2(
                    primary_logs / "primary_prediction.command.txt",
                    primary_logs / "prediction.command.txt",
                )
            else:
                atomic_write_text(
                    context.layout.stage("primary_prediction").logs
                    / "prediction.command.txt",
                    message,
                )
            if context.config.scoring.esm.enabled:
                esm_input = Path(context.config.project.work_dir) / context.run_id / "esm_inputs"
                esm_layout = context.layout.stage("esm")
                esm_output = esm_layout.artifacts / "esm"
                esm_output.mkdir(parents=True, exist_ok=True)
                missing_predictions = [
                    UnifiedPrediction(job.job_id, "alphafold3", "missing")
                    for job in context.plan.jobs
                ]
                write_esm_inputs(context.plan.jobs, missing_predictions, esm_input)
                if context.config.scoring.esm.esmfold:
                    esm_shards = plan_gpu_job_shards(
                        context.plan.jobs,
                        _runtime_gpus(
                            context,
                            job_count=len(context.plan.jobs),
                            stage_name="esmfold",
                        ),
                    )
                    _record_gpu_assignments(
                        manifest,
                        context.manifest_path,
                        "esmfold",
                        esm_shards,
                    )
                    esm_commands: list[tuple[GpuJobShard, Sequence[str]]] = []
                    for shard in esm_shards:
                        shard_input = esm_input / "dry_run" / f"gpu_{shard.gpu.index}"
                        shard_output = esm_output / "dry_run" / f"gpu_{shard.gpu.index}"
                        write_esm_inputs(shard.jobs, missing_predictions, shard_input)
                        esm_commands.append((
                            shard,
                            build_esmfold_container_command(
                                context.config,
                                input_dir=shard_input,
                                output_dir=shard_output,
                                gpu_index=shard.gpu.index,
                                container_name=_container_name(
                                    context,
                                    "esmfold",
                                    shard.gpu.index,
                                ),
                            ),
                        ))
                    _run_sharded_commands(
                        context,
                        "esmfold",
                        esm_commands,
                    )
                else:
                    atomic_write_text(
                        esm_layout.logs / "esmfold.command.txt",
                        "# disabled\n",
                    )
                atomic_write_text(
                    esm_layout.logs / "esm_if.command.txt",
                    message,
                )
            if context.config.secondary_backend.enabled:
                atomic_write_text(
                    context.layout.stage("secondary_prediction").logs
                    / "secondary_prediction.command.txt",
                    message,
                )
            for stage_spec in stage_specs:
                stage = stage_spec.key
                manifest.stage_status[stage] = "dry_run"
                if stage == "features":
                    continue
                start_stage(stage)
                reporter.task_started(
                    stage,
                    "Planned jobs",
                    total=len(context.plan.jobs),
                    completed=len(context.plan.jobs),
                    detail="dry-run",
                )
                reporter.task_finished(
                    stage,
                    "Planned jobs",
                    completed=len(context.plan.jobs),
                    total=len(context.plan.jobs),
                    skipped=len(context.plan.jobs),
                    detail="dry-run",
                )
                finish_stage(stage, "dry_run")
            manifest.status = "dry_run"
            manifest.write(context.manifest_path)
            write_public_reports(context.layout, (), clustering_status="dry_run")
            finish_pipeline(
                "dry_run",
                f"{len(context.plan.jobs)} jobs planned; results in {context.results_dir}",
            )
            return []
        if preparation.bundle is None:
            raise PipelineExecutionError("feature preparation produced no bundle")
        if preparation.bundle.fingerprint != expected_feature_fingerprint:
            raise PipelineExecutionError(
                "prepared target feature fingerprint differs from the run plan"
            )
        _bind_feature_content(manifest, preparation.bundle)
        manifest.stage_status["features"] = "success"
        manifest.write(context.manifest_path)
        reporter.task_finished(
            "features",
            "Target MSA/templates",
            completed=1,
            total=1,
            success=1,
            detail=("cache hit" if preparation.reused else "cache missing resolved"),
        )
        finish_stage(
            "features",
            "success",
            "cache hit" if preparation.reused else "cache missing resolved",
        )

        if not isinstance(preparation.bundle, (AF3FeatureBundle, FeatureBundle)):
            raise PipelineExecutionError(
                "primary AF3 preparation did not return compatible local features"
            )

        start_stage("primary_prediction")
        manifest.stage_status["primary_prediction"] = "running"
        manifest.write(context.manifest_path)
        primary_predictions, primary_prediction_failed = prediction_stage(
            context,
            preparation.bundle,
            manifest,
            stage_name="primary_prediction",
            reporter=reporter,
        )
        manifest.stage_status["primary_prediction"] = (
            "partial" if primary_prediction_failed else "success"
        )
        finish_stage(
            "primary_prediction",
            manifest.stage_status["primary_prediction"],
        )
        required_failure |= primary_prediction_failed

        primary_rows = _prediction_rows(context.plan.jobs, primary_predictions)
        start_stage("primary_interface")
        manifest.stage_status["primary_interface"] = "running"
        manifest.write(context.manifest_path)
        primary_rows = interface_stage(
            context,
            primary_predictions,
            primary_rows,
            label="primary",
            write_outputs=True,
            reporter=reporter,
        )
        primary_interface_failed = _interface_stage_failed(
            primary_rows,
            energy_engine=context.config.interface.energy_engine,
        )
        manifest.stage_status["primary_interface"] = (
            "partial" if primary_interface_failed else "success"
        )
        finish_stage(
            "primary_interface",
            manifest.stage_status["primary_interface"],
        )
        required_failure |= primary_interface_failed

        secondary_rows: list[dict[str, Any]] = []
        secondary_predictions: list[UnifiedPrediction] = []
        eligible_jobs: tuple[JobSpec, ...] = ()
        if context.config.secondary_backend.enabled:
            threshold = context.config.secondary_backend.minimum_primary_iptm
            eligible_ids = secondary_gate_job_ids(primary_predictions, threshold)
            eligible_jobs = tuple(
                replace(
                    job,
                    backend=context.config.secondary_backend.name,
                    model=context.config.secondary_backend.model,
                )
                for job in context.plan.jobs
                if job.job_id in eligible_ids
            )
            eligible_detail = (
                f"{len(eligible_jobs)} eligible / "
                f"{len(context.plan.jobs)} total"
            )
            start_stage("secondary_features")
            manifest.stage_status["secondary_features"] = "running"
            reporter.task_started(
                "secondary_features",
                "Adapt AF3 target features",
                total=1,
                detail=eligible_detail,
            )
            if isinstance(preparation.bundle, FeatureBundle):
                secondary_features = adapt_local_features_for_secondary(
                    preparation.bundle,
                    context.plan.target_sequence,
                    force=context.config.runtime.force,
                )
            else:
                secondary_features = adapt_af3_features_for_secondary(
                    preparation.bundle,
                    context.plan.target_sequence,
                    force=context.config.runtime.force,
                )
            manifest.stage_status["secondary_features"] = "success"
            reporter.task_finished(
                "secondary_features",
                "Adapt AF3 target features",
                completed=1,
                total=1,
                success=1,
                detail=eligible_detail,
            )
            finish_stage(
                "secondary_features",
                "success",
                eligible_detail,
            )

            start_stage("secondary_prediction")
            manifest.stage_status["secondary_prediction"] = "running"
            if eligible_jobs:
                secondary_predictions, secondary_prediction_failed = prediction_stage(
                    context,
                    secondary_features,
                    manifest,
                    jobs=eligible_jobs,
                    backend_settings=context.config.secondary_backend,
                    stage_name="secondary_prediction",
                    reporter=reporter,
                )
                secondary_rows = _prediction_rows(eligible_jobs, secondary_predictions)
                secondary_prediction_status = (
                    "partial" if secondary_prediction_failed else "success"
                )
            else:
                secondary_prediction_failed = False
                secondary_prediction_status = "skipped"
                reporter.cache_status(
                    "secondary_prediction",
                    hits=0,
                    misses=0,
                    total=0,
                )
                reporter.task_started(
                    "secondary_prediction",
                    f"{context.config.secondary_backend.name} predictions",
                    total=0,
                    detail=eligible_detail,
                )
                reporter.task_finished(
                    "secondary_prediction",
                    f"{context.config.secondary_backend.name} predictions",
                    completed=0,
                    total=0,
                    skipped=0,
                    detail="no eligible jobs",
                )
            manifest.stage_status["secondary_prediction"] = secondary_prediction_status
            finish_stage(
                "secondary_prediction",
                secondary_prediction_status,
                eligible_detail,
            )
            required_failure |= secondary_prediction_failed

            start_stage("secondary_interface")
            manifest.stage_status["secondary_interface"] = "running"
            if eligible_jobs:
                secondary_rows = interface_stage(
                    context,
                    secondary_predictions,
                    secondary_rows,
                    jobs=eligible_jobs,
                    label="secondary",
                    write_outputs=True,
                    reporter=reporter,
                )
            else:
                reporter.task_started(
                    "secondary_interface",
                    "Interface analysis",
                    total=0,
                    detail=eligible_detail,
                )
                reporter.task_finished(
                    "secondary_interface",
                    "Interface analysis",
                    completed=0,
                    total=0,
                    skipped=0,
                    detail="no eligible jobs",
                )
            secondary_interface_failed = _interface_stage_failed(
                secondary_rows,
                energy_engine=context.config.interface.energy_engine,
            )
            secondary_interface_status = (
                "skipped"
                if not eligible_jobs
                else ("partial" if secondary_interface_failed else "success")
            )
            manifest.stage_status["secondary_interface"] = secondary_interface_status
            finish_stage(
                "secondary_interface",
                secondary_interface_status,
                eligible_detail,
            )
            required_failure |= secondary_interface_failed
        else:
            manifest.stage_status.update(
                {
                    "secondary_features": "disabled",
                    "secondary_prediction": "disabled",
                    "secondary_interface": "disabled",
                }
            )

        start_stage("consensus")
        manifest.stage_status["consensus"] = "running"
        manifest.write(context.manifest_path)
        reporter.task_started(
            "consensus",
            "Merge backend results",
            total=len(context.plan.jobs),
        )
        if context.config.secondary_backend.enabled:
            final_rows = consensus_rows(
                primary_rows, secondary_rows, context.config.consensus
            )
            consensus_failed = any(
                row.get("secondary_status") == "success"
                and row.get("consensus_status") != "success"
                for row in final_rows
            )
        else:
            final_rows = []
            for primary in primary_rows:
                row = dict(primary)
                row.update({f"primary_{key}": value for key, value in primary.items()})
                row.update(
                    {
                        "secondary_backend": "none",
                        "secondary_status": "disabled",
                        "consensus_status": "not_applicable",
                        "manual_review": False,
                        "manual_review_reason": "",
                    }
                )
                final_rows.append(row)
            consensus_failed = False
        manifest.stage_status["consensus"] = (
            "partial" if consensus_failed else "success"
        )
        consensus_error_count = sum(
            row.get("consensus_status") == "error" for row in final_rows
        )
        reporter.task_finished(
            "consensus",
            "Merge backend results",
            completed=len(context.plan.jobs),
            total=len(context.plan.jobs),
            success=len(context.plan.jobs) - consensus_error_count,
            failed=consensus_error_count,
        )
        required_failure |= consensus_failed

        secondary_by_job = {str(row["job_name"]): row for row in secondary_rows}
        eligible_ids = {job.job_id for job in eligible_jobs}
        for row in final_rows:
            job_name = str(row["job_name"])
            primary_pass = bool(row.get("final_pass"))
            if context.config.secondary_backend.enabled:
                secondary = secondary_by_job.get(job_name, {})
                secondary_success = secondary.get("job_status") == "success"
                secondary_pass = bool(secondary.get("final_pass"))
                row["secondary_gate_pass"] = job_name in eligible_ids
                row["cross_validation_pass"] = (
                    job_name in eligible_ids
                    and secondary_success
                    and (primary_pass or secondary_pass)
                )
                row["candidate_pool"] = row["cross_validation_pass"]
                review_reasons = {
                    value
                    for value in str(row.get("manual_review_reason", "")).split(";")
                    if value
                }
                if secondary_success and secondary_pass and not primary_pass:
                    review_reasons.add("secondary_rescue")
                if secondary_success and row.get("consensus_status") == "error":
                    review_reasons.add("consensus_or_target_alignment_failure")
                try:
                    same_fold = (
                        float(row.get("consensus_binder_fold_tm"))
                        >= context.config.consensus.same_fold_tm_threshold
                    )
                    different_pose = (
                        float(row.get("consensus_binder_fixed_frame_rmsd"))
                        >= context.config.consensus.different_pose_rmsd_threshold
                    )
                    if same_fold and different_pose:
                        review_reasons.add("same_fold_different_pose")
                except (TypeError, ValueError):
                    pass
                if review_reasons:
                    row["manual_review"] = True
                    row["manual_review_reason"] = ";".join(sorted(review_reasons))
            else:
                row["secondary_gate_pass"] = False
                row["cross_validation_pass"] = None
                row["candidate_pool"] = primary_pass

        final_rows = [apply_effective_backend(row) for row in final_rows]
        effective_predictions = _effective_predictions_from_rows(
            context.plan.jobs,
            final_rows,
        )
        candidates = [row for row in final_rows if row.get("candidate_pool")]
        manual_review = [row for row in final_rows if row.get("manual_review")]
        consensus_layout = context.layout.stage("consensus")
        consensus_results_path = consensus_layout.tables / "consensus_results.csv"
        candidates_full_path = consensus_layout.tables / "candidates_full.csv"
        atomic_write_csv(consensus_results_path, final_rows)
        atomic_write_csv(candidates_full_path, candidates)
        atomic_write_csv(
            consensus_layout.tables / "secondary_backend_rows.csv", secondary_rows
        )
        atomic_write_csv(consensus_layout.tables / "manual_review.csv", manual_review)
        manifest.artifact_sha256["consensus_results"] = (
            file_sha256(consensus_results_path) or ""
        )
        manifest.artifact_sha256["consensus_candidates"] = (
            file_sha256(candidates_full_path) or ""
        )
        manifest.write(context.manifest_path)
        finish_stage("consensus", manifest.stage_status["consensus"])

        if context.config.scoring.esm.enabled:
            start_stage("esm")
            manifest.stage_status["esm"] = "running"
            manifest.write(context.manifest_path)
            esm_rows, esm_failed = esm_stage(
                context,
                effective_predictions,
                manifest,
                primary_predictions=primary_predictions,
                secondary_predictions=secondary_predictions,
                reporter=reporter,
            )
            manifest.stage_status["esm"] = (
                "partial" if esm_failed else "success"
            )
            finish_stage("esm", manifest.stage_status["esm"])
        else:
            esm_rows, esm_failed = esm_stage(
                context,
                effective_predictions,
                manifest,
            )
            manifest.stage_status["esm"] = "disabled"
        final_rows = _merge_rows_by_job(final_rows, esm_rows)
        final_rows.sort(key=_final_sort_key)
        candidates = [row for row in final_rows if row.get("candidate_pool")]
        required_failure |= esm_failed

        _persist_clustering_inputs(context, final_rows, manifest)
        _clustering_all_rows, clustering_candidates = _validated_clustering_inputs(
            context, manifest
        )

        start_stage("clustering")
        manifest.stage_status["clustering"] = "running"
        manifest.write(context.manifest_path)
        candidate_ids = {str(row["job_name"]) for row in clustering_candidates}
        cluster_jobs = tuple(
            job for job in context.plan.jobs if job.job_id in candidate_ids
        )
        cluster_predictions = tuple(
            _effective_predictions_from_rows(cluster_jobs, clustering_candidates)
        )
        cluster_detail = (
            f"{len(cluster_jobs)} candidates / "
            f"{len(context.plan.jobs)} total"
        )
        reporter.task_started(
            "clustering",
            "Foldseek clustering",
            total=None,
            detail=cluster_detail,
        )
        if cluster_jobs:
            cluster_outcome = clustering_stage(
                context,
                cluster_predictions,
                clustering_candidates,
                jobs=cluster_jobs,
                manifest=manifest,
            )
        else:
            cluster_layout = context.layout.stage("clustering")
            member_rows, representative_rows, shortlist_rows = write_cluster_outputs(
                results_dir=cluster_layout.tables,
                artifacts_dir=cluster_layout.artifacts,
                jobs=(),
                rows=(),
                binder_membership={},
                binder_raw_representatives={},
                complex_membership={},
                complex_raw_representatives={},
                epitope_membership={},
                epitope_raw_representatives={},
            )
            cluster_outcome = ClusteringOutcome(
                failed=False,
                member_rows=tuple(member_rows),
                representative_rows=tuple(representative_rows),
                final_rows=tuple(shortlist_rows),
            )
        cluster_failed = cluster_outcome.failed
        manifest.stage_status["clustering"] = "partial" if cluster_failed else "success"
        reporter.task_finished(
            "clustering",
            "Foldseek clustering",
            completed=len(cluster_jobs),
            total=len(cluster_jobs),
            success=(0 if cluster_failed else len(cluster_jobs)),
            failed=(len(cluster_jobs) if cluster_failed else 0),
            detail=(
                f"representatives={len(cluster_outcome.representative_rows)} "
                f"shortlist={len(cluster_outcome.final_rows)}"
            ),
        )
        finish_stage(
            "clustering",
            manifest.stage_status["clustering"],
            (
                f"{cluster_detail}; "
                f"representatives={len(cluster_outcome.representative_rows)}"
            ),
        )
        required_failure |= cluster_failed

        write_public_reports(
            context.layout,
            final_rows,
            member_rows=cluster_outcome.member_rows,
            representative_rows=cluster_outcome.representative_rows,
            final_job_ids=tuple(
                str(row.get("job_name")) for row in cluster_outcome.final_rows
            ),
            clustering_status=manifest.stage_status["clustering"],
        )
        manifest.artifact_sha256["public_all_results"] = (
            file_sha256(context.layout.all_results) or ""
        )
        manifest.artifact_sha256["public_candidates"] = (
            file_sha256(context.layout.candidates) or ""
        )
        manifest.artifact_sha256["public_final_shortlist"] = (
            file_sha256(context.layout.final_shortlist) or ""
        )
        manifest.artifact_sha256["backend_review"] = (
            file_sha256(context.layout.backend_review) or ""
        )

        manifest.status = "partial" if required_failure else "success"
        manifest.write(context.manifest_path)
        if required_failure and not context.config.project.allow_partial:
            raise PipelineExecutionError(
                "one or more required stages failed; partial results were preserved in "
                f"{context.results_dir}"
            )
        interface_success = sum(
            row.get("interface_status") == "success" for row in final_rows
        )
        finish_pipeline(
            manifest.status,
            (
                f"interface_success={interface_success}/{len(final_rows)} "
                f"candidates={len(candidates)} "
                f"shortlist={len(cluster_outcome.final_rows)}; "
                f"results in {context.results_dir}"
            ),
        )
        return final_rows
    except Exception as exc:
        if active_stage is not None:
            finish_stage(active_stage, "error", str(exc))
        for stage, status in tuple(manifest.stage_status.items()):
            if status == "running":
                manifest.stage_status[stage] = "error"
        if str(exc) not in manifest.errors:
            manifest.errors.append(str(exc))
        if manifest.status != "partial":
            manifest.status = "error"
        manifest.write(context.manifest_path)
        finish_pipeline(manifest.status, str(exc))
        raise


def load_predictions_for_context(
    context: RunContext,
) -> tuple[list[UnifiedPrediction], list[dict[str, Any]]]:
    adapter = output_adapter(context.config.backend.name)
    output_root = (
        Path(context.config.project.output_dir)
        / context.run_id
        / context.config.backend.name
    )
    feature_fingerprint = _primary_prediction_feature_identity(context)
    input_dir = (
        Path(context.config.project.work_dir)
        / context.run_id
        / "inputs"
        / context.config.backend.name
    )
    predictions: list[UnifiedPrediction] = []
    for job in context.plan.jobs:
        parsed = adapter.parse(job, output_root)
        if parsed.status != "success":
            predictions.append(parsed)
            continue
        # Keep standalone stages on the exact same artifact identity contract
        # used by prediction_stage().  A hand-copied subset previously omitted
        # the checkpoint field and rejected every valid job manifest.
        fingerprint = _backend_job_fingerprint(
            context,
            job,
            feature_fingerprint,
            context.config.backend,
        )
        job_manifest = load_manifest(output_root / job.job_id / JOB_MANIFEST_NAME)
        matched = (
            job_manifest is not None
            and job_manifest.get("fingerprint") == fingerprint
        )
        adopted = False
        if job_manifest is None and context.config.project.adopt_legacy:
            input_path = (
                input_dir / f"{job.job_id}.json"
                if context.config.backend.name == "alphafold3"
                else input_dir / f"{context.config.backend.name}_jobs.json"
            )
            if input_path.is_file():
                adopted = _legacy_output_valid(job, input_path, parsed)
        structure_valid = (
            parsed.best_model_path is not None
            and structure_has_chains(
                parsed.best_model_path,
                job.target_chain,
                job.binder_chain,
            )
        )
        if not structure_valid:
            parsed = replace(
                parsed,
                status="error",
                error="best model is missing required target/binder protein chains",
            )
        elif not (matched or adopted):
            parsed = replace(
                parsed,
                status="error",
                error="output manifest is missing or has a different run fingerprint",
            )
        elif adopted:
            job_dir = output_root / job.job_id
            job_dir.mkdir(parents=True, exist_ok=True)
            write_job_manifest(
                job_dir,
                job=job,
                fingerprint=fingerprint,
                backend=context.config.backend.name,
                artifacts={
                    "best_model_path": str(parsed.best_model_path)
                    if parsed.best_model_path
                    else None,
                    "summary_path": str(parsed.summary_path)
                    if parsed.summary_path
                    else None,
                    "confidence_path": str(parsed.confidence_path)
                    if parsed.confidence_path
                    else None,
                },
            )
        predictions.append(parsed)
    return predictions, _prediction_rows(context.plan.jobs, predictions)


def run_interface_only(context: RunContext) -> list[dict[str, Any]]:
    feature_fingerprint = _context_feature_fingerprint(context)
    manifest = _existing_or_new_manifest(context, feature_fingerprint)
    manifest.stage_status["primary_interface"] = "running"
    manifest.status = "running"
    manifest.write(context.manifest_path)
    try:
        predictions, base_rows = load_predictions_for_context(context)
        atomic_write_csv(
            context.layout.stage("primary_prediction").tables / "predictions.csv",
            base_rows,
        )
        rows = interface_stage(
            context, predictions, base_rows, label="primary", write_outputs=True
        )
        for row in rows:
            row["candidate_pool"] = bool(row.get("final_pass"))
        write_public_reports(context.layout, rows, clustering_status="not_run")
        manifest.artifact_sha256["public_all_results"] = (
            file_sha256(context.layout.all_results) or ""
        )
        manifest.artifact_sha256["public_candidates"] = (
            file_sha256(context.layout.candidates) or ""
        )
        manifest.artifact_sha256["backend_review"] = (
            file_sha256(context.layout.backend_review) or ""
        )
        failed = _interface_stage_failed(
            rows,
            energy_engine=context.config.interface.energy_engine,
        )
        manifest.stage_status["primary_interface"] = "partial" if failed else "success"
        manifest.status = "partial" if failed else "success"
        manifest.write(context.manifest_path)
        if failed and not context.config.project.allow_partial:
            raise PipelineExecutionError(
                "one or more interface analyses failed; partial results were preserved in "
                f"{context.results_dir}"
            )
        return rows
    except Exception as exc:
        if manifest.stage_status.get("primary_interface") == "running":
            manifest.stage_status["primary_interface"] = "error"
            manifest.status = "error"
        if str(exc) not in manifest.errors:
            manifest.errors.append(str(exc))
        manifest.write(context.manifest_path)
        raise


def _read_interface_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise PipelineExecutionError(f"interface candidates do not exist: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _clustering_input_paths(context: RunContext) -> tuple[Path, Path]:
    # Construct paths without ``RunOutputLayout.stage()``, whose ensure call
    # would create directories before standalone artifact validation.
    tables = (
        context.results_dir
        / "stages"
        / STAGE_DIRECTORIES["clustering"]
        / "tables"
    )
    return tables / "clustering_input.csv", tables / "clustering_candidates.csv"


def _persist_clustering_inputs(
    context: RunContext,
    rows: Sequence[dict[str, Any]],
    manifest: RunManifest,
) -> None:
    """Commit the complete post-ESM table and its candidate projection."""

    ordered = sorted((dict(row) for row in rows), key=_final_sort_key)
    candidates = [row for row in ordered if _row_truthy(row.get("candidate_pool"))]
    fieldnames = sorted({key for row in ordered for key in row})
    all_path, candidate_path = _clustering_input_paths(context)
    model_identities: dict[str, str] = {}
    for row in ordered:
        job_id = str(row.get("job_name", row.get("job_id", "")))
        model_value = row.get("effective_best_model_path")
        if model_value in (None, ""):
            continue
        model_sha = file_sha256(Path(str(model_value)))
        if model_sha is None:
            raise PipelineExecutionError(
                f"effective model is missing while freezing clustering input: {job_id}"
            )
        derived_sha = row.get("effective_derived_source_model_sha256")
        if derived_sha not in (None, "") and str(derived_sha) != model_sha:
            raise PipelineExecutionError(
                f"effective model content changed after interface analysis: {job_id}"
            )
        model_identities[job_id] = model_sha
    atomic_write_csv(all_path, ordered, fieldnames=fieldnames)
    atomic_write_csv(candidate_path, candidates, fieldnames=fieldnames)
    manifest.artifact_sha256["clustering_input_schema"] = sequence_sha256(
        json.dumps(fieldnames, separators=(",", ":"), ensure_ascii=True)
    )
    manifest.artifact_sha256["clustering_input"] = file_sha256(all_path) or ""
    manifest.artifact_sha256["clustering_candidates"] = (
        file_sha256(candidate_path) or ""
    )
    manifest.effective_model_sha256 = model_identities
    manifest.write(context.manifest_path)


def _row_truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _validated_clustering_inputs(
    context: RunContext,
    manifest: RunManifest,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate both canonical tables and every selected effective model."""

    all_path, candidate_path = _clustering_input_paths(context)
    for key, path in (
        ("clustering_input", all_path),
        ("clustering_candidates", candidate_path),
    ):
        expected = manifest.artifact_sha256.get(key)
        actual = file_sha256(path)
        if not expected or actual != expected:
            raise PipelineExecutionError(
                f"{key} does not match the run manifest: {path}"
            )
    all_rows = _read_interface_rows(all_path)
    candidate_rows = _read_interface_rows(candidate_path)
    if not all_rows:
        raise PipelineExecutionError("clustering_input.csv has no job rows")
    all_columns = set(all_rows[0])
    candidate_columns = set(candidate_rows[0]) if candidate_rows else all_columns
    with all_path.open(encoding="utf-8", newline="") as handle:
        all_header = next(csv.reader(handle), [])
    with candidate_path.open(encoding="utf-8", newline="") as handle:
        candidate_header = next(csv.reader(handle), [])
    schema_sha = sequence_sha256(
        json.dumps(all_header, separators=(",", ":"), ensure_ascii=True)
    )
    if (
        all_header != candidate_header
        or manifest.artifact_sha256.get("clustering_input_schema") != schema_sha
    ):
        raise PipelineExecutionError(
            "clustering input column schema does not match the run manifest"
        )
    required_columns = {
        "job_name",
        "candidate_pool",
        "effective_backend",
        "effective_status",
        "effective_best_model_path",
    }
    if context.config.scoring.esm.enabled and context.config.scoring.esm.esmfold:
        required_columns.add("esmfold_status")
    if (
        context.config.scoring.esm.enabled
        and context.config.scoring.esm.inverse_folding
    ):
        required_columns.add("esm_if_status")
    if not required_columns.issubset(all_columns) or candidate_columns != all_columns:
        raise PipelineExecutionError(
            "clustering input schema is incomplete or candidate/all schemas differ"
        )

    def indexed(rows: Sequence[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            job_id = str(row.get("job_name", row.get("job_id", "")))
            if not job_id or job_id in result:
                raise PipelineExecutionError(
                    f"{label} has a missing or duplicate job identity: {job_id!r}"
                )
            row["job_name"] = job_id
            result[job_id] = row
        return result

    all_by_job = indexed(all_rows, "clustering_input.csv")
    candidates_by_job = indexed(candidate_rows, "clustering_candidates.csv")
    planned_ids = {job.job_id for job in context.plan.jobs}
    selected_ids = {
        job_id
        for job_id, row in all_by_job.items()
        if _row_truthy(row.get("candidate_pool"))
    }
    if set(all_by_job) != planned_ids:
        raise PipelineExecutionError(
            "clustering_input.csv job membership does not match the immutable plan"
        )
    if set(candidates_by_job) != selected_ids:
        raise PipelineExecutionError(
            "clustering_candidates.csv membership does not match candidate_pool"
        )
    expected_bound_ids = {
        job_id
        for job_id, row in all_by_job.items()
        if row.get("effective_best_model_path") not in (None, "")
    }
    if set(manifest.effective_model_sha256) != expected_bound_ids:
        raise PipelineExecutionError(
            "effective model bindings do not match clustering_input.csv"
        )
    for job_id in expected_bound_ids:
        row = all_by_job[job_id]
        actual_model_sha = file_sha256(
            Path(str(row["effective_best_model_path"]))
        )
        if actual_model_sha != manifest.effective_model_sha256[job_id]:
            raise PipelineExecutionError(
                f"effective model does not match the run manifest for {job_id}"
            )
        derived_sha = row.get("effective_derived_source_model_sha256")
        if derived_sha not in (None, "") and str(derived_sha) != actual_model_sha:
            raise PipelineExecutionError(
                f"derived/model identity mismatch for {job_id}"
            )
    for job_id, candidate in candidates_by_job.items():
        if candidate != all_by_job[job_id]:
            raise PipelineExecutionError(
                f"candidate row differs from clustering input for {job_id}"
            )
        model_value = candidate.get("effective_best_model_path")
        expected_model_sha = manifest.effective_model_sha256.get(job_id)
        if model_value in (None, "") or not expected_model_sha:
            raise PipelineExecutionError(
                f"candidate {job_id} has no manifest-bound effective model"
            )
    return all_rows, candidate_rows


def run_clustering_only(context: RunContext) -> bool:
    feature_fingerprint = _context_feature_fingerprint(context)
    manifest = _existing_or_new_manifest(context, feature_fingerprint)
    stage_started = False
    try:
        all_rows, selected_rows = _validated_clustering_inputs(context, manifest)
        stage_started = True
        manifest.stage_status["clustering"] = "running"
        manifest.status = "running"
        manifest.write(context.manifest_path)
        selected_ids = {str(row["job_name"]) for row in selected_rows}
        jobs = tuple(job for job in context.plan.jobs if job.job_id in selected_ids)
        selected_predictions = tuple(
            _effective_predictions_from_rows(jobs, selected_rows)
        )
        outcome = (
            clustering_stage(
                context,
                selected_predictions,
                selected_rows,
                jobs=jobs,
                manifest=manifest,
            )
            if jobs
            else ClusteringOutcome(failed=False)
        )
        failed = outcome.failed
        write_public_reports(
            context.layout,
            all_rows,
            member_rows=outcome.member_rows,
            representative_rows=outcome.representative_rows,
            final_job_ids=tuple(
                str(row.get("job_name")) for row in outcome.final_rows
            ),
            clustering_status="partial" if failed else "success",
        )
        manifest.artifact_sha256["public_all_results"] = (
            file_sha256(context.layout.all_results) or ""
        )
        manifest.artifact_sha256["public_candidates"] = (
            file_sha256(context.layout.candidates) or ""
        )
        manifest.artifact_sha256["public_final_shortlist"] = (
            file_sha256(context.layout.final_shortlist) or ""
        )
        manifest.artifact_sha256["backend_review"] = (
            file_sha256(context.layout.backend_review) or ""
        )
        if context.config.runtime.dry_run:
            manifest.stage_status["clustering"] = "dry_run"
            manifest.status = "dry_run"
        else:
            manifest.stage_status["clustering"] = "partial" if failed else "success"
            manifest.status = "partial" if failed else "success"
        manifest.write(context.manifest_path)
        return failed
    except Exception as exc:
        if stage_started:
            manifest.stage_status["clustering"] = "error"
            manifest.status = "error"
            if str(exc) not in manifest.errors:
                manifest.errors.append(str(exc))
            manifest.write(context.manifest_path)
        raise
