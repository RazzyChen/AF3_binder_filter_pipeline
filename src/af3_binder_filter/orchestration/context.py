"""Cohesive context orchestration boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Iterable,
    Sequence,
)
from omegaconf import OmegaConf
from af3_binder_filter.config import (
    AerithConfig,
    compose_hydra_config,
)
from af3_binder_filter.config_tools import resolve_docker_image_id
from af3_binder_filter.features import _bundle as expected_feature_bundle
from af3_binder_filter.io_utils import atomic_write_yaml
from af3_binder_filter.jobs import (
    JobPlan,
    JobSpec,
    build_job_plan,
    feature_generation_fingerprint,
    file_sha256,
    job_fingerprint,
    run_fingerprint,
    run_provenance,
    sequence_sha256,
)
from af3_binder_filter.gpu import (
    GPUError,
    GPUInfo,
    query_gpus,
    select_free_gpus,
)
from af3_binder_filter.manifest import (
    MANIFEST_VERSION,
    RunManifest,
    load_manifest,
)
from af3_binder_filter.output_layout import (
    OUTPUT_SCHEMA_VERSION,
    RunOutputLayout,
)
from af3_binder_filter.progress import StageSpec
from af3_binder_filter.orchestration.stage_registry import progress_stage_specs


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


def _pipeline_stage_specs(config: AerithConfig) -> tuple[StageSpec, ...]:
    """Compatibility wrapper around the explicit ten-stage registry."""

    return progress_stage_specs(config)


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
