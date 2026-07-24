"""Cohesive prediction stage orchestration boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    Sequence,
)
from af3_binder_filter.backends import (
    UnifiedPrediction,
    build_backend_command,
    output_adapter,
    write_backend_inputs,
)
from af3_binder_filter.config import BackendSettings
from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeatureBundle,
)
from af3_binder_filter.interface import structure_has_chains
from af3_binder_filter.io_utils import atomic_write_csv
from af3_binder_filter.jobs import (
    JobSpec,
    checkpoint_identity,
    job_fingerprint,
    sequence_sha256,
)
from af3_binder_filter.manifest import (
    JOB_MANIFEST_NAME,
    RunManifest,
    load_manifest,
    validate_legacy_input,
    write_job_manifest,
)
from af3_binder_filter.progress import (
    NullProgressReporter,
    PipelineProgressReporter,
)
from af3_binder_filter.secondary_features import SecondaryFeatureBundle
from af3_binder_filter.orchestration.command_runtime import (
    file_signature,
    path_belongs_to_job,
    return_code_failure_message,
    run_sharded_commands,
    small_json_is_complete,
    stable_completion_probe,
)
from af3_binder_filter.orchestration.context import (
    GpuJobShard,
    RunContext,
    container_name,
    record_gpu_assignments,
    runtime_gpus,
    plan_gpu_job_shards,
)
from af3_binder_filter.orchestration.feature_identity import (
    prediction_feature_identity,
)


def _input_for_job(input_paths: Sequence[Path], job: JobSpec, backend: str) -> Path:
    if backend == "alphafold3":
        return next(path for path in input_paths if path.stem == job.job_id)
    return input_paths[0]


def legacy_output_valid(
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


def backend_job_fingerprint(
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
        fingerprint = backend_job_fingerprint(
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
            and legacy_output_valid(
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
        if path_belongs_to_job(path, job.job_id)
        and small_json_is_complete(path)
    ]
    models = [
        path
        for path in set(models)
        if path_belongs_to_job(path, job.job_id)
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
            if path_belongs_to_job(path, job.job_id)
            and path.is_file()
            and path.stat().st_size > 0
        ]
        ranking = job_root / f"{job.job_id}_ranking_scores.csv"
        if not confidences or not ranking.is_file() or ranking.stat().st_size == 0:
            return ()
        required.extend(confidences)
        required.append(ranking)
    return file_signature(required)


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
    prediction_feature_identity = prediction_feature_identity(target_features)
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
            runtime_gpus(
                context,
                job_count=len(pending),
                stage_name=stage_name,
            ),
        )
        record_gpu_assignments(
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
                container_name=container_name(
                    context,
                    stage_name,
                    shard.gpu.index,
                ),
            )
            commands.append((shard, command))
        pending_by_id = {job.job_id: job for job in pending}
        completion_probe = stable_completion_probe(
            tuple(pending_by_id),
            lambda job_id: _prediction_completion_signature(
                backend.name,
                pending_by_id[job_id],
                output_root,
            ),
        )
        return_codes, command_errors = run_sharded_commands(
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
                    return_code_failure_message(
                        stage_name,
                        gpu_index,
                        return_code,
                    )
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
                fingerprint=backend_job_fingerprint(
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
        prediction_rows(active_jobs, predictions),
    )
    return predictions, stage_failed


def prediction_rows(
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
