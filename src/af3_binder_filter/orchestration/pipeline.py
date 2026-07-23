"""Cohesive pipeline orchestration boundary."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    Sequence,
)
from af3_binder_filter.backends import (
    UnifiedPrediction,
    build_backend_command,
    write_backend_inputs,
)
from af3_binder_filter.clustering import write_cluster_outputs
from af3_binder_filter.config import validate_hydra_config
from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeatureBundle,
)
from af3_binder_filter.io_utils import (
    atomic_write_csv,
    atomic_write_text,
)
from af3_binder_filter.jobs import (
    JobSpec,
    file_sha256,
)
from af3_binder_filter.progress import (
    NullProgressReporter,
    PipelineProgressReporter,
    PipelineRunInfo,
)
from af3_binder_filter.reporting import write_public_reports
from af3_binder_filter.secondary_features import adapt_af3_features_for_secondary
from af3_binder_filter.secondary_features import adapt_local_features_for_secondary
from af3_binder_filter.consensus import consensus_rows
from af3_binder_filter.effective import apply_effective_backend
from af3_binder_filter.esm_tools import (
    build_esmfold_container_command,
    write_esm_inputs,
)
from af3_binder_filter.orchestration.clustering_stage import clustering_stage
from af3_binder_filter.orchestration.command_runtime import _run_sharded_commands
from af3_binder_filter.orchestration.context import (
    ClusteringOutcome,
    GpuJobShard,
    PipelineExecutionError,
    RunContext,
    _container_name,
    _context_feature_fingerprint,
    _existing_or_new_manifest,
    _pipeline_stage_specs,
    _record_gpu_assignments,
    _runtime_gpus,
    plan_gpu_job_shards,
)
from af3_binder_filter.orchestration.esm_stage import esm_stage
from af3_binder_filter.orchestration.feature_identity import (
    _bind_feature_content,
    _target_feature_cache_hit,
)
from af3_binder_filter.orchestration.feature_stage import prepare_features_stage
from af3_binder_filter.orchestration.interface_stage import (
    _interface_stage_failed,
    interface_stage,
)
from af3_binder_filter.orchestration.prediction_stage import (
    _prediction_rows,
    prediction_stage,
)
from af3_binder_filter.orchestration.resume import (
    _persist_clustering_inputs,
    _validated_clustering_inputs,
)
from af3_binder_filter.orchestration.selection import (
    _effective_predictions_from_rows,
    _final_sort_key,
    _merge_rows_by_job,
    secondary_gate_job_ids,
)


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
                        timeout_seconds=context.config.scoring.esm.timeout_seconds,
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
                structure_rows=final_rows,
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
                structure_rows=final_rows,
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
    except KeyboardInterrupt:
        message = "pipeline interrupted by user"
        interrupted_stage = active_stage
        for stage, status in tuple(manifest.stage_status.items()):
            if status == "running":
                manifest.stage_status[stage] = "interrupted"
        manifest.status = "interrupted"
        if message not in manifest.errors:
            manifest.errors.append(message)
        manifest.write(context.manifest_path)
        if interrupted_stage is not None:
            finish_stage(interrupted_stage, "interrupted", message)
        finish_pipeline("interrupted", message)
        raise
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
