"""Cohesive esm stage orchestration boundary."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import (
    Any,
    Sequence,
)
from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.io_utils import atomic_write_csv
from af3_binder_filter.jobs import (
    JobSpec,
    sequence_sha256,
)
from af3_binder_filter.manifest import RunManifest
from af3_binder_filter.progress import (
    NullProgressReporter,
    PipelineProgressReporter,
)
from af3_binder_filter.esm_tools import (
    add_esmfold_backend_comparison,
    build_esm_if_container_command,
    build_esmfold_container_command,
    collect_esm_rows,
    load_cached_esm_rows,
    write_esm_inputs,
)
from af3_binder_filter.orchestration.command_runtime import (
    file_signature,
    return_code_failure_message,
    run_sharded_commands,
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


def esm_stage(
    context: RunContext,
    predictions: Sequence[UnifiedPrediction],
    manifest: RunManifest,
    *,
    primary_predictions: Sequence[UnifiedPrediction] = (),
    secondary_predictions: Sequence[UnifiedPrediction] = (),
    structure_rows: Sequence[dict[str, Any]] = (),
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
            structure_rows=structure_rows,
            primary_predictions=primary_predictions,
            secondary_predictions=secondary_predictions,
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
    write_esm_inputs(
        context.plan.jobs,
        predictions,
        input_dir,
        structure_rows=structure_rows,
    )
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
            runtime_gpus(
                context,
                job_count=len(jobs),
                stage_name=tool_name,
            ),
        )
        record_gpu_assignments(
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
            write_esm_inputs(
                shard.jobs,
                predictions,
                shard_input,
                structure_rows=structure_rows,
            )
            if tool_name == "esmfold":
                command = build_esmfold_container_command(
                    context.config,
                    input_dir=shard_input,
                    output_dir=shard_output,
                    gpu_index=shard.gpu.index,
                    container_name=container_name(
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
                    container_name=container_name(
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
        esmfold_probe = stable_completion_probe(
            tuple(job.job_id for job in context.plan.jobs),
            lambda job_id: file_signature(
                tuple(
                    path
                    for shard_output in shard_outputs
                    for path in (shard_output / "esmfold").glob(
                        f"{job_id}*.pdb"
                    )
                )
            ),
        )
        return_codes, errors = run_sharded_commands(
            context,
            "esmfold",
            commands,
            timeout_seconds=context.config.scoring.esm.timeout_seconds,
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
            return_code_failure_message("esmfold", gpu_index, code)
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
            structure_rows=structure_rows,
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
        esm_if_probe = stable_completion_probe(
            tuple(job.job_id for job in inverse_jobs),
            lambda job_id: file_signature(
                tuple(
                    shard_output
                    / ".aerith_progress"
                    / "esm_if"
                    / f"{sequence_sha256(job_id)}.json"
                    for shard_output in shard_outputs
                )
            ),
        )
        return_codes, errors = run_sharded_commands(
            context,
            "esm_if",
            commands,
            timeout_seconds=context.config.scoring.esm.timeout_seconds,
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
            return_code_failure_message("esm_if", gpu_index, code)
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
        structure_rows=structure_rows,
    )
    if primary_predictions:
        rows = add_esmfold_backend_comparison(
            rows,
            context.plan.jobs,
            primary_predictions,
            output_dir,
            label="primary",
            structure_rows=structure_rows,
        )
    if secondary_predictions:
        rows = add_esmfold_backend_comparison(
            rows,
            context.plan.jobs,
            secondary_predictions,
            output_dir,
            label="secondary",
            structure_rows=structure_rows,
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
