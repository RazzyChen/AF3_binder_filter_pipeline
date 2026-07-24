"""Cohesive clustering stage orchestration boundary."""

from __future__ import annotations

import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import (
    Any,
    Sequence,
)
from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.clustering import (
    build_foldseek_container_command,
    greedy_epitope_clusters,
    parse_foldseek_clusters,
    prepare_foldseek_inputs,
    run_foldseek_command,
    write_cluster_outputs,
)
from af3_binder_filter.io_utils import atomic_write_json
from af3_binder_filter.jobs import JobSpec
from af3_binder_filter.manifest import RunManifest
from af3_binder_filter.orchestration.context import (
    ClusteringOutcome,
    GpuJobShard,
    RunContext,
    container_name,
    record_gpu_assignments,
    runtime_gpus,
)


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
        rows=rows,
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
    gpus = runtime_gpus(
        context,
        job_count=requested_workers,
        stage_name="clustering",
    )
    binder_gpu = gpus[0]
    complex_gpu = gpus[1] if len(gpus) > 1 else gpus[0]
    if manifest is not None:
        record_gpu_assignments(
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
        container_name=container_name(context, "foldseek-binder", binder_gpu.index),
    )
    complex_command = build_foldseek_container_command(
        context.config.clustering,
        layer="complex",
        docker_bin=context.config.backend.docker_bin,
        image=context.config.backend.image,
        gpu_index=complex_gpu.index,
        input_dir=complex_dir,
        execution_dir=foldseek_result_dir,
        container_name=container_name(context, "foldseek-complex", complex_gpu.index),
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
