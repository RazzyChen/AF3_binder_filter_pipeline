"""Cohesive interface stage orchestration boundary."""

from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from pathlib import Path
from typing import (
    Any,
    Sequence,
)

from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.interface import (
    analyze_interface_geometry,
    apply_balanced_shortlist,
)
from af3_binder_filter.io_utils import atomic_write_csv
from af3_binder_filter.jobs import JobSpec
from af3_binder_filter.orchestration.context import RunContext
from af3_binder_filter.orchestration.factories import create_interface_energy_engine
from af3_binder_filter.progress import (
    NullProgressReporter,
    PipelineProgressReporter,
)


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
    engine = create_interface_energy_engine(context.config.interface)
    if engine is not None:
        reporter.task_started(
            stage_name,
            rosetta_task,
            total=len(active_jobs),
            detail=eligibility,
        )

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


def interface_stage_failed(rows: Sequence[dict[str, Any]], *, energy_engine: str) -> bool:
    """Propagate both geometry and configured energy-substage failures."""

    for row in rows:
        if row.get("interface_status") != "success":
            return True
        if energy_engine == "rosetta_cli" and row.get("rosetta_status") != "success":
            return True
    return False
