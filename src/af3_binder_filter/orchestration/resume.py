"""Cohesive resume orchestration boundary."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import (
    Any,
    Sequence,
)
from af3_binder_filter.backends import (
    UnifiedPrediction,
    output_adapter,
)
from af3_binder_filter.derived_structures import (
    DerivedStructureValidationError,
    validated_artifacts_from_row,
)
from af3_binder_filter.interface import structure_has_chains
from af3_binder_filter.io_utils import atomic_write_csv
from af3_binder_filter.jobs import (
    file_sha256,
    sequence_sha256,
)
from af3_binder_filter.manifest import (
    JOB_MANIFEST_NAME,
    RunManifest,
    load_manifest,
    write_job_manifest,
)
from af3_binder_filter.output_layout import STAGE_DIRECTORIES
from af3_binder_filter.reporting import write_public_reports
from af3_binder_filter.orchestration.clustering_stage import clustering_stage
from af3_binder_filter.orchestration.context import (
    ClusteringOutcome,
    PipelineExecutionError,
    RunContext,
    _context_feature_fingerprint,
    _existing_or_new_manifest,
)
from af3_binder_filter.orchestration.feature_identity import (
    _primary_prediction_feature_identity,
)
from af3_binder_filter.orchestration.interface_stage import (
    _interface_stage_failed,
    interface_stage,
)
from af3_binder_filter.orchestration.prediction_stage import (
    _backend_job_fingerprint,
    _legacy_output_valid,
    _prediction_rows,
)
from af3_binder_filter.orchestration.selection import (
    _effective_predictions_from_rows,
    _final_sort_key,
)


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
        try:
            validated_artifacts_from_row(
                candidate,
                prefix="effective",
                require_declared=True,
            )
        except DerivedStructureValidationError as exc:
            raise PipelineExecutionError(str(exc)) from exc
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
