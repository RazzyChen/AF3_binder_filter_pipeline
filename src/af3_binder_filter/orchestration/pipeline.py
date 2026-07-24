"""Explicit, stateful orchestration for the production Aerith pipeline."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

from af3_binder_filter.backends import (
    UnifiedPrediction,
    build_backend_command,
    write_backend_inputs,
)
from af3_binder_filter.clustering import write_cluster_outputs
from af3_binder_filter.config import validate_hydra_config
from af3_binder_filter.consensus import consensus_rows
from af3_binder_filter.effective import apply_effective_backend
from af3_binder_filter.esm_tools import (
    build_esmfold_container_command,
    write_esm_inputs,
)
from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeatureBundle,
    FeaturePreparation,
)
from af3_binder_filter.io_utils import atomic_write_csv, atomic_write_text
from af3_binder_filter.jobs import JobSpec, file_sha256
from af3_binder_filter.manifest import RunManifest
from af3_binder_filter.progress import (
    NullProgressReporter,
    PipelineProgressReporter,
    PipelineRunInfo,
)
from af3_binder_filter.reporting import write_public_reports
from af3_binder_filter.secondary_features import (
    SecondaryFeatureBundle,
    adapt_af3_features_for_secondary,
    adapt_local_features_for_secondary,
)

from af3_binder_filter.orchestration.clustering_stage import clustering_stage
from af3_binder_filter.orchestration.command_runtime import run_sharded_commands
from af3_binder_filter.orchestration.context import (
    ClusteringOutcome,
    GpuJobShard,
    PipelineExecutionError,
    RunContext,
    container_name,
    context_feature_fingerprint,
    existing_or_new_manifest,
    pipeline_stage_specs,
    record_gpu_assignments,
    runtime_gpus,
    plan_gpu_job_shards,
)
from af3_binder_filter.orchestration.esm_stage import esm_stage
from af3_binder_filter.orchestration.feature_identity import (
    bind_feature_content,
    target_feature_cache_hit,
)
from af3_binder_filter.orchestration.feature_stage import prepare_features_stage
from af3_binder_filter.orchestration.interface_stage import (
    interface_stage_failed,
    interface_stage,
)
from af3_binder_filter.orchestration.prediction_stage import (
    prediction_rows,
    prediction_stage,
)
from af3_binder_filter.orchestration.resume import (
    persist_clustering_inputs,
    validated_clustering_inputs,
)
from af3_binder_filter.orchestration.selection import (
    effective_predictions_from_rows,
    final_sort_key,
    merge_rows_by_job,
    secondary_gate_job_ids,
)


PrimaryFeatureBundle = AF3FeatureBundle | FeatureBundle


@dataclass(slots=True)
class PipelineState:
    """Mutable values that intentionally cross production stage boundaries."""

    expected_feature_fingerprint: str
    manifest: RunManifest
    required_failure: bool = False
    primary_features: PrimaryFeatureBundle | None = None
    primary_predictions: list[UnifiedPrediction] = field(default_factory=list)
    primary_rows: list[dict[str, Any]] = field(default_factory=list)
    secondary_features: SecondaryFeatureBundle | None = None
    secondary_predictions: list[UnifiedPrediction] = field(default_factory=list)
    secondary_rows: list[dict[str, Any]] = field(default_factory=list)
    eligible_jobs: tuple[JobSpec, ...] = ()
    final_rows: list[dict[str, Any]] = field(default_factory=list)
    effective_predictions: list[UnifiedPrediction] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    cluster_outcome: ClusteringOutcome | None = None


class PipelineRunner:
    """Run the fixed ten-stage pipeline while keeping control flow explicit."""

    def __init__(
        self,
        context: RunContext,
        *,
        reporter: PipelineProgressReporter | None = None,
    ) -> None:
        self.context = context
        self.reporter = reporter or NullProgressReporter()
        self.stage_specs = pipeline_stage_specs(context.config)
        self.active_stage: str | None = None
        self.pipeline_reported = False
        self._state: PipelineState | None = None

    @property
    def state(self) -> PipelineState:
        if self._state is None:
            raise RuntimeError("pipeline state has not been initialized")
        return self._state

    def run(self) -> list[dict[str, Any]]:
        self._start_pipeline_reporting()
        self._initialize_state()
        self._run_preflight()
        try:
            preparation = self._run_features()
            if self.context.config.runtime.dry_run:
                return self._complete_dry_run(preparation)
            self._run_primary_prediction()
            self._run_primary_interface()
            self._run_secondary()
            self._run_consensus()
            self._run_esm()
            self._run_clustering()
            return self._finalize()
        except KeyboardInterrupt:
            self._persist_interruption()
            raise
        except Exception as exc:
            self._persist_error(exc)
            raise

    def _start_pipeline_reporting(self) -> None:
        config = self.context.config
        self.reporter.pipeline_started(
            PipelineRunInfo(
                run_id=self.context.run_id,
                job_count=len(self.context.plan.jobs),
                primary_backend=config.backend.name,
                secondary_backend=(
                    config.secondary_backend.name
                    if config.secondary_backend.enabled
                    else "none"
                ),
                gpu_ids=tuple(config.runtime.gpu_ids),
                results_dir=self.context.results_dir,
                output_dir=Path(config.project.output_dir),
                logs_dir=self.context.results_dir / "stages",
            ),
            self.stage_specs,
        )

    def _initialize_state(self) -> None:
        expected = context_feature_fingerprint(self.context)
        manifest = existing_or_new_manifest(self.context, expected)
        manifest.write(self.context.manifest_path)
        self._state = PipelineState(expected, manifest)

    def _start_stage(self, stage: str) -> None:
        self.active_stage = stage
        self.reporter.stage_started(
            stage,
            log_dir=self.context.layout.stage(stage).logs,
        )

    def _finish_stage(self, stage: str, status: str, detail: str = "") -> None:
        self.reporter.stage_finished(stage, status=status, detail=detail)
        if self.active_stage == stage:
            self.active_stage = None

    def _finish_pipeline(self, status: str, detail: str = "") -> None:
        if self.pipeline_reported:
            return
        self.reporter.pipeline_finished(status=status, detail=detail)
        self.pipeline_reported = True

    def _run_preflight(self) -> None:
        config = self.context.config
        manifest = self.state.manifest
        self.reporter.message(
            "Preflight: validating configuration, images, paths, and GPU policy"
        )
        if config.runtime.dry_run:
            manifest.stage_status["preflight"] = "dry_run"
            self.reporter.message("Preflight: DRY-RUN")
            manifest.write(self.context.manifest_path)
            return

        validation = validate_hydra_config(config)
        errors = list(validation.errors)
        if config.backend.image_id is None:
            errors.append(
                f"backend Docker image is not available: {config.backend.image}"
            )
        if (
            config.secondary_backend.enabled
            and config.secondary_backend.image_id is None
        ):
            errors.append(
                "secondary backend Docker image is not available: "
                f"{config.secondary_backend.image}"
            )
        if errors:
            message = "pipeline preflight failed: " + "; ".join(errors)
            manifest.stage_status["preflight"] = "error"
            manifest.status = "error"
            manifest.errors.append(message)
            manifest.write(self.context.manifest_path)
            self._finish_pipeline("error", message)
            raise PipelineExecutionError(message)
        manifest.stage_status["preflight"] = "success"
        self.reporter.message("Preflight: SUCCESS", level="success")
        manifest.write(self.context.manifest_path)

    def _run_features(self) -> FeaturePreparation:
        manifest = self.state.manifest
        self._start_stage("features")
        manifest.stage_status["features"] = "running"
        manifest.write(self.context.manifest_path)
        cache_hit = target_feature_cache_hit(self.context)
        self.reporter.cache_status(
            "features",
            hits=int(cache_hit),
            misses=int(not cache_hit),
            total=1,
            force=self.context.config.runtime.force,
        )
        self.reporter.task_started(
            "features",
            "Target MSA/templates",
            total=1,
            completed=int(cache_hit),
        )
        preparation = prepare_features_stage(self.context)
        if self.context.config.runtime.dry_run:
            self.reporter.task_finished(
                "features",
                "Target MSA/templates",
                completed=1,
                total=1,
                skipped=1,
                detail="planned only",
            )
            self._finish_stage("features", "dry_run")
            return preparation

        bundle = preparation.bundle
        if bundle is None:
            raise PipelineExecutionError("feature preparation produced no bundle")
        if bundle.fingerprint != self.state.expected_feature_fingerprint:
            raise PipelineExecutionError(
                "prepared target feature fingerprint differs from the run plan"
            )
        if not isinstance(bundle, (AF3FeatureBundle, FeatureBundle)):
            raise PipelineExecutionError(
                "primary AF3 preparation did not return compatible local features"
            )
        bind_feature_content(manifest, bundle)
        self.state.primary_features = bundle
        manifest.stage_status["features"] = "success"
        manifest.write(self.context.manifest_path)
        detail = "cache hit" if preparation.reused else "cache missing resolved"
        self.reporter.task_finished(
            "features",
            "Target MSA/templates",
            completed=1,
            total=1,
            success=1,
            detail=detail,
        )
        self._finish_stage("features", "success", detail)
        return preparation

    def _complete_dry_run(
        self,
        preparation: FeaturePreparation,
    ) -> list[dict[str, Any]]:
        context = self.context
        config = context.config
        manifest = self.state.manifest
        message = (
            "# deferred until GPU MMseqs2 preprocessing produces "
            "validated features\n"
        )
        if preparation.bundle is not None:
            input_root = (
                Path(config.project.work_dir)
                / context.run_id
                / "inputs"
                / config.backend.name
            )
            write_backend_inputs(
                context.plan.jobs,
                config,
                input_dir=input_root,
                target_features=preparation.bundle,
                backend_settings=config.backend,
                force=True,
            )
            shards = plan_gpu_job_shards(
                context.plan.jobs,
                runtime_gpus(
                    context,
                    job_count=len(context.plan.jobs),
                    stage_name="primary_prediction",
                ),
            )
            record_gpu_assignments(
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
                    config,
                    input_dir=input_dir,
                    target_features=preparation.bundle,
                    backend_settings=config.backend,
                    force=True,
                )
                commands.append(
                    (
                        shard,
                        build_backend_command(
                            config,
                            input_dir=input_dir,
                            output_dir=(
                                Path(config.project.output_dir)
                                / context.run_id
                                / config.backend.name
                            ),
                            gpu_index=shard.gpu.index,
                            feature_dir=preparation.bundle.cache_dir,
                            backend_settings=config.backend,
                            container_name=container_name(
                                context,
                                "primary_prediction",
                                shard.gpu.index,
                            ),
                        ),
                    )
                )
            run_sharded_commands(context, "primary_prediction", commands)
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

        if config.scoring.esm.enabled:
            self._write_dry_run_esm_commands(message)
        if config.secondary_backend.enabled:
            atomic_write_text(
                context.layout.stage("secondary_prediction").logs
                / "secondary_prediction.command.txt",
                message,
            )
        for stage_spec in self.stage_specs:
            stage = stage_spec.key
            manifest.stage_status[stage] = "dry_run"
            if stage == "features":
                continue
            self._start_stage(stage)
            self.reporter.task_started(
                stage,
                "Planned jobs",
                total=len(context.plan.jobs),
                completed=len(context.plan.jobs),
                detail="dry-run",
            )
            self.reporter.task_finished(
                stage,
                "Planned jobs",
                completed=len(context.plan.jobs),
                total=len(context.plan.jobs),
                skipped=len(context.plan.jobs),
                detail="dry-run",
            )
            self._finish_stage(stage, "dry_run")
        manifest.status = "dry_run"
        manifest.write(context.manifest_path)
        write_public_reports(context.layout, (), clustering_status="dry_run")
        self._finish_pipeline(
            "dry_run",
            f"{len(context.plan.jobs)} jobs planned; results in {context.results_dir}",
        )
        return []

    def _write_dry_run_esm_commands(self, message: str) -> None:
        context = self.context
        config = context.config
        manifest = self.state.manifest
        esm_input = Path(config.project.work_dir) / context.run_id / "esm_inputs"
        esm_layout = context.layout.stage("esm")
        esm_output = esm_layout.artifacts / "esm"
        esm_output.mkdir(parents=True, exist_ok=True)
        missing_predictions = [
            UnifiedPrediction(job.job_id, "alphafold3", "missing")
            for job in context.plan.jobs
        ]
        write_esm_inputs(context.plan.jobs, missing_predictions, esm_input)
        if config.scoring.esm.esmfold:
            shards = plan_gpu_job_shards(
                context.plan.jobs,
                runtime_gpus(
                    context,
                    job_count=len(context.plan.jobs),
                    stage_name="esmfold",
                ),
            )
            record_gpu_assignments(
                manifest,
                context.manifest_path,
                "esmfold",
                shards,
            )
            commands: list[tuple[GpuJobShard, Sequence[str]]] = []
            for shard in shards:
                shard_input = esm_input / "dry_run" / f"gpu_{shard.gpu.index}"
                shard_output = esm_output / "dry_run" / f"gpu_{shard.gpu.index}"
                write_esm_inputs(shard.jobs, missing_predictions, shard_input)
                commands.append(
                    (
                        shard,
                        build_esmfold_container_command(
                            config,
                            input_dir=shard_input,
                            output_dir=shard_output,
                            gpu_index=shard.gpu.index,
                            container_name=container_name(
                                context,
                                "esmfold",
                                shard.gpu.index,
                            ),
                        ),
                    )
                )
            run_sharded_commands(
                context,
                "esmfold",
                commands,
                timeout_seconds=config.scoring.esm.timeout_seconds,
            )
        else:
            atomic_write_text(esm_layout.logs / "esmfold.command.txt", "# disabled\n")
        atomic_write_text(esm_layout.logs / "esm_if.command.txt", message)

    def _run_primary_prediction(self) -> None:
        features = self.state.primary_features
        if features is None:
            raise PipelineExecutionError("primary features are unavailable")
        manifest = self.state.manifest
        self._start_stage("primary_prediction")
        manifest.stage_status["primary_prediction"] = "running"
        manifest.write(self.context.manifest_path)
        predictions, failed = prediction_stage(
            self.context,
            features,
            manifest,
            stage_name="primary_prediction",
            reporter=self.reporter,
        )
        self.state.primary_predictions = predictions
        manifest.stage_status["primary_prediction"] = "partial" if failed else "success"
        self._finish_stage(
            "primary_prediction",
            manifest.stage_status["primary_prediction"],
        )
        self.state.required_failure |= failed

    def _run_primary_interface(self) -> None:
        state = self.state
        state.primary_rows = prediction_rows(
            self.context.plan.jobs,
            state.primary_predictions,
        )
        self._start_stage("primary_interface")
        state.manifest.stage_status["primary_interface"] = "running"
        state.manifest.write(self.context.manifest_path)
        state.primary_rows = interface_stage(
            self.context,
            state.primary_predictions,
            state.primary_rows,
            label="primary",
            write_outputs=True,
            reporter=self.reporter,
        )
        failed = interface_stage_failed(
            state.primary_rows,
            energy_engine=self.context.config.interface.energy_engine,
        )
        status = "partial" if failed else "success"
        state.manifest.stage_status["primary_interface"] = status
        self._finish_stage("primary_interface", status)
        state.required_failure |= failed

    def _run_secondary(self) -> None:
        if not self.context.config.secondary_backend.enabled:
            self.state.manifest.stage_status.update(
                {
                    "secondary_features": "disabled",
                    "secondary_prediction": "disabled",
                    "secondary_interface": "disabled",
                }
            )
            return
        self._select_secondary_jobs()
        self._run_secondary_features()
        self._run_secondary_prediction()
        self._run_secondary_interface()

    def _select_secondary_jobs(self) -> None:
        config = self.context.config
        threshold = config.secondary_backend.minimum_primary_iptm
        eligible_ids = secondary_gate_job_ids(
            self.state.primary_predictions,
            threshold,
        )
        self.state.eligible_jobs = tuple(
            replace(
                job,
                backend=config.secondary_backend.name,
                model=config.secondary_backend.model,
            )
            for job in self.context.plan.jobs
            if job.job_id in eligible_ids
        )

    def _secondary_detail(self) -> str:
        return (
            f"{len(self.state.eligible_jobs)} eligible / "
            f"{len(self.context.plan.jobs)} total"
        )

    def _run_secondary_features(self) -> None:
        primary_features = self.state.primary_features
        if primary_features is None:
            raise PipelineExecutionError("primary features are unavailable")
        detail = self._secondary_detail()
        manifest = self.state.manifest
        self._start_stage("secondary_features")
        manifest.stage_status["secondary_features"] = "running"
        self.reporter.task_started(
            "secondary_features",
            "Adapt AF3 target features",
            total=1,
            detail=detail,
        )
        if isinstance(primary_features, FeatureBundle):
            secondary_features = adapt_local_features_for_secondary(
                primary_features,
                self.context.plan.target_sequence,
                force=self.context.config.runtime.force,
            )
        else:
            secondary_features = adapt_af3_features_for_secondary(
                primary_features,
                self.context.plan.target_sequence,
                force=self.context.config.runtime.force,
            )
        self.state.secondary_features = secondary_features
        manifest.stage_status["secondary_features"] = "success"
        self.reporter.task_finished(
            "secondary_features",
            "Adapt AF3 target features",
            completed=1,
            total=1,
            success=1,
            detail=detail,
        )
        self._finish_stage("secondary_features", "success", detail)

    def _run_secondary_prediction(self) -> None:
        state = self.state
        detail = self._secondary_detail()
        self._start_stage("secondary_prediction")
        state.manifest.stage_status["secondary_prediction"] = "running"
        if state.eligible_jobs:
            if state.secondary_features is None:
                raise PipelineExecutionError("secondary features are unavailable")
            predictions, failed = prediction_stage(
                self.context,
                state.secondary_features,
                state.manifest,
                jobs=state.eligible_jobs,
                backend_settings=self.context.config.secondary_backend,
                stage_name="secondary_prediction",
                reporter=self.reporter,
            )
            state.secondary_predictions = predictions
            state.secondary_rows = prediction_rows(
                state.eligible_jobs,
                state.secondary_predictions,
            )
            status = "partial" if failed else "success"
        else:
            failed = False
            status = "skipped"
            self.reporter.cache_status(
                "secondary_prediction",
                hits=0,
                misses=0,
                total=0,
            )
            self.reporter.task_started(
                "secondary_prediction",
                f"{self.context.config.secondary_backend.name} predictions",
                total=0,
                detail=detail,
            )
            self.reporter.task_finished(
                "secondary_prediction",
                f"{self.context.config.secondary_backend.name} predictions",
                completed=0,
                total=0,
                skipped=0,
                detail="no eligible jobs",
            )
        state.manifest.stage_status["secondary_prediction"] = status
        self._finish_stage("secondary_prediction", status, detail)
        state.required_failure |= failed

    def _run_secondary_interface(self) -> None:
        state = self.state
        detail = self._secondary_detail()
        self._start_stage("secondary_interface")
        state.manifest.stage_status["secondary_interface"] = "running"
        if state.eligible_jobs:
            state.secondary_rows = interface_stage(
                self.context,
                state.secondary_predictions,
                state.secondary_rows,
                jobs=state.eligible_jobs,
                label="secondary",
                write_outputs=True,
                reporter=self.reporter,
            )
        else:
            self.reporter.task_started(
                "secondary_interface",
                "Interface analysis",
                total=0,
                detail=detail,
            )
            self.reporter.task_finished(
                "secondary_interface",
                "Interface analysis",
                completed=0,
                total=0,
                skipped=0,
                detail="no eligible jobs",
            )
        failed = interface_stage_failed(
            state.secondary_rows,
            energy_engine=self.context.config.interface.energy_engine,
        )
        status = (
            "skipped"
            if not state.eligible_jobs
            else ("partial" if failed else "success")
        )
        state.manifest.stage_status["secondary_interface"] = status
        self._finish_stage("secondary_interface", status, detail)
        state.required_failure |= failed

    def _run_consensus(self) -> None:
        context = self.context
        config = context.config
        state = self.state
        manifest = state.manifest
        self._start_stage("consensus")
        manifest.stage_status["consensus"] = "running"
        manifest.write(context.manifest_path)
        self.reporter.task_started(
            "consensus",
            "Merge backend results",
            total=len(context.plan.jobs),
        )
        if config.secondary_backend.enabled:
            final_rows = consensus_rows(
                state.primary_rows,
                state.secondary_rows,
                config.consensus,
            )
            failed = any(
                row.get("secondary_status") == "success"
                and row.get("consensus_status") != "success"
                for row in final_rows
            )
        else:
            final_rows = []
            for primary in state.primary_rows:
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
            failed = False
        status = "partial" if failed else "success"
        manifest.stage_status["consensus"] = status
        error_count = sum(
            row.get("consensus_status") == "error" for row in final_rows
        )
        self.reporter.task_finished(
            "consensus",
            "Merge backend results",
            completed=len(context.plan.jobs),
            total=len(context.plan.jobs),
            success=len(context.plan.jobs) - error_count,
            failed=error_count,
        )
        state.required_failure |= failed
        self._annotate_candidate_pool(final_rows)
        state.final_rows = [apply_effective_backend(row) for row in final_rows]
        state.effective_predictions = effective_predictions_from_rows(
            context.plan.jobs,
            state.final_rows,
        )
        state.candidates = [
            row for row in state.final_rows if row.get("candidate_pool")
        ]
        self._write_consensus_outputs()
        self._finish_stage("consensus", status)

    def _annotate_candidate_pool(self, final_rows: list[dict[str, Any]]) -> None:
        config = self.context.config
        secondary_by_job = {
            str(row["job_name"]): row for row in self.state.secondary_rows
        }
        eligible_ids = {job.job_id for job in self.state.eligible_jobs}
        for row in final_rows:
            job_name = str(row["job_name"])
            primary_pass = bool(row.get("final_pass"))
            if not config.secondary_backend.enabled:
                row["secondary_gate_pass"] = False
                row["cross_validation_pass"] = None
                row["candidate_pool"] = primary_pass
                continue

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
                    >= config.consensus.same_fold_tm_threshold
                )
                different_pose = (
                    float(row.get("consensus_binder_fixed_frame_rmsd"))
                    >= config.consensus.different_pose_rmsd_threshold
                )
                if same_fold and different_pose:
                    review_reasons.add("same_fold_different_pose")
            except (TypeError, ValueError):
                pass
            if review_reasons:
                row["manual_review"] = True
                row["manual_review_reason"] = ";".join(sorted(review_reasons))

    def _write_consensus_outputs(self) -> None:
        state = self.state
        layout = self.context.layout.stage("consensus")
        results_path = layout.tables / "consensus_results.csv"
        candidates_path = layout.tables / "candidates_full.csv"
        manual_review = [
            row for row in state.final_rows if row.get("manual_review")
        ]
        atomic_write_csv(results_path, state.final_rows)
        atomic_write_csv(candidates_path, state.candidates)
        atomic_write_csv(
            layout.tables / "secondary_backend_rows.csv",
            state.secondary_rows,
        )
        atomic_write_csv(layout.tables / "manual_review.csv", manual_review)
        state.manifest.artifact_sha256["consensus_results"] = (
            file_sha256(results_path) or ""
        )
        state.manifest.artifact_sha256["consensus_candidates"] = (
            file_sha256(candidates_path) or ""
        )
        state.manifest.write(self.context.manifest_path)

    def _run_esm(self) -> None:
        state = self.state
        config = self.context.config
        if config.scoring.esm.enabled:
            self._start_stage("esm")
            state.manifest.stage_status["esm"] = "running"
            state.manifest.write(self.context.manifest_path)
            esm_rows, failed = esm_stage(
                self.context,
                state.effective_predictions,
                state.manifest,
                primary_predictions=state.primary_predictions,
                secondary_predictions=state.secondary_predictions,
                structure_rows=state.final_rows,
                reporter=self.reporter,
            )
            status = "partial" if failed else "success"
            state.manifest.stage_status["esm"] = status
            self._finish_stage("esm", status)
        else:
            esm_rows, failed = esm_stage(
                self.context,
                state.effective_predictions,
                state.manifest,
                structure_rows=state.final_rows,
            )
            state.manifest.stage_status["esm"] = "disabled"
        state.final_rows = merge_rows_by_job(state.final_rows, esm_rows)
        state.final_rows.sort(key=final_sort_key)
        state.candidates = [
            row for row in state.final_rows if row.get("candidate_pool")
        ]
        state.required_failure |= failed

    def _run_clustering(self) -> None:
        context = self.context
        state = self.state
        persist_clustering_inputs(context, state.final_rows, state.manifest)
        _all_rows, clustering_candidates = validated_clustering_inputs(
            context,
            state.manifest,
        )
        self._start_stage("clustering")
        state.manifest.stage_status["clustering"] = "running"
        state.manifest.write(context.manifest_path)
        candidate_ids = {
            str(row["job_name"]) for row in clustering_candidates
        }
        cluster_jobs = tuple(
            job for job in context.plan.jobs if job.job_id in candidate_ids
        )
        cluster_predictions = tuple(
            effective_predictions_from_rows(cluster_jobs, clustering_candidates)
        )
        detail = (
            f"{len(cluster_jobs)} candidates / "
            f"{len(context.plan.jobs)} total"
        )
        self.reporter.task_started(
            "clustering",
            "Foldseek clustering",
            total=None,
            detail=detail,
        )
        if cluster_jobs:
            outcome = clustering_stage(
                context,
                cluster_predictions,
                clustering_candidates,
                jobs=cluster_jobs,
                manifest=state.manifest,
            )
        else:
            outcome = self._empty_clustering_outcome()
        state.cluster_outcome = outcome
        status = "partial" if outcome.failed else "success"
        state.manifest.stage_status["clustering"] = status
        self.reporter.task_finished(
            "clustering",
            "Foldseek clustering",
            completed=len(cluster_jobs),
            total=len(cluster_jobs),
            success=(0 if outcome.failed else len(cluster_jobs)),
            failed=(len(cluster_jobs) if outcome.failed else 0),
            detail=(
                f"representatives={len(outcome.representative_rows)} "
                f"shortlist={len(outcome.final_rows)}"
            ),
        )
        self._finish_stage(
            "clustering",
            status,
            f"{detail}; representatives={len(outcome.representative_rows)}",
        )
        state.required_failure |= outcome.failed

    def _empty_clustering_outcome(self) -> ClusteringOutcome:
        layout = self.context.layout.stage("clustering")
        member_rows, representative_rows, shortlist_rows = write_cluster_outputs(
            results_dir=layout.tables,
            artifacts_dir=layout.artifacts,
            jobs=(),
            rows=(),
            binder_membership={},
            binder_raw_representatives={},
            complex_membership={},
            complex_raw_representatives={},
            epitope_membership={},
            epitope_raw_representatives={},
        )
        return ClusteringOutcome(
            failed=False,
            member_rows=tuple(member_rows),
            representative_rows=tuple(representative_rows),
            final_rows=tuple(shortlist_rows),
        )

    def _finalize(self) -> list[dict[str, Any]]:
        state = self.state
        outcome = state.cluster_outcome
        if outcome is None:
            raise PipelineExecutionError("clustering outcome is unavailable")
        write_public_reports(
            self.context.layout,
            state.final_rows,
            member_rows=outcome.member_rows,
            representative_rows=outcome.representative_rows,
            final_job_ids=tuple(
                str(row.get("job_name")) for row in outcome.final_rows
            ),
            clustering_status=state.manifest.stage_status["clustering"],
        )
        self._bind_public_report_hashes()
        state.manifest.status = "partial" if state.required_failure else "success"
        state.manifest.write(self.context.manifest_path)
        if state.required_failure and not self.context.config.project.allow_partial:
            raise PipelineExecutionError(
                "one or more required stages failed; partial results were preserved in "
                f"{self.context.results_dir}"
            )
        interface_success = sum(
            row.get("interface_status") == "success" for row in state.final_rows
        )
        self._finish_pipeline(
            state.manifest.status,
            (
                f"interface_success={interface_success}/{len(state.final_rows)} "
                f"candidates={len(state.candidates)} "
                f"shortlist={len(outcome.final_rows)}; "
                f"results in {self.context.results_dir}"
            ),
        )
        return state.final_rows

    def _bind_public_report_hashes(self) -> None:
        manifest = self.state.manifest
        layout = self.context.layout
        manifest.artifact_sha256["public_all_results"] = (
            file_sha256(layout.all_results) or ""
        )
        manifest.artifact_sha256["public_candidates"] = (
            file_sha256(layout.candidates) or ""
        )
        manifest.artifact_sha256["public_final_shortlist"] = (
            file_sha256(layout.final_shortlist) or ""
        )
        manifest.artifact_sha256["backend_review"] = (
            file_sha256(layout.backend_review) or ""
        )

    def _persist_interruption(self) -> None:
        manifest = self.state.manifest
        message = "pipeline interrupted by user"
        interrupted_stage = self.active_stage
        for stage, status in tuple(manifest.stage_status.items()):
            if status == "running":
                manifest.stage_status[stage] = "interrupted"
        manifest.status = "interrupted"
        if message not in manifest.errors:
            manifest.errors.append(message)
        manifest.write(self.context.manifest_path)
        if interrupted_stage is not None:
            self._finish_stage(interrupted_stage, "interrupted", message)
        self._finish_pipeline("interrupted", message)

    def _persist_error(self, exc: Exception) -> None:
        manifest = self.state.manifest
        message = str(exc)
        if self.active_stage is not None:
            self._finish_stage(self.active_stage, "error", message)
        for stage, status in tuple(manifest.stage_status.items()):
            if status == "running":
                manifest.stage_status[stage] = "error"
        if message not in manifest.errors:
            manifest.errors.append(message)
        if manifest.status != "partial":
            manifest.status = "error"
        manifest.write(self.context.manifest_path)
        self._finish_pipeline(manifest.status, message)


def run_pipeline(
    context: RunContext,
    *,
    reporter: PipelineProgressReporter | None = None,
) -> list[dict[str, Any]]:
    """Compatibility entry point for CLI and external callers."""

    return PipelineRunner(context, reporter=reporter).run()
