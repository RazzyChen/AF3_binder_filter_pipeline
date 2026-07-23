"""Stable compatibility facade for Aerith pipeline orchestration.

Implementation lives in :mod:`af3_binder_filter.orchestration`; new code
should import the owning module when monkeypatching implementation details.
"""

from __future__ import annotations

from af3_binder_filter.orchestration.context import (
    PipelineExecutionError,
    RunContext,
    ClusteringOutcome,
    GpuJobShard,
    _job_estimated_cost,
    plan_gpu_job_shards,
    _runtime_gpus,
    _container_name,
    _record_gpu_assignments,
    _overrides_with_runtime,
    _af3_feature_fingerprint,
    _expected_feature_fingerprint,
    _expected_feature_cache_dir,
    _pipeline_stage_specs,
    create_run_context,
    _new_manifest,
    _context_provenance,
    _context_feature_fingerprint,
    _existing_or_new_manifest,
    _manifest_from_payload,
)
from af3_binder_filter.orchestration.feature_identity import (
    _target_feature_cache_hit,
    _absolute_target_features,
    _af3_bundle_from_manifest,
    _af3_bundle_artifact_identity,
    _bind_feature_content,
    _prediction_feature_identity,
    _primary_prediction_feature_identity,
)
from af3_binder_filter.orchestration.command_runtime import (
    _command_stage_name,
    _run_prediction_command,
    _run_sharded_commands,
    _return_code_failure_message,
    _file_signature,
    _small_json_is_complete,
    _path_belongs_to_job,
    _stable_completion_probe,
)
from af3_binder_filter.orchestration.feature_stage import (
    prepare_features_stage,
    run_prepare_features_only,
    _prepare_af3_target_features,
)
from af3_binder_filter.orchestration.prediction_stage import (
    _input_for_job,
    _legacy_output_valid,
    _backend_job_fingerprint,
    _reusable_predictions,
    _prediction_completion_signature,
    _prediction_artifact_signature,
    prediction_stage,
    _prediction_rows,
)
from af3_binder_filter.orchestration.interface_stage import (
    interface_stage,
    _interface_stage_failed,
)
from af3_binder_filter.orchestration.clustering_stage import (
    clustering_stage,
)
from af3_binder_filter.orchestration.esm_stage import (
    esm_stage,
)
from af3_binder_filter.orchestration.selection import (
    _merge_rows_by_job,
    _optional_float,
    _effective_predictions_from_rows,
    _final_sort_key,
    secondary_gate_job_ids,
)
from af3_binder_filter.orchestration.resume import (
    load_predictions_for_context,
    run_interface_only,
    _read_interface_rows,
    _clustering_input_paths,
    _persist_clustering_inputs,
    _row_truthy,
    _validated_clustering_inputs,
    run_clustering_only,
)
from af3_binder_filter.orchestration.pipeline import (
    run_pipeline,
)

__all__ = [
    'PipelineExecutionError',
    'RunContext',
    'ClusteringOutcome',
    'GpuJobShard',
    '_job_estimated_cost',
    'plan_gpu_job_shards',
    '_runtime_gpus',
    '_container_name',
    '_record_gpu_assignments',
    '_overrides_with_runtime',
    '_af3_feature_fingerprint',
    '_expected_feature_fingerprint',
    '_expected_feature_cache_dir',
    '_pipeline_stage_specs',
    'create_run_context',
    '_new_manifest',
    '_context_provenance',
    '_context_feature_fingerprint',
    '_existing_or_new_manifest',
    '_manifest_from_payload',
    '_target_feature_cache_hit',
    '_absolute_target_features',
    '_af3_bundle_from_manifest',
    '_af3_bundle_artifact_identity',
    '_bind_feature_content',
    '_prediction_feature_identity',
    '_primary_prediction_feature_identity',
    '_command_stage_name',
    '_run_prediction_command',
    '_run_sharded_commands',
    '_return_code_failure_message',
    '_file_signature',
    '_small_json_is_complete',
    '_path_belongs_to_job',
    '_stable_completion_probe',
    'prepare_features_stage',
    'run_prepare_features_only',
    '_prepare_af3_target_features',
    '_input_for_job',
    '_legacy_output_valid',
    '_backend_job_fingerprint',
    '_reusable_predictions',
    '_prediction_completion_signature',
    '_prediction_artifact_signature',
    'prediction_stage',
    '_prediction_rows',
    'interface_stage',
    '_interface_stage_failed',
    'clustering_stage',
    'esm_stage',
    '_merge_rows_by_job',
    '_optional_float',
    '_effective_predictions_from_rows',
    '_final_sort_key',
    'secondary_gate_job_ids',
    'load_predictions_for_context',
    'run_interface_only',
    '_read_interface_rows',
    '_clustering_input_paths',
    '_persist_clustering_inputs',
    '_row_truthy',
    '_validated_clustering_inputs',
    'run_clustering_only',
    'run_pipeline',
]
