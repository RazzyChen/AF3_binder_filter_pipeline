"""Stable public entry points for production Aerith orchestration.

Individual stage modules own implementation details. Applications should use
these functions when they need a supported orchestration boundary.
"""

from af3_binder_filter.orchestration.context import (
    PipelineExecutionError,
    RunContext,
    create_run_context,
)
from af3_binder_filter.orchestration.feature_stage import run_prepare_features_only
from af3_binder_filter.orchestration.pipeline import run_pipeline
from af3_binder_filter.orchestration.resume import (
    load_predictions_for_context,
    run_clustering_only,
    run_interface_only,
)

__all__ = [
    "PipelineExecutionError",
    "RunContext",
    "create_run_context",
    "load_predictions_for_context",
    "run_pipeline",
    "run_prepare_features_only",
    "run_interface_only",
    "run_clustering_only",
]
