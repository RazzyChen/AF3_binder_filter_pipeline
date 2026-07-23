from __future__ import annotations

import importlib
import pkgutil
from dataclasses import fields

import af3_binder_filter.orchestration as orchestration
import af3_binder_filter.orchestration.pipeline as pipeline_module
from af3_binder_filter.orchestration.pipeline import PipelineRunner, PipelineState


def test_pipeline_state_names_all_cross_stage_values() -> None:
    assert {field.name for field in fields(PipelineState)} == {
        "expected_feature_fingerprint",
        "manifest",
        "required_failure",
        "primary_features",
        "primary_predictions",
        "primary_rows",
        "secondary_features",
        "secondary_predictions",
        "secondary_rows",
        "eligible_jobs",
        "final_rows",
        "effective_predictions",
        "candidates",
        "cluster_outcome",
    }


def test_pipeline_runner_keeps_stage_control_flow_explicit() -> None:
    expected_methods = {
        "_run_preflight",
        "_run_features",
        "_run_primary_prediction",
        "_run_primary_interface",
        "_run_secondary_features",
        "_run_secondary_prediction",
        "_run_secondary_interface",
        "_run_consensus",
        "_run_esm",
        "_run_clustering",
        "_finalize",
    }
    assert expected_methods <= set(vars(PipelineRunner))


def test_run_pipeline_remains_a_thin_compatibility_entry_point(monkeypatch) -> None:
    observed: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, context, *, reporter=None) -> None:
            observed["context"] = context
            observed["reporter"] = reporter

        def run(self):
            return [{"status": "success"}]

    context = object()
    reporter = object()
    monkeypatch.setattr(pipeline_module, "PipelineRunner", FakeRunner)

    assert pipeline_module.run_pipeline(context, reporter=reporter) == [
        {"status": "success"}
    ]
    assert observed == {"context": context, "reporter": reporter}


def test_every_orchestration_module_imports_without_cycles() -> None:
    module_names = sorted(
        module.name
        for module in pkgutil.iter_modules(
            orchestration.__path__,
            prefix=f"{orchestration.__name__}.",
        )
    )
    assert module_names
    for module_name in module_names:
        assert importlib.import_module(module_name).__name__ == module_name
