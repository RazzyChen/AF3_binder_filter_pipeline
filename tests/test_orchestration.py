from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest

from af3_binder_filter.config import AerithConfig
from af3_binder_filter.orchestration.factories import (
    ComponentFactoryError,
    create_interface_energy_engine,
)
from af3_binder_filter.orchestration.pipeline import run_pipeline
from af3_binder_filter.orchestration.stage_registry import (
    PIPELINE_STAGE_REGISTRY,
    enabled_stage_definitions,
    progress_stage_specs,
)
from af3_binder_filter.rosetta import RosettaCliEngine


def test_stage_registry_has_one_explicit_order_for_all_ten_stages() -> None:
    assert [stage.key for stage in PIPELINE_STAGE_REGISTRY] == [
        "preflight",
        "features",
        "primary_prediction",
        "primary_interface",
        "secondary_features",
        "secondary_prediction",
        "secondary_interface",
        "consensus",
        "esm",
        "clustering",
    ]


def test_stage_registry_applies_secondary_and_esm_conditions() -> None:
    config = AerithConfig()
    config.secondary_backend.enabled = False
    config.scoring.esm.enabled = False

    assert [stage.key for stage in enabled_stage_definitions(config)] == [
        "preflight",
        "features",
        "primary_prediction",
        "primary_interface",
        "consensus",
        "clustering",
    ]
    assert [stage.key for stage in progress_stage_specs(config)] == [
        "features",
        "primary_prediction",
        "primary_interface",
        "consensus",
        "clustering",
    ]


def test_interface_energy_factory_is_explicit_and_typed() -> None:
    config = AerithConfig()
    assert isinstance(
        create_interface_energy_engine(config.interface),
        RosettaCliEngine,
    )

    config.interface.energy_engine = "none"
    assert create_interface_energy_engine(config.interface) is None

    config.interface.energy_engine = "mystery"
    with pytest.raises(ComponentFactoryError, match="unsupported"):
        create_interface_energy_engine(config.interface)


def test_pipeline_is_owned_by_the_pipeline_module() -> None:
    assert run_pipeline.__module__ == "af3_binder_filter.orchestration.pipeline"


def test_deprecated_workflow_module_is_absent() -> None:
    assert importlib.util.find_spec("af3_binder_filter.workflow") is None


def test_production_orchestration_does_not_import_private_cross_module_symbols() -> None:
    root = Path(__file__).resolve().parents[1] / "src" / "af3_binder_filter" / "orchestration"
    violations: list[str] = []
    for source in sorted(root.glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if not (
                isinstance(node.module, str)
                and node.module.startswith("af3_binder_filter.orchestration")
            ):
                continue
            private_names = [alias.name for alias in node.names if alias.name.startswith("_")]
            if private_names:
                violations.append(f"{source.name}: {', '.join(private_names)}")
    assert not violations, "\n".join(violations)
