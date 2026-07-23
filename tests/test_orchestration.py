from __future__ import annotations

import pytest

from af3_binder_filter.config import AerithConfig
from af3_binder_filter.orchestration.factories import (
    ComponentFactoryError,
    create_interface_energy_engine,
)
from af3_binder_filter.orchestration.stage_registry import (
    PIPELINE_STAGE_REGISTRY,
    enabled_stage_definitions,
    progress_stage_specs,
)
from af3_binder_filter.rosetta import RosettaCliEngine
from af3_binder_filter.workflow import run_pipeline


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


def test_workflow_is_a_compatibility_facade() -> None:
    assert run_pipeline.__module__ == "af3_binder_filter.orchestration.pipeline"
