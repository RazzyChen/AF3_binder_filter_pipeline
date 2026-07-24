"""Explicit metadata registry for the ten production pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from af3_binder_filter.config import AerithConfig
from af3_binder_filter.progress import StageSpec

StageCondition = Literal["always", "secondary", "esm"]


@dataclass(frozen=True, slots=True)
class PipelineStageDefinition:
    """Stable stage identity separated from its implementation function."""

    key: str
    title: str
    condition: StageCondition = "always"
    progress_visible: bool = True

    def enabled(self, config: AerithConfig) -> bool:
        if self.condition == "secondary":
            return config.secondary_backend.enabled
        if self.condition == "esm":
            return config.scoring.esm.enabled
        return True

    def progress_spec(self) -> StageSpec:
        return StageSpec(self.key, self.title)


PIPELINE_STAGE_REGISTRY: tuple[PipelineStageDefinition, ...] = (
    PipelineStageDefinition("preflight", "Preflight", progress_visible=False),
    PipelineStageDefinition("features", "MSA/template searching"),
    PipelineStageDefinition("primary_prediction", "Primary prediction"),
    PipelineStageDefinition("primary_interface", "Primary interface analysis"),
    PipelineStageDefinition(
        "secondary_features",
        "Secondary feature adaptation",
        condition="secondary",
    ),
    PipelineStageDefinition(
        "secondary_prediction",
        "Secondary prediction",
        condition="secondary",
    ),
    PipelineStageDefinition(
        "secondary_interface",
        "Secondary interface analysis",
        condition="secondary",
    ),
    PipelineStageDefinition("consensus", "Backend consensus / effective selection"),
    PipelineStageDefinition(
        "esm",
        "Effective ESMFold / ESM-IF scoring",
        condition="esm",
    ),
    PipelineStageDefinition("clustering", "Foldseek / epitope clustering"),
)


def enabled_stage_definitions(
    config: AerithConfig,
) -> tuple[PipelineStageDefinition, ...]:
    """Return enabled stages in their only valid execution order."""

    return tuple(stage for stage in PIPELINE_STAGE_REGISTRY if stage.enabled(config))


def progress_stage_specs(config: AerithConfig) -> tuple[StageSpec, ...]:
    """Return reporter-visible stages without changing the preflight UI."""

    return tuple(
        stage.progress_spec()
        for stage in enabled_stage_definitions(config)
        if stage.progress_visible
    )
