"""Factories only for runtime components with interchangeable implementations."""

from __future__ import annotations

from af3_binder_filter.config import InterfaceSettings
from af3_binder_filter.rosetta import (
    InterfaceEnergyEngine,
    RosettaCliEngine,
)


class ComponentFactoryError(ValueError):
    """Raised when a validated component selector has no implementation."""


def create_interface_energy_engine(
    settings: InterfaceSettings,
) -> InterfaceEnergyEngine | None:
    """Build the configured energy adapter without hiding stage control flow."""

    if settings.energy_engine == "none":
        return None
    if settings.energy_engine == "rosetta_cli":
        return RosettaCliEngine(settings.rosetta)
    raise ComponentFactoryError(
        f"unsupported interface energy engine: {settings.energy_engine}"
    )
