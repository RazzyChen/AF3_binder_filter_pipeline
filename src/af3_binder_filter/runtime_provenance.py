"""Docker runtime image inspection and release-provenance validation."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Any, Callable


class RuntimeImageError(RuntimeError):
    """Raised when a runtime image cannot be inspected or trusted."""


Runner = Callable[..., subprocess.CompletedProcess[str]]

PROVENANCE_SHA256_LABELS = (
    "org.opencontainers.image.runtime-lock.sha256",
    "org.aerith.runtime.recipe.sha256",
    "org.aerith.runtime.source-lock.sha256",
    "org.aerith.runtime.source-bundle.sha256",
    "org.aerith.runtime.component.uv.sha256",
    "org.aerith.runtime.component.conda.sha256",
    "org.aerith.runtime.source.af3.sha256",
    "org.aerith.runtime.source.protenix.sha256",
    "org.aerith.runtime.source.opendde.sha256",
    "org.aerith.runtime.source.esm.sha256",
    "org.aerith.runtime.source.openfold.sha256",
)
RUNTIME_SOURCE_DIRTY_LABEL = "org.aerith.runtime.source.dirty"
UBUNTU_SNAPSHOT_LABEL = "org.aerith.runtime.ubuntu-snapshot"


def parse_image_inspect(stdout: str, image: str) -> dict[str, Any]:
    """Parse the one-image JSON payload produced by ``docker image inspect``."""

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeImageError(f"docker image inspect returned invalid JSON for {image}") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise RuntimeImageError(f"docker image inspect returned an unexpected payload for {image}")
    image_id = payload[0].get("Id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise RuntimeImageError(f"docker image inspect returned an invalid image ID: {image_id!r}")
    return payload[0]


def inspect_image(
    image: str,
    *,
    docker_bin: str = "docker",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Inspect one local Docker image and return its normalized raw payload."""

    completed = runner(
        [docker_bin, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout).strip()
        raise RuntimeImageError(f"docker image inspect failed for {image}: {error}")
    return parse_image_inspect(completed.stdout, image)


def validate_release_provenance(inspect: dict[str, Any]) -> None:
    """Require complete, clean, immutable provenance labels for release use."""

    config = inspect.get("Config") if isinstance(inspect.get("Config"), dict) else {}
    labels = config.get("Labels") if isinstance(config.get("Labels"), dict) else {}
    invalid = [
        name
        for name in PROVENANCE_SHA256_LABELS
        if re.fullmatch(r"[0-9a-f]{64}", str(labels.get(name, ""))) is None
    ]
    if invalid:
        raise RuntimeImageError(
            "image is missing release-grade SHA256 provenance labels: " + ", ".join(invalid)
        )
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", str(labels.get(UBUNTU_SNAPSHOT_LABEL, ""))) is None:
        raise RuntimeImageError("image is missing a valid Ubuntu snapshot provenance label")
    if labels.get(RUNTIME_SOURCE_DIRTY_LABEL) != "false":
        raise RuntimeImageError("image is not release-grade because source provenance is not clean")
