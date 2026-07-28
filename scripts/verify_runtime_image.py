#!/usr/bin/env python3
"""Verify that a local runtime image has release-grade Aerith provenance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(ROOT / "src"))

from export_runtime_image import (  # noqa: E402
    ImageExportError,
    inspect_image,
    validate_release_provenance,
)

Runner = Callable[..., subprocess.CompletedProcess[str]]


def verify_runtime_image(
    image: str,
    *,
    docker_bin: str = "docker",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Inspect one local image and require clean, complete release provenance."""

    inspected = inspect_image(image, docker_bin=docker_bin, runner=runner)
    validate_release_provenance(inspected)
    return inspected


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--docker-bin", default="docker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        inspected = verify_runtime_image(args.image, docker_bin=args.docker_bin)
    except (ImageExportError, OSError, subprocess.SubprocessError) as exc:
        print(f"runtime image verification error: {exc}", file=sys.stderr)
        return 2
    config = inspected.get("Config") if isinstance(inspected.get("Config"), dict) else {}
    labels = config.get("Labels") if isinstance(config.get("Labels"), dict) else {}
    print(
        json.dumps(
            {
                "image": args.image,
                "image_id": inspected["Id"],
                "source_dirty": labels.get("org.aerith.runtime.source.dirty"),
                "status": "ok",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
