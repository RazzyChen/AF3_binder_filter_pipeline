#!/usr/bin/env python3
"""Build the unified runtime image from a verified source bundle."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from af3_binder_filter.backends import BackendError, build_runtime_image_command  # noqa: E402
from af3_binder_filter.config import ConfigError, compose_hydra_config  # noqa: E402
from af3_binder_filter.runtime_sources import (  # noqa: E402
    RuntimeSourceLockError,
    apply_source_lock_to_config,
    load_runtime_source_lock,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--image", help="candidate image reference to tag")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="persistent BuildKit local cache directory",
    )
    parser.add_argument(
        "--builder",
        help="Buildx builder name for this build",
    )
    parser.add_argument(
        "--target",
        choices=("runtime-base", "uv-component", "conda-component", "fold-runtime"),
        help="optional staged Docker target; default builds the final runtime",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="push the selected target instead of loading it into the local engine",
    )
    parser.add_argument(
        "--registry-cache",
        help="OCI registry cache reference for remote BuildKit cache import/export",
    )
    parser.add_argument(
        "--platform",
        help="explicit OCI target platform, for example linux/amd64",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config, _resolved = compose_hydra_config(
            args.config,
            overrides=args.override,
        )
        dockerfile = Path(config.runtime.dockerfile).expanduser().resolve()
        source_lock_path = Path(config.runtime.source_lock).expanduser()
        if not source_lock_path.is_absolute():
            source_lock_path = dockerfile.parents[2] / source_lock_path
        apply_source_lock_to_config(config, load_runtime_source_lock(source_lock_path))
        if args.image:
            config.backend.image = args.image
        if args.cache_dir:
            args.cache_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        command = build_runtime_image_command(
            config,
            source_bundle=args.source_bundle,
            build_cache_dir=args.cache_dir,
            buildx_builder=args.builder,
            target=args.target,
            push=args.push,
            registry_cache_ref=args.registry_cache,
            build_platform=args.platform,
        )
        print(shlex.join(command))
        if args.dry_run:
            return 0
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise BackendError(
                f"runtime image build failed with return code {completed.returncode}"
            )
    except (
        BackendError,
        ConfigError,
        OSError,
        RuntimeSourceLockError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"runtime image build error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
