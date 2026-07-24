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
        if args.image:
            config.backend.image = args.image
        if args.cache_dir:
            args.cache_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
        command = build_runtime_image_command(
            config,
            source_bundle=args.source_bundle,
            build_cache_dir=args.cache_dir,
            buildx_builder=args.builder,
        )
        print(shlex.join(command))
        if args.dry_run:
            return 0
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise BackendError(
                f"runtime image build failed with return code {completed.returncode}"
            )
    except (BackendError, ConfigError, OSError, subprocess.SubprocessError) as exc:
        print(f"runtime image build error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
