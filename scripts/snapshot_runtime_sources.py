#!/usr/bin/env python3
"""Create or verify portable, filtered runtime source bundles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from af3_binder_filter.backends import (  # noqa: E402
    BackendError,
    create_runtime_source_bundle,
    verify_runtime_source_bundle,
)
from af3_binder_filter.config import ConfigError, compose_hydra_config  # noqa: E402


def _summary(bundle: object) -> dict[str, object]:
    return {
        "root": str(bundle.root),
        "bundle_sha256": bundle.bundle_sha256,
        "context_sha256": bundle.context_sha256,
        "manifest": str(bundle.root / "manifest.json"),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)

    create = subcommands.add_parser("create", help="copy filtered source trees")
    create.add_argument("--config", type=Path, default=Path("config.yaml"))
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--override", action="append", default=[])
    create.add_argument("--force", action="store_true")

    verify = subcommands.add_parser("verify", help="rehash and verify a bundle")
    verify.add_argument("bundle", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            config, _resolved = compose_hydra_config(
                args.config,
                overrides=args.override,
            )
            bundle = create_runtime_source_bundle(
                config,
                args.output,
                force=args.force,
            )
        else:
            bundle = verify_runtime_source_bundle(args.bundle)
    except (BackendError, ConfigError, OSError) as exc:
        print(f"runtime source bundle error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(_summary(bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
