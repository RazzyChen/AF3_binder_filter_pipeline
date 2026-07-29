#!/usr/bin/env python3
"""Validate or materialize the repository-tracked runtime source lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from af3_binder_filter.runtime_sources import (  # noqa: E402
    RUNTIME_COMPONENTS,
    RuntimeSourceLockError,
    load_runtime_source_lock,
    prepare_locked_runtime_source_bundle,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=ROOT / "docker" / "runtime" / "sources.lock.yaml",
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("validate", help="validate schema, pins, checksums, and patches")
    metadata = subcommands.add_parser("metadata", help="print lock and component identities")
    metadata.add_argument("--github-output", type=Path)
    prepare = subcommands.add_parser("prepare", help="fetch and verify a locked source bundle")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--force", action="store_true")
    return parser


def _metadata(lock: object) -> dict[str, object]:
    return {
        "schema": lock.schema,
        "lock_path": str(lock.path),
        "lock_sha256": lock.sha256,
        "uv_sha256": lock.component_sha256("uv"),
        "conda_sha256": lock.component_sha256("conda"),
        "sources": {
            name: {
                "component": source.component,
                "context": source.context,
                "repository": source.repository,
                "ref": source.ref,
                "commit": source.commit,
                "tree_sha256": source.tree_sha256,
            }
            for name, source in lock.sources.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        lock = load_runtime_source_lock(args.lock)
        payload = _metadata(lock)
        if args.command == "prepare":
            bundle = prepare_locked_runtime_source_bundle(lock, args.output, force=args.force)
            payload.update(
                {
                    "bundle": str(bundle.root),
                    "bundle_sha256": bundle.bundle_sha256,
                    "context_sha256": bundle.context_sha256,
                }
            )
        if args.command == "metadata" and args.github_output is not None:
            lines = [f"source_lock_sha256={lock.sha256}"]
            lines.extend(
                f"{component}_sha256={lock.component_sha256(component)}"
                for component in RUNTIME_COMPONENTS
            )
            with args.github_output.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
    except (OSError, RuntimeSourceLockError) as exc:
        print(f"runtime source lock error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
