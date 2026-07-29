#!/usr/bin/env python3
"""Validate or materialize the repository-tracked runtime source lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from af3_binder_filter.backends import (  # noqa: E402
    runtime_dependency_lock_sha256,
    runtime_recipe_sha256,
)
from af3_binder_filter.runtime_sources import (  # noqa: E402
    RuntimeSourceLock,
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


def _metadata(lock: RuntimeSourceLock) -> dict[str, object]:
    dockerfile = ROOT / "docker" / "runtime" / "Dockerfile"
    lock_root = dockerfile.parent / "locks"
    recipe_sha256 = runtime_recipe_sha256(dockerfile)
    uv_recipe_sha256 = runtime_recipe_sha256(dockerfile, "uv")
    conda_recipe_sha256 = runtime_recipe_sha256(dockerfile, "conda")
    shared_recipe_sha256 = runtime_recipe_sha256(dockerfile, "shared")
    dependency_lock_sha256 = runtime_dependency_lock_sha256(lock_root)
    uv_dependency_sha256 = runtime_dependency_lock_sha256(lock_root, "uv")
    conda_dependency_sha256 = runtime_dependency_lock_sha256(lock_root, "conda")
    return {
        "schema": lock.schema,
        "lock_path": str(lock.path),
        "lock_sha256": lock.sha256,
        "uv_sha256": lock.component_sha256("uv"),
        "conda_sha256": lock.component_sha256("conda"),
        "shared_sha256": lock.shared_sha256(),
        "recipe_sha256": recipe_sha256,
        "uv_recipe_sha256": uv_recipe_sha256,
        "conda_recipe_sha256": conda_recipe_sha256,
        "shared_recipe_sha256": shared_recipe_sha256,
        "dependency_lock_sha256": dependency_lock_sha256,
        "build": dict(lock.build),
        "artifacts": {name: dict(artifact) for name, artifact in lock.artifacts.items()},
        "uv_dependency_lock_sha256": uv_dependency_sha256,
        "conda_dependency_lock_sha256": conda_dependency_sha256,
        "uv_build_sha256": lock.build_sha256(
            "uv",
            recipe_sha256=uv_recipe_sha256,
            dependency_lock_sha256=uv_dependency_sha256,
        ),
        "conda_build_sha256": lock.build_sha256(
            "conda",
            recipe_sha256=conda_recipe_sha256,
            dependency_lock_sha256=conda_dependency_sha256,
        ),
        "runtime_base_build_sha256": lock.build_sha256(
            "shared",
            recipe_sha256=shared_recipe_sha256,
        ),
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
            output_names = (
                "lock_sha256",
                "recipe_sha256",
                "uv_recipe_sha256",
                "conda_recipe_sha256",
                "shared_recipe_sha256",
                "dependency_lock_sha256",
                "uv_sha256",
                "conda_sha256",
                "shared_sha256",
                "uv_build_sha256",
                "conda_build_sha256",
                "runtime_base_build_sha256",
            )
            lines = [f"{name}={payload[name]}" for name in output_names]
            with args.github_output.open("a", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
    except (OSError, RuntimeSourceLockError) as exc:
        print(f"runtime source lock error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
