from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from af3_binder_filter.backends import runtime_recipe_sha256
from af3_binder_filter.config import AerithConfig
from af3_binder_filter.runtime_sources import (
    RuntimeSourceLockError,
    apply_source_lock_to_config,
    git_tree_sha256,
    load_runtime_source_lock,
)

ROOT = Path(__file__).parents[1]
LOCK = ROOT / "docker" / "runtime" / "sources.lock.yaml"


def test_repository_runtime_source_lock_is_complete_and_component_addressed() -> None:
    lock = load_runtime_source_lock(LOCK)

    assert set(lock.sources) == {"af3", "opendde", "protenix", "esm", "openfold"}
    assert {source.context for source in lock.sources.values()} == {
        "af3-src",
        "opendde-src",
        "protenix-src",
        "esm-src",
        "openfold-src",
    }
    assert {lock.sources[name].component for name in ("af3", "opendde")} == {"uv"}
    assert {lock.sources[name].component for name in ("protenix", "esm", "openfold")} == {"conda"}
    assert re.fullmatch(r"[0-9a-f]{64}", lock.sha256)
    assert lock.component_sha256("uv") != lock.component_sha256("conda")
    changed_artifacts = {name: dict(artifact) for name, artifact in lock.artifacts.items()}
    changed_artifacts["uv"]["version"] = "changed"
    changed = replace(lock, artifacts=changed_artifacts)
    assert changed.component_sha256("uv") != lock.component_sha256("uv")
    assert changed.component_sha256("conda") != lock.component_sha256("conda")
    assert lock.shared_sha256() not in {
        lock.component_sha256("uv"),
        lock.component_sha256("conda"),
    }
    uv_build = lock.build_sha256(
        "uv",
        recipe_sha256="1" * 64,
        dependency_lock_sha256="2" * 64,
    )
    conda_build = lock.build_sha256(
        "conda",
        recipe_sha256="1" * 64,
        dependency_lock_sha256="2" * 64,
    )
    assert uv_build != conda_build
    assert re.fullmatch(r"[0-9a-f]{64}", lock.build_sha256("shared", recipe_sha256="1" * 64))


def test_component_recipe_hashes_ignore_unrelated_stages(tmp_path: Path) -> None:
    dockerfile = tmp_path / "docker" / "runtime" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text(
        "# syntax=docker/dockerfile:1\n"
        "ARG BASE=example\n"
        "FROM base AS build-base\nRUN echo shared\n"
        "FROM build-base AS uv-builder\nRUN echo uv\n"
        "FROM scratch AS uv-component\nCOPY --from=uv-builder /uv /uv\n"
        "FROM build-base AS conda-builder\nRUN echo conda\n"
        "FROM scratch AS conda-component\nCOPY --from=conda-builder /conda /conda\n"
        "FROM build-base AS tool-builder\nRUN echo tools\n"
        "FROM base AS runtime-base\nRUN echo runtime\n"
    )
    feature_builder = tmp_path / "docker" / "feature-builder"
    feature_builder.mkdir()
    for name in ("build_local_features.py", "mmseqs_wrapper.py", "convert_af3_templates.py"):
        (feature_builder / name).write_text(name)
    for name in ("entrypoint.sh", "esm_if_batch.py", "validate_runtime.sh"):
        (dockerfile.parent / name).write_text(name)
    (dockerfile.parent / "Dockerfile.assemble").write_text("FROM base\n")

    uv_before = runtime_recipe_sha256(dockerfile, "uv")
    conda_before = runtime_recipe_sha256(dockerfile, "conda")
    dockerfile.write_text(dockerfile.read_text().replace("echo uv", "echo uv changed"))

    assert runtime_recipe_sha256(dockerfile, "uv") != uv_before
    assert runtime_recipe_sha256(dockerfile, "conda") == conda_before


def test_runtime_source_lock_rejects_patch_checksum_drift(tmp_path: Path) -> None:
    payload = yaml.safe_load(LOCK.read_text())
    repository = tmp_path / "repository"
    patch = repository / "docker" / "runtime" / "patches" / "compat.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text("not the locked patch\n")
    (repository / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    payload["sources"]["esm"]["patches"] = [
        {"path": "docker/runtime/patches/compat.patch", "sha256": "0" * 64}
    ]
    lock_path = repository / "docker" / "runtime" / "sources.lock.yaml"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(RuntimeSourceLockError, match="sha256 mismatch"):
        load_runtime_source_lock(lock_path)


def test_git_tree_sha256_matches_canonical_recursive_listing(tmp_path: Path) -> None:
    repository = tmp_path / "source"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Aerith Test"], check=True)
    subprocess.run(
        ["git", "-C", str(repository), "config", "user.email", "aerith@example.invalid"],
        check=True,
    )
    (repository / "package.py").write_text("VALUE = 1\n")
    subprocess.run(["git", "-C", str(repository), "add", "package.py"], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "--quiet", "-m", "source"], check=True)
    listing = subprocess.run(
        ["git", "-C", str(repository), "ls-tree", "-r", "-z", "HEAD"],
        capture_output=True,
        check=True,
    ).stdout

    assert git_tree_sha256(repository) == hashlib.sha256(listing).hexdigest()


def test_source_lock_projects_versions_into_runtime_config() -> None:
    lock = load_runtime_source_lock(LOCK)
    config = AerithConfig()

    apply_source_lock_to_config(config, lock)

    assert config.runtime.source_lock == str(LOCK.resolve())
    assert config.runtime.af3_source_commit == lock.sources["af3"].commit
    assert config.runtime.protenix_source_commit == lock.sources["protenix"].commit
    assert config.runtime.opendde_source_commit == lock.sources["opendde"].commit
    assert config.runtime.esm_source_commit == lock.sources["esm"].commit
    assert config.runtime.openfold_source_commit == lock.sources["openfold"].commit
    assert config.runtime.mmseqs_archive_sha256 == lock.artifacts["mmseqs"]["sha256"]
    assert config.runtime.foldseek_archive_sha256 == lock.artifacts["foldseek"]["sha256"]
