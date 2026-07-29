"""Immutable Git source locks and release source-bundle preparation."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

import yaml

from af3_binder_filter.backends import (
    RUNTIME_SOURCE_CONTEXTS,
    RuntimeSourceBundle,
    create_runtime_source_bundle,
    verify_runtime_source_bundle,
)
from af3_binder_filter.config import AerithConfig
from af3_binder_filter.io_utils import atomic_write_json

RUNTIME_SOURCE_LOCK_SCHEMA = "aerith.runtime-sources.v1"
RUNTIME_SOURCE_NAMES = ("af3", "opendde", "protenix", "esm", "openfold")
RUNTIME_COMPONENTS = ("uv", "conda")
_SHA1_PATTERN = re.compile(r"[0-9a-f]{40}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class RuntimeSourceLockError(RuntimeError):
    """Raised when a runtime source lock or locked checkout is invalid."""


@dataclass(frozen=True, slots=True)
class LockedPatch:
    path: str
    sha256: str
    absolute_path: Path


@dataclass(frozen=True, slots=True)
class LockedSource:
    name: str
    component: str
    context: str
    repository: str
    ref: str
    commit: str
    tree_sha256: str
    patches: tuple[LockedPatch, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSourceLock:
    path: Path
    sha256: str
    schema: str
    build: Mapping[str, str]
    sources: Mapping[str, LockedSource]
    artifacts: Mapping[str, Mapping[str, str]]

    def component_sha256(self, component: str) -> str:
        """Return the deterministic recipe identity for one environment group."""

        if component not in RUNTIME_COMPONENTS:
            raise RuntimeSourceLockError(f"unsupported runtime component: {component}")
        identity = {
            "schema": self.schema,
            "build": dict(self.build),
            "sources": {
                name: _source_identity(source)
                for name, source in self.sources.items()
                if source.component == component
            },
            "artifacts": {
                name: dict(artifact)
                for name, artifact in self.artifacts.items()
                if artifact.get("component") == component
                # Both environment builders invoke uv. The conda component
                # therefore depends on the uv binary even though it does not
                # contain the AF3/OpenDDE uv environments.
                or (component == "conda" and name == "uv")
            },
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def shared_sha256(self) -> str:
        """Return the source identity of the common runtime tool layer."""

        identity = {
            "schema": self.schema,
            "build": dict(self.build),
            # tool-builder consumes AF3's jackhmmer patch.
            "af3": _source_identity(self.sources["af3"]),
            "artifacts": {
                name: dict(artifact)
                for name, artifact in self.artifacts.items()
                if artifact.get("component") == "shared"
            },
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def build_sha256(
        self,
        component: str,
        *,
        recipe_sha256: str,
        dependency_lock_sha256: str | None = None,
    ) -> str:
        """Combine source, recipe, and dependency identities for an OCI target."""

        if component in RUNTIME_COMPONENTS:
            source_sha256 = self.component_sha256(component)
            if dependency_lock_sha256 is None:
                raise RuntimeSourceLockError(
                    f"{component} build identity requires a dependency lock digest"
                )
        elif component == "shared":
            source_sha256 = self.shared_sha256()
            if dependency_lock_sha256 is not None:
                raise RuntimeSourceLockError(
                    "shared build identity must not include Python dependency locks"
                )
        else:
            raise RuntimeSourceLockError(f"unsupported runtime build component: {component}")
        for label, digest in (
            ("recipe_sha256", recipe_sha256),
            ("dependency_lock_sha256", dependency_lock_sha256),
        ):
            if digest is not None and _SHA256_PATTERN.fullmatch(digest) is None:
                raise RuntimeSourceLockError(f"{label} must be a SHA-256 digest")
        identity = {
            "schema": "aerith.runtime-component-build.v1",
            "component": component,
            "source_sha256": source_sha256,
            "recipe_sha256": recipe_sha256,
            "dependency_lock_sha256": dependency_lock_sha256,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _source_identity(source: LockedSource) -> dict[str, Any]:
    return {
        "component": source.component,
        "context": source.context,
        "repository": source.repository,
        "ref": source.ref,
        "commit": source.commit,
        "tree_sha256": source.tree_sha256,
        "patches": [{"path": patch.path, "sha256": patch.sha256} for patch in source.patches],
    }


def _mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise RuntimeSourceLockError(f"{label} must be a YAML mapping")
    return value


def _text(mapping: Mapping[str, Any], key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeSourceLockError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def _repository_root(lock_path: Path) -> Path:
    for candidate in (lock_path.parent, *lock_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return lock_path.parent


def _validate_github_repository(repository: str, label: str) -> None:
    if repository.startswith("git@github.com:"):
        if repository.count("@") != 1:
            raise RuntimeSourceLockError(f"{label}.repository contains embedded credentials")
        return
    parsed = urlparse(repository)
    if parsed.scheme not in {"https", "ssh"} or parsed.hostname != "github.com":
        raise RuntimeSourceLockError(f"{label}.repository must be a github.com Git URL")
    if parsed.username not in {None, "git"} or parsed.password is not None:
        raise RuntimeSourceLockError(f"{label}.repository must not contain credentials")


def load_runtime_source_lock(path: Path) -> RuntimeSourceLock:
    """Load and fully validate the repository-tracked runtime source lock."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeSourceLockError(f"runtime source lock does not exist: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise RuntimeSourceLockError(f"invalid runtime source lock YAML: {exc}") from exc
    root = _mapping(payload, "runtime source lock")
    schema = _text(root, "schema", "runtime source lock")
    if schema != RUNTIME_SOURCE_LOCK_SCHEMA:
        raise RuntimeSourceLockError(f"unsupported runtime source lock schema: {schema}")
    build_raw = _mapping(root.get("build"), "runtime source lock.build")
    build = {
        "ubuntu_snapshot": _text(build_raw, "ubuntu_snapshot", "build"),
        "cuda_base": _text(build_raw, "cuda_base", "build"),
    }
    if re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", build["ubuntu_snapshot"]) is None:
        raise RuntimeSourceLockError("build.ubuntu_snapshot must use YYYYMMDDTHHMMSSZ")
    if "@sha256:" not in build["cuda_base"]:
        raise RuntimeSourceLockError("build.cuda_base must include an immutable sha256 digest")

    sources_raw = _mapping(root.get("sources"), "runtime source lock.sources")
    if set(sources_raw) != set(RUNTIME_SOURCE_NAMES):
        raise RuntimeSourceLockError(
            "runtime source lock must define exactly: " + ", ".join(RUNTIME_SOURCE_NAMES)
        )
    repository_root = _repository_root(resolved)
    sources: dict[str, LockedSource] = {}
    contexts: set[str] = set()
    for name in RUNTIME_SOURCE_NAMES:
        label = f"sources.{name}"
        source_raw = _mapping(sources_raw[name], label)
        component = _text(source_raw, "component", label)
        if component not in RUNTIME_COMPONENTS:
            raise RuntimeSourceLockError(f"{label}.component must be uv or conda")
        context = _text(source_raw, "context", label)
        if context not in RUNTIME_SOURCE_CONTEXTS:
            raise RuntimeSourceLockError(f"{label}.context is not a known BuildKit context")
        if context in contexts:
            raise RuntimeSourceLockError(f"duplicate BuildKit source context: {context}")
        contexts.add(context)
        repository = _text(source_raw, "repository", label)
        _validate_github_repository(repository, label)
        commit = _text(source_raw, "commit", label)
        tree_sha256 = _text(source_raw, "tree_sha256", label)
        if _SHA1_PATTERN.fullmatch(commit) is None:
            raise RuntimeSourceLockError(f"{label}.commit must be a full 40-character SHA")
        if _SHA256_PATTERN.fullmatch(tree_sha256) is None:
            raise RuntimeSourceLockError(f"{label}.tree_sha256 must be a SHA-256 digest")
        patches_raw = source_raw.get("patches", [])
        if not isinstance(patches_raw, list):
            raise RuntimeSourceLockError(f"{label}.patches must be a list")
        patches: list[LockedPatch] = []
        for index, item in enumerate(patches_raw):
            patch_label = f"{label}.patches[{index}]"
            patch_raw = _mapping(item, patch_label)
            patch_path = _text(patch_raw, "path", patch_label)
            patch_sha256 = _text(patch_raw, "sha256", patch_label)
            if _SHA256_PATTERN.fullmatch(patch_sha256) is None:
                raise RuntimeSourceLockError(f"{patch_label}.sha256 must be a SHA-256 digest")
            absolute = (repository_root / patch_path).resolve()
            if not absolute.is_relative_to(repository_root):
                raise RuntimeSourceLockError(f"{patch_label}.path escapes the repository")
            if not absolute.is_file():
                raise RuntimeSourceLockError(f"locked patch does not exist: {absolute}")
            actual_patch_sha256 = hashlib.sha256(absolute.read_bytes()).hexdigest()
            if actual_patch_sha256 != patch_sha256:
                raise RuntimeSourceLockError(
                    f"{patch_label}.sha256 mismatch: expected {patch_sha256}, "
                    f"found {actual_patch_sha256}"
                )
            patches.append(LockedPatch(patch_path, patch_sha256, absolute))
        sources[name] = LockedSource(
            name=name,
            component=component,
            context=context,
            repository=repository,
            ref=_text(source_raw, "ref", label),
            commit=commit,
            tree_sha256=tree_sha256,
            patches=tuple(patches),
        )
    if contexts != set(RUNTIME_SOURCE_CONTEXTS):
        raise RuntimeSourceLockError("every BuildKit source context must be owned by one source")

    artifacts_raw = _mapping(root.get("artifacts"), "runtime source lock.artifacts")
    artifacts: dict[str, dict[str, str]] = {}
    for name, value in artifacts_raw.items():
        label = f"artifacts.{name}"
        artifact_raw = _mapping(value, label)
        component = _text(artifact_raw, "component", label)
        if component not in {*RUNTIME_COMPONENTS, "shared"}:
            raise RuntimeSourceLockError(f"{label}.component must be uv, conda, or shared")
        sha256 = _text(artifact_raw, "sha256", label)
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise RuntimeSourceLockError(f"{label}.sha256 must be a SHA-256 digest")
        artifacts[name] = {
            key: str(item)
            for key, item in artifact_raw.items()
            if isinstance(key, str) and isinstance(item, (str, int))
        }
    return RuntimeSourceLock(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        schema=schema,
        build=build,
        sources=sources,
        artifacts=artifacts,
    )


def git_tree_sha256(source: Path, revision: str = "HEAD") -> str:
    """Hash the canonical recursive Git tree listing for one revision."""

    completed = subprocess.run(
        ["git", "-C", str(source), "ls-tree", "-r", "-z", revision],
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        error = completed.stderr.decode(errors="replace").strip()
        raise RuntimeSourceLockError(f"git ls-tree failed for {source}: {error}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _run_git(arguments: list[str], *, cwd: Path | None = None) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeSourceLockError(f"git {' '.join(arguments)} failed: {detail}")


def _checkout_locked_source(source: LockedSource, destination: Path) -> None:
    destination.mkdir(parents=True)
    _run_git(["init", "--quiet", str(destination)])
    _run_git(["-C", str(destination), "remote", "add", "origin", source.repository])
    _run_git(
        [
            "-C",
            str(destination),
            "fetch",
            "--depth=1",
            "--filter=blob:none",
            "origin",
            source.commit,
        ]
    )
    sparse_patterns = [
        "/*",
        "!/.github/",
        "!/**/.venv/",
        "!/**/__pycache__/",
        "!/**/checkpoint/",
        "!/**/ckpt/",
        "!/**/output/",
        "!/**/outputs/",
        "!/**/test_outputs/",
        "!/**/search_database/",
        "!/**/examples/",
        "!/**/tests/",
        "!/**/test_data/",
        "!/**/docs/",
        "!/**/benchmarks/",
        "!/**/assets/",
        "!/**/build/",
        "!/**/.pytest_cache/",
    ]
    if source.name == "opendde":
        sparse_patterns.append("!/common/")
    if source.name == "protenix":
        sparse_patterns.append("!/scripts/msa/data/")
    _run_git(["-C", str(destination), "sparse-checkout", "set", "--no-cone", *sparse_patterns])
    _run_git(["-C", str(destination), "checkout", "--quiet", "--detach", source.commit])
    completed = subprocess.run(
        ["git", "-C", str(destination), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    actual_commit = completed.stdout.strip() if completed.returncode == 0 else ""
    if actual_commit != source.commit:
        raise RuntimeSourceLockError(
            f"{source.name} commit mismatch: expected {source.commit}, "
            f"found {actual_commit or 'unavailable'}"
        )
    actual_tree = git_tree_sha256(destination, source.commit)
    if actual_tree != source.tree_sha256:
        raise RuntimeSourceLockError(
            f"{source.name} tree_sha256 mismatch: expected {source.tree_sha256}, "
            f"found {actual_tree}"
        )
    for patch in source.patches:
        _run_git(["-C", str(destination), "apply", "--check", str(patch.absolute_path)])
        _run_git(["-C", str(destination), "apply", str(patch.absolute_path)])


def apply_source_lock_to_config(config: AerithConfig, lock: RuntimeSourceLock) -> None:
    """Project locked versions into the existing runtime build configuration."""

    config.runtime.source_lock = str(lock.path)
    for name in RUNTIME_SOURCE_NAMES:
        setattr(config.runtime, f"{name}_source_commit", lock.sources[name].commit)
    mmseqs = lock.artifacts["mmseqs"]
    foldseek = lock.artifacts["foldseek"]
    config.runtime.mmseqs_release = mmseqs["release"]
    config.runtime.mmseqs_version = mmseqs["version"]
    config.runtime.mmseqs_archive_sha256 = mmseqs["sha256"]
    config.runtime.foldseek_release = foldseek["release"]
    config.runtime.foldseek_version = foldseek["version"]
    config.runtime.foldseek_archive_sha256 = foldseek["sha256"]


def prepare_locked_runtime_source_bundle(
    lock: RuntimeSourceLock,
    destination: Path,
    *,
    force: bool = False,
) -> RuntimeSourceBundle:
    """Fetch, verify, patch, and atomically publish the five locked contexts."""

    target = destination.expanduser().resolve()
    if target.exists():
        if not force:
            raise RuntimeSourceLockError(
                f"runtime source bundle already exists: {target}; use force to replace it"
            )
        manifest_path = target / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeSourceLockError(f"refusing to replace non-bundle path: {target}") from exc
        if manifest.get("source_lock_schema") != lock.schema:
            raise RuntimeSourceLockError(f"refusing to replace non-locked bundle: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.locked-", dir=target.parent))
    checkout_root = staging / "checkouts"
    bundle_root = staging / "bundle"
    try:
        checkout_root.mkdir()
        for source in lock.sources.values():
            _checkout_locked_source(source, checkout_root / source.name)
        config = AerithConfig()
        config.runtime.minimum_build_free_gib = 0
        config.runtime.allow_dirty_source_trees = True
        for source in lock.sources.values():
            setattr(config.runtime, f"{source.name}_source_dir", str(checkout_root / source.name))
            setattr(config.runtime, f"{source.name}_source_commit", source.commit)
        bundle = create_runtime_source_bundle(config, bundle_root)
        manifest = dict(bundle.manifest)
        contexts = _mapping(manifest.get("contexts"), "runtime source bundle.contexts")
        for source in lock.sources.values():
            context = _mapping(contexts[source.context], f"contexts.{source.context}")
            context.update(
                {
                    "source_repository": source.repository,
                    "source_ref": source.ref,
                    "source_tree_sha256": source.tree_sha256,
                    "source_git_clean": True,
                    "source_git_status": None,
                    "declared_patches": [
                        {"path": patch.path, "sha256": patch.sha256} for patch in source.patches
                    ],
                }
            )
        manifest["source_lock_schema"] = lock.schema
        manifest["source_lock_sha256"] = lock.sha256
        manifest["source_components"] = {
            component: [
                source.context for source in lock.sources.values() if source.component == component
            ]
            for component in RUNTIME_COMPONENTS
        }
        atomic_write_json(bundle_root / "manifest.json", manifest)
        verify_runtime_source_bundle(bundle_root)
        if target.exists():
            shutil.rmtree(target)
        bundle_root.replace(target)
        return verify_runtime_source_bundle(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
