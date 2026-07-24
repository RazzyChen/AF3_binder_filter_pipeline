from __future__ import annotations

import hashlib
import io
import json
import runpy
import subprocess
from pathlib import Path
from typing import Any

import pytest

from af3_binder_filter.backends import (
    BackendError,
    build_runtime_image_command,
    create_runtime_source_bundle,
    verify_runtime_source_bundle,
)
from af3_binder_filter.config import AerithConfig


def _runtime_sources(tmp_path: Path) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for name in ("af3", "protenix", "opendde", "esm"):
        source = tmp_path / "upstream" / name
        source.mkdir(parents=True)
        (source / "package.py").write_text(f"NAME = {name!r}\n")
        (source / ".venv").mkdir()
        (source / ".venv" / "must-not-copy").write_text("environment")
        sources[name] = source
    af3_common = sources["af3"] / "src" / "alphafold3" / "common"
    af3_common.mkdir(parents=True)
    (af3_common / "resources.py").write_text("RESOURCE = True\n")
    opendde_common = sources["opendde"] / "common"
    opendde_common.mkdir()
    (opendde_common / "components.cif").write_text("mounted separately")
    return sources


def _runtime_config(tmp_path: Path, sources: dict[str, Path]) -> AerithConfig:
    config = AerithConfig()
    config.project.work_dir = str(tmp_path / "work")
    config.runtime.af3_source_dir = str(sources["af3"])
    config.runtime.protenix_source_dir = str(sources["protenix"])
    config.runtime.opendde_source_dir = str(sources["opendde"])
    config.runtime.esm_source_dir = str(sources["esm"])
    config.runtime.opendde_source_commit = "deadbeef"
    config.runtime.esm_source_commit = "deadbeef"
    config.runtime.minimum_build_free_gib = 0
    config.runtime.dockerfile = str(Path(__file__).parents[1] / "docker" / "runtime" / "Dockerfile")
    return config


def test_source_bundle_is_filtered_hashed_and_does_not_touch_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _runtime_sources(tmp_path)
    config = _runtime_config(tmp_path, sources)
    monkeypatch.setattr(
        "af3_binder_filter.backends._git_head",
        lambda _source: "deadbeef",
    )

    bundle = create_runtime_source_bundle(
        config,
        tmp_path / "data" / "runtime-sources",
    )

    assert (sources["opendde"] / "common" / "components.cif").is_file()
    assert (sources["af3"] / ".venv" / "must-not-copy").is_file()
    assert not (bundle.root / "opendde-src" / "common").exists()
    assert not (bundle.root / "af3-src" / ".venv").exists()
    assert (bundle.root / "af3-src" / "src" / "alphafold3" / "common" / "resources.py").is_file()
    manifest = json.loads((bundle.root / "manifest.json").read_text())
    assert manifest["bundle_sha256"] == bundle.bundle_sha256
    assert set(manifest["contexts"]) == {
        "af3-src",
        "protenix-src",
        "opendde-src",
        "esm-src",
    }
    assert verify_runtime_source_bundle(bundle.root).bundle_sha256 == bundle.bundle_sha256

    (bundle.root / "esm-src" / "package.py").write_text("tampered = True\n")
    with pytest.raises(BackendError, match="esm-src sha256 mismatch"):
        verify_runtime_source_bundle(bundle.root)


def test_source_bundle_rejects_dirty_git_trees_without_explicit_opt_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _runtime_sources(tmp_path)
    config = _runtime_config(tmp_path, sources)
    monkeypatch.setattr(
        "af3_binder_filter.backends._git_head",
        lambda _source: "deadbeef",
    )
    monkeypatch.setattr(
        "af3_binder_filter.backends._git_worktree_status",
        lambda source: "M package.py" if source == sources["opendde"] else "",
    )

    with pytest.raises(BackendError, match="source trees are dirty: opendde-src"):
        create_runtime_source_bundle(config, tmp_path / "data" / "dirty-bundle")

    config.runtime.allow_dirty_source_trees = True
    bundle = create_runtime_source_bundle(config, tmp_path / "data" / "dirty-bundle")
    source_state = bundle.manifest["contexts"]["opendde-src"]
    assert source_state["source_git_clean"] is False
    assert source_state["source_git_status"] == "M package.py"


def test_direct_build_command_rejects_dirty_git_trees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _runtime_sources(tmp_path)
    config = _runtime_config(tmp_path, sources)
    monkeypatch.setattr(
        "af3_binder_filter.backends._git_head",
        lambda _source: "deadbeef",
    )
    monkeypatch.setattr(
        "af3_binder_filter.backends._git_worktree_status",
        lambda source: "M package.py" if source == sources["esm"] else "",
    )

    with pytest.raises(BackendError, match="source trees are dirty: esm-src"):
        build_runtime_image_command(config)


def test_build_command_verifies_bundle_and_records_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _runtime_sources(tmp_path)
    config = _runtime_config(tmp_path, sources)
    monkeypatch.setattr(
        "af3_binder_filter.backends._git_head",
        lambda _source: "deadbeef",
    )
    bundle = create_runtime_source_bundle(config, tmp_path / "data" / "bundle")

    command = build_runtime_image_command(config, source_bundle=bundle.root)
    joined = " ".join(command)
    assert f"RUNTIME_SOURCE_BUNDLE_SHA256={bundle.bundle_sha256}" in command
    assert f"AF3_SOURCE_SHA256={bundle.context_sha256['af3-src']}" in command
    assert f"PROTENIX_SOURCE_SHA256={bundle.context_sha256['protenix-src']}" in command
    assert f"OPENDDE_SOURCE_SHA256={bundle.context_sha256['opendde-src']}" in command
    assert f"ESM_SOURCE_SHA256={bundle.context_sha256['esm-src']}" in command
    for name in bundle.context_paths:
        assert f"{name}={bundle.context_paths[name]}" in command
    assert "RUNTIME_RECIPE_SHA256=" in joined

    (bundle.root / "protenix-src" / "package.py").write_text("changed = True\n")
    with pytest.raises(BackendError, match="protenix-src sha256 mismatch"):
        build_runtime_image_command(config, source_bundle=bundle.root)


def test_build_command_accepts_candidate_tag_and_persistent_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _runtime_sources(tmp_path)
    config = _runtime_config(tmp_path, sources)
    config.backend.image = "aerith/fold-runtime:ci-deadbeef"
    monkeypatch.setattr(
        "af3_binder_filter.backends._git_head",
        lambda _source: "deadbeef",
    )
    bundle = create_runtime_source_bundle(config, tmp_path / "data" / "bundle")
    cache = tmp_path / "buildkit-cache"

    command = build_runtime_image_command(
        config,
        source_bundle=bundle.root,
        build_cache_dir=cache,
    )

    assert "--cache-from" not in command
    assert ["--cache-to", f"type=local,dest={cache.resolve()},mode=max"] == command[
        command.index("--cache-to") : command.index("--cache-to") + 2
    ]
    assert command[command.index("--tag") + 1] == "aerith/fold-runtime:ci-deadbeef"

    cache.mkdir()
    (cache / "index.json").write_text("{}")
    warmed_command = build_runtime_image_command(
        config,
        source_bundle=bundle.root,
        build_cache_dir=cache,
    )
    assert ["--cache-from", f"type=local,src={cache.resolve()}"] == warmed_command[
        warmed_command.index("--cache-from") : warmed_command.index("--cache-from") + 2
    ]


def test_runtime_dockerfile_keeps_build_tools_out_of_the_final_image() -> None:
    dockerfile = (Path(__file__).parents[1] / "docker" / "runtime" / "Dockerfile").read_text()
    builder, runtime = dockerfile.split("FROM ${CUDA_BASE} AS runtime", maxsplit=1)

    assert dockerfile.startswith("ARG CUDA_BASE=")
    assert "FROM ${CUDA_BASE} AS builder" in builder
    assert "cuda-nvcc-12-6" in builder
    assert "LAYERNORM_TYPE=fast_layernorm" in builder
    assert "from opendde.model.layer_norm import layer_norm" in builder
    for path in (
        "/hmmer",
        "/opt/conda",
        "/opt/uv-python",
        "/opt/envs",
        "/opt/apps",
        "/opt/mmseqs",
        "/opt/foldseek",
        "/opt/aerith",
    ):
        assert f"COPY --from=builder {path} {path}" in runtime

    assert "cuda-nvcc-12-6" not in runtime
    assert "rm -f /opt/conda/envs/esm/bin/nvcc" in runtime
    assert "test ! -e /usr/local/cuda-12.6/bin/nvcc" in runtime
    assert "find /opt/envs/af3 -type f -name ptxas" in runtime
    assert 'ENTRYPOINT ["/usr/local/bin/fold-runtime"]' in runtime

    entrypoint = (Path(__file__).parents[1] / "docker" / "runtime" / "entrypoint.sh").read_text()
    assert "fast_layer_norm_cuda_v2 is not None" in entrypoint


_EXPORTER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "export_runtime_image.py"))
export_commands = _EXPORTER["export_commands"]
export_runtime_image = _EXPORTER["export_runtime_image"]
immutable_tag_for_image = _EXPORTER["immutable_tag_for_image"]
repository_from_reference = _EXPORTER["repository_from_reference"]
ImageExportError = _EXPORTER["ImageExportError"]


def test_export_plan_uses_docker_save_and_zstd() -> None:
    image_id = "sha256:" + "a" * 64
    repository = repository_from_reference("registry:5000/aerith/fold-runtime:local")
    immutable = immutable_tag_for_image(repository, image_id)
    save, compress = export_commands(immutable, compression_level=7)

    assert repository == "registry:5000/aerith/fold-runtime"
    assert immutable.endswith("sha256-" + "a" * 64)
    assert save == ["docker", "image", "save", immutable]
    assert compress == [
        "zstd",
        "--threads=0",
        "--stdout",
        "--no-progress",
        "-7",
    ]


def test_production_export_rejects_missing_sha256_provenance(tmp_path: Path) -> None:
    payload = {"Id": "sha256:" + "c" * 64, "Config": {"Labels": {}}}

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps([payload]), "")

    with pytest.raises(ImageExportError, match="release-grade SHA256 provenance"):
        export_runtime_image(
            "aerith/fold-runtime:local",
            tmp_path,
            runner=runner,
        )


def test_export_writes_checksum_metadata_and_content_derived_tag(
    tmp_path: Path,
) -> None:
    image_id = "sha256:" + "b" * 64
    provenance = {
        "org.opencontainers.image.runtime-lock.sha256": "1" * 64,
        "org.aerith.runtime.recipe.sha256": "2" * 64,
        "org.aerith.runtime.source-bundle.sha256": "3" * 64,
        "org.aerith.runtime.source.af3.sha256": "4" * 64,
        "org.aerith.runtime.source.protenix.sha256": "5" * 64,
        "org.aerith.runtime.source.opendde.sha256": "6" * 64,
        "org.aerith.runtime.source.esm.sha256": "7" * 64,
    }
    inspect_payload = {
        "Id": image_id,
        "Created": "2026-07-23T00:00:00Z",
        "RepoTags": ["aerith/fold-runtime:local"],
        "RepoDigests": [],
        "Architecture": "amd64",
        "Os": "linux",
        "Size": 123,
        "RootFS": {"Layers": ["sha256:layer"]},
        "Config": {
            "Labels": provenance,
            "Entrypoint": ["/usr/local/bin/fold-runtime"],
            "Cmd": [],
        },
    }
    calls: list[list[str]] = []

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            if command[-1].endswith("sha256-" + "b" * 64) and len(calls) == 2:
                return subprocess.CompletedProcess(command, 1, "", "No such image")
            return subprocess.CompletedProcess(command, 0, json.dumps([inspect_payload]), "")
        if command[:3] == ["docker", "image", "tag"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(command)

    process_calls: list[list[str]] = []

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs: Any):
            process_calls.append(command)
            self.returncode = 0
            if command[:3] == ["docker", "image", "save"]:
                self.stdout = io.BytesIO(b"uncompressed docker archive")
            else:
                self.stdout = None
                kwargs["stdout"].write(b"compressed docker archive")

        def communicate(self) -> tuple[None, bytes]:
            return None, b""

        def wait(self) -> int:
            return 0

    result = export_runtime_image(
        "aerith/fold-runtime:local",
        tmp_path,
        runner=runner,
        popen_factory=FakeProcess,
    )

    expected_bytes = b"compressed docker archive"
    assert result.archive.read_bytes() == expected_bytes
    assert result.archive_sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert result.checksum_file.read_text() == (f"{result.archive_sha256}  {result.archive.name}\n")
    metadata = json.loads(result.metadata_file.read_text())
    assert metadata["archive_format"] == "docker-image-save+zstd"
    assert metadata["immutable_tag"] == result.immutable_tag
    assert metadata["image_inspect"]["labels"] == provenance
    assert process_calls[0] == ["docker", "image", "save", result.immutable_tag]
    assert process_calls[1][0] == "zstd"
