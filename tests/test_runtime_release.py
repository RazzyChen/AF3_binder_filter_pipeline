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
    for name in ("af3", "protenix", "opendde", "esm", "openfold"):
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
    config.runtime.af3_source_commit = "deadbeef"
    config.runtime.protenix_source_dir = str(sources["protenix"])
    config.runtime.protenix_source_commit = "deadbeef"
    config.runtime.opendde_source_dir = str(sources["opendde"])
    config.runtime.esm_source_dir = str(sources["esm"])
    config.runtime.openfold_source_dir = str(sources["openfold"])
    config.runtime.opendde_source_commit = "deadbeef"
    config.runtime.esm_source_commit = "deadbeef"
    config.runtime.openfold_source_commit = "deadbeef"
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
        "openfold-src",
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

    config.runtime.allow_dirty_source_trees = False
    with pytest.raises(BackendError, match="bundle contains dirty source trees: opendde-src"):
        build_runtime_image_command(config, source_bundle=bundle.root)

    config.runtime.allow_dirty_source_trees = True
    dirty_command = build_runtime_image_command(config, source_bundle=bundle.root)
    assert "RUNTIME_SOURCE_DIRTY=true" in dirty_command


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
    assert "RUNTIME_SOURCE_DIRTY=false" in command
    assert f"AF3_SOURCE_SHA256={bundle.context_sha256['af3-src']}" in command
    assert f"PROTENIX_SOURCE_SHA256={bundle.context_sha256['protenix-src']}" in command
    assert f"OPENDDE_SOURCE_SHA256={bundle.context_sha256['opendde-src']}" in command
    assert f"ESM_SOURCE_SHA256={bundle.context_sha256['esm-src']}" in command
    assert f"OPENFOLD_SOURCE_SHA256={bundle.context_sha256['openfold-src']}" in command
    assert "RUNTIME_SOURCE_LOCK_SHA256=unavailable" in command
    for name in bundle.context_paths:
        assert f"{name}={bundle.context_paths[name]}" in command
    assert "RUNTIME_RECIPE_SHA256=" in joined
    assert "SHARED_COMPONENT_SHA256=" in joined

    config.runtime.opendde_source_commit = "different"
    with pytest.raises(BackendError, match="opendde-src commit mismatch"):
        build_runtime_image_command(config, source_bundle=bundle.root)
    config.runtime.opendde_source_commit = "deadbeef"
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
        buildx_builder="aerith-test-builder",
    )

    assert ["--builder", "aerith-test-builder"] == command[
        command.index("--builder") : command.index("--builder") + 2
    ]
    assert "--load" in command
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
        buildx_builder="aerith-test-builder",
    )
    assert ["--cache-from", f"type=local,src={cache.resolve()}"] == warmed_command[
        warmed_command.index("--cache-from") : warmed_command.index("--cache-from") + 2
    ]


def test_build_command_can_push_one_content_component(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _runtime_sources(tmp_path)
    config = _runtime_config(tmp_path, sources)
    config.backend.image = "ghcr.io/razzychen/aerith-uv-component:test"
    monkeypatch.setattr("af3_binder_filter.backends._git_head", lambda _source: "deadbeef")
    bundle = create_runtime_source_bundle(config, tmp_path / "data" / "bundle")

    command = build_runtime_image_command(
        config,
        source_bundle=bundle.root,
        buildx_builder="remote-aerith",
        target="uv-component",
        push=True,
        registry_cache_ref="ghcr.io/razzychen/aerith-build-cache:uv",
        build_platform="linux/amd64",
    )

    assert "--push" in command
    assert "--load" not in command
    assert command[command.index("--target") + 1] == "uv-component"
    assert command[command.index("--platform") + 1] == "linux/amd64"
    assert command[command.index("--tag") + 1].endswith("aerith-uv-component:test")
    assert "type=registry,ref=ghcr.io/razzychen/aerith-build-cache:uv" in command
    assert "type=registry,ref=ghcr.io/razzychen/aerith-build-cache:uv,mode=max" in command


def test_build_command_rejects_push_without_buildx_builder(tmp_path: Path) -> None:
    sources = _runtime_sources(tmp_path)
    config = _runtime_config(tmp_path, sources)

    with pytest.raises(BackendError, match="requires an explicit Buildx builder"):
        build_runtime_image_command(config, push=True)


def test_runtime_dockerfile_keeps_build_tools_out_of_the_final_image() -> None:
    dockerfile = (Path(__file__).parents[1] / "docker" / "runtime" / "Dockerfile").read_text()
    build_base, tool_builder = dockerfile.split("FROM build-base AS tool-builder", maxsplit=1)
    tool_builder, uv_builder = tool_builder.split("FROM build-base AS uv-builder", maxsplit=1)
    uv_builder, uv_component = uv_builder.split("FROM scratch AS uv-component", maxsplit=1)
    uv_component, conda_builder = uv_component.split("FROM build-base AS conda-builder", maxsplit=1)
    conda_builder, conda_component = conda_builder.split(
        "FROM scratch AS conda-component", maxsplit=1
    )
    conda_component, runtime_base = conda_component.split(
        "FROM ${CUDA_BASE} AS runtime-base", maxsplit=1
    )
    runtime_base, final = runtime_base.split("FROM runtime-base AS fold-runtime", maxsplit=1)

    assert "FROM ${CUDA_BASE} AS build-base" in build_base
    assert "cuda-nvcc-12-6" in build_base
    assert "MMSEQS_RELEASE" not in build_base
    assert "COPY --from=af3-src /docker/jackhmmer_seq_limit.patch" in tool_builder
    assert "MMSEQS_RELEASE" in tool_builder
    assert "COPY --from=af3-src" in uv_builder
    assert "COPY --from=opendde-src" in uv_builder
    assert "protenix-src" not in uv_builder
    assert "COPY --from=uv-builder /opt/envs /opt/envs" in uv_component
    assert "org.aerith.runtime.component.uv.sha256" in uv_component
    assert "COPY --from=protenix-src" in conda_builder
    assert "COPY --from=esm-src" in conda_builder
    assert "COPY --from=openfold-src" in conda_builder
    assert "opendde-src" not in conda_builder
    assert "COPY --from=conda-builder /opt/conda /opt/conda" in conda_component
    assert "org.aerith.runtime.component.conda.sha256" in conda_component
    assert "cuda-nvcc-12-6" not in runtime_base
    assert "COPY --from=tool-builder /opt/mmseqs /opt/mmseqs" in runtime_base
    assert "org.aerith.runtime.component.shared.sha256" in runtime_base
    assert "COPY --from=uv-component /opt/envs /opt/envs" in final
    assert "COPY --from=conda-component /opt/conda /opt/conda" in final
    assert "org.aerith.runtime.component.runtime-base.image-digest" in final
    assert "rm -f /opt/conda/envs/esm/bin/nvcc" in final
    assert 'ENTRYPOINT ["/usr/local/bin/fold-runtime"]' in runtime_base
    assert 'org.aerith.runtime.source-lock.sha256="${RUNTIME_SOURCE_LOCK_SHA256}"' in final
    assert 'org.aerith.runtime.source.openfold.sha256="${OPENFOLD_SOURCE_SHA256}"' in final
    assert "ARG UBUNTU_SNAPSHOT=20260723T000000Z" in dockerfile
    assert dockerfile.count("snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}") == 4
    assert 'Acquire::Check-Valid-Until "false"' in dockerfile
    assert dockerfile.count("id=aerith-apt-cache") == 2
    assert dockerfile.count("id=aerith-apt-lists") == 2
    assert dockerfile.count("rm -f /etc/apt/apt.conf.d/docker-clean") == 2
    assert "rm -rf /var/lib/apt/lists/*" not in dockerfile
    assert 'org.aerith.runtime.ubuntu-snapshot="${UBUNTU_SNAPSHOT}"' in final

    entrypoint = (Path(__file__).parents[1] / "docker" / "runtime" / "entrypoint.sh").read_text()
    assert "fast_layer_norm_cuda_v2 is not None" in entrypoint
    validator = (
        Path(__file__).parents[1] / "docker" / "runtime" / "validate_runtime.sh"
    ).read_text()
    assert "test ! -e /usr/local/cuda-12.6/bin/nvcc" in validator
    assert "fast_layer_norm_cuda_v2 is not None" in validator

    assembly = (
        Path(__file__).parents[1] / "docker" / "runtime" / "Dockerfile.assemble"
    ).read_text()
    assert "FROM ${UV_COMPONENT_IMAGE} AS uv-component" in assembly
    assert "FROM ${CONDA_COMPONENT_IMAGE} AS conda-component" in assembly
    assert "FROM ${RUNTIME_BASE_IMAGE} AS fold-runtime" in assembly
    assert "COPY --from=uv-component /opt/envs /opt/envs" in assembly
    assert "COPY --from=conda-component /opt/conda /opt/conda" in assembly
    assert "org.aerith.runtime.component.shared.sha256" in assembly
    assert "org.aerith.runtime.component.runtime-base.image-digest" in assembly


_EXPORTER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "export_runtime_image.py"))
export_commands = _EXPORTER["export_commands"]
export_runtime_image = _EXPORTER["export_runtime_image"]
immutable_tag_for_image = _EXPORTER["immutable_tag_for_image"]
repository_from_reference = _EXPORTER["repository_from_reference"]
ImageExportError = _EXPORTER["ImageExportError"]

_VERIFIER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "verify_runtime_image.py"))
verify_runtime_image = _VERIFIER["verify_runtime_image"]


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


def test_runtime_image_verifier_accepts_clean_release_provenance() -> None:
    image_id = "sha256:" + "e" * 64
    labels = {
        "org.opencontainers.image.runtime-lock.sha256": "1" * 64,
        "org.aerith.runtime.recipe.sha256": "2" * 64,
        "org.aerith.runtime.source-lock.sha256": "3" * 64,
        "org.aerith.runtime.source-bundle.sha256": "4" * 64,
        "org.aerith.runtime.component.uv.sha256": "a" * 64,
        "org.aerith.runtime.component.conda.sha256": "b" * 64,
        "org.aerith.runtime.component.shared.sha256": "e" * 64,
        "org.aerith.runtime.component.uv.image-digest": "sha256:" + "c" * 64,
        "org.aerith.runtime.component.conda.image-digest": "sha256:" + "d" * 64,
        "org.aerith.runtime.component.runtime-base.image-digest": "sha256:" + "e" * 64,
        "org.aerith.runtime.source.af3.sha256": "5" * 64,
        "org.aerith.runtime.source.protenix.sha256": "6" * 64,
        "org.aerith.runtime.source.opendde.sha256": "7" * 64,
        "org.aerith.runtime.source.esm.sha256": "8" * 64,
        "org.aerith.runtime.source.openfold.sha256": "9" * 64,
        "org.aerith.runtime.source.dirty": "false",
        "org.aerith.runtime.ubuntu-snapshot": "20260723T000000Z",
    }
    payload = {"Id": image_id, "Config": {"Labels": labels}}

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps([payload]), "")

    inspected = verify_runtime_image("aerith/fold-runtime:test", runner=runner)
    assert inspected["Id"] == image_id


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


def test_production_export_rejects_missing_ubuntu_snapshot(tmp_path: Path) -> None:
    labels = {
        "org.opencontainers.image.runtime-lock.sha256": "1" * 64,
        "org.aerith.runtime.recipe.sha256": "2" * 64,
        "org.aerith.runtime.source-lock.sha256": "3" * 64,
        "org.aerith.runtime.source-bundle.sha256": "4" * 64,
        "org.aerith.runtime.component.uv.sha256": "a" * 64,
        "org.aerith.runtime.component.conda.sha256": "b" * 64,
        "org.aerith.runtime.component.shared.sha256": "e" * 64,
        "org.aerith.runtime.component.uv.image-digest": "sha256:" + "c" * 64,
        "org.aerith.runtime.component.conda.image-digest": "sha256:" + "d" * 64,
        "org.aerith.runtime.component.runtime-base.image-digest": "sha256:" + "e" * 64,
        "org.aerith.runtime.source.af3.sha256": "5" * 64,
        "org.aerith.runtime.source.protenix.sha256": "6" * 64,
        "org.aerith.runtime.source.opendde.sha256": "7" * 64,
        "org.aerith.runtime.source.esm.sha256": "8" * 64,
        "org.aerith.runtime.source.openfold.sha256": "9" * 64,
        "org.aerith.runtime.source.dirty": "false",
    }
    payload = {"Id": "sha256:" + "f" * 64, "Config": {"Labels": labels}}

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps([payload]), "")

    with pytest.raises(ImageExportError, match="Ubuntu snapshot provenance"):
        export_runtime_image(
            "aerith/fold-runtime:local",
            tmp_path,
            runner=runner,
        )


def test_production_export_rejects_dirty_source_provenance(tmp_path: Path) -> None:
    labels = {
        "org.opencontainers.image.runtime-lock.sha256": "1" * 64,
        "org.aerith.runtime.recipe.sha256": "2" * 64,
        "org.aerith.runtime.source-lock.sha256": "3" * 64,
        "org.aerith.runtime.source-bundle.sha256": "4" * 64,
        "org.aerith.runtime.component.uv.sha256": "a" * 64,
        "org.aerith.runtime.component.conda.sha256": "b" * 64,
        "org.aerith.runtime.component.shared.sha256": "e" * 64,
        "org.aerith.runtime.component.uv.image-digest": "sha256:" + "c" * 64,
        "org.aerith.runtime.component.conda.image-digest": "sha256:" + "d" * 64,
        "org.aerith.runtime.component.runtime-base.image-digest": "sha256:" + "e" * 64,
        "org.aerith.runtime.source.af3.sha256": "5" * 64,
        "org.aerith.runtime.source.protenix.sha256": "6" * 64,
        "org.aerith.runtime.source.opendde.sha256": "7" * 64,
        "org.aerith.runtime.source.esm.sha256": "8" * 64,
        "org.aerith.runtime.source.openfold.sha256": "9" * 64,
        "org.aerith.runtime.source.dirty": "true",
        "org.aerith.runtime.ubuntu-snapshot": "20260723T000000Z",
    }
    payload = {"Id": "sha256:" + "d" * 64, "Config": {"Labels": labels}}

    def runner(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, json.dumps([payload]), "")

    with pytest.raises(ImageExportError, match="source provenance is not clean"):
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
        "org.aerith.runtime.source-lock.sha256": "3" * 64,
        "org.aerith.runtime.source-bundle.sha256": "4" * 64,
        "org.aerith.runtime.component.uv.sha256": "a" * 64,
        "org.aerith.runtime.component.conda.sha256": "b" * 64,
        "org.aerith.runtime.component.shared.sha256": "e" * 64,
        "org.aerith.runtime.component.uv.image-digest": "sha256:" + "c" * 64,
        "org.aerith.runtime.component.conda.image-digest": "sha256:" + "d" * 64,
        "org.aerith.runtime.component.runtime-base.image-digest": "sha256:" + "e" * 64,
        "org.aerith.runtime.source.af3.sha256": "5" * 64,
        "org.aerith.runtime.source.protenix.sha256": "6" * 64,
        "org.aerith.runtime.source.opendde.sha256": "7" * 64,
        "org.aerith.runtime.source.esm.sha256": "8" * 64,
        "org.aerith.runtime.source.openfold.sha256": "9" * 64,
        "org.aerith.runtime.source.dirty": "false",
        "org.aerith.runtime.ubuntu-snapshot": "20260723T000000Z",
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
