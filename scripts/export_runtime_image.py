#!/usr/bin/env python3
"""Export a Docker runtime image as a checksummed zstd Docker archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from af3_binder_filter.io_utils import atomic_write_json, atomic_write_text  # noqa: E402


class ImageExportError(RuntimeError):
    """Raised when an image cannot be pinned or exported safely."""


@dataclass(frozen=True, slots=True)
class ImageExportResult:
    immutable_tag: str
    image_id: str
    archive: Path
    archive_sha256: str
    checksum_file: Path
    metadata_file: Path


Runner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[bytes]]

_PROVENANCE_SHA256_LABELS = (
    "org.opencontainers.image.runtime-lock.sha256",
    "org.aerith.runtime.recipe.sha256",
    "org.aerith.runtime.source-bundle.sha256",
    "org.aerith.runtime.source.af3.sha256",
    "org.aerith.runtime.source.protenix.sha256",
    "org.aerith.runtime.source.opendde.sha256",
    "org.aerith.runtime.source.esm.sha256",
)
_RUNTIME_SOURCE_DIRTY_LABEL = "org.aerith.runtime.source.dirty"


def _parse_inspect(stdout: str, image: str) -> dict[str, Any]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ImageExportError(f"docker image inspect returned invalid JSON for {image}") from exc
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
        raise ImageExportError(f"docker image inspect returned an unexpected payload for {image}")
    image_id = payload[0].get("Id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None:
        raise ImageExportError(f"docker image inspect returned an invalid image ID: {image_id!r}")
    return payload[0]


def inspect_image(
    image: str,
    *,
    docker_bin: str = "docker",
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    completed = runner(
        [docker_bin, "image", "inspect", image],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error = (completed.stderr or completed.stdout).strip()
        raise ImageExportError(f"docker image inspect failed for {image}: {error}")
    return _parse_inspect(completed.stdout, image)


def repository_from_reference(reference: str) -> str:
    if not reference or any(character.isspace() for character in reference):
        raise ImageExportError("image reference must be non-empty and contain no whitespace")
    without_digest = reference.split("@", 1)[0]
    final_component = without_digest.rsplit("/", 1)[-1]
    if ":" in final_component:
        without_digest = without_digest[: -(len(final_component) - final_component.rfind(":"))]
    if not without_digest or without_digest.endswith("/"):
        raise ImageExportError(f"cannot derive repository from image reference: {reference}")
    return without_digest


def immutable_tag_for_image(repository: str, image_id: str) -> str:
    if not repository or any(character.isspace() for character in repository):
        raise ImageExportError("repository must be non-empty and contain no whitespace")
    match = re.fullmatch(r"sha256:([0-9a-f]{64})", image_id)
    if match is None:
        raise ImageExportError(f"invalid image ID: {image_id}")
    return f"{repository}:sha256-{match.group(1)}"


def export_commands(
    immutable_tag: str,
    *,
    docker_bin: str = "docker",
    zstd_bin: str = "zstd",
    compression_level: int = 3,
) -> tuple[list[str], list[str]]:
    if compression_level < 1 or compression_level > 19:
        raise ImageExportError("zstd compression level must be between 1 and 19")
    return (
        [docker_bin, "image", "save", immutable_tag],
        [
            zstd_bin,
            "--threads=0",
            "--stdout",
            "--no-progress",
            f"-{compression_level}",
        ],
    )


def _tag_immutable_image(
    source_image: str,
    immutable_tag: str,
    image_id: str,
    *,
    docker_bin: str,
    runner: Runner,
) -> dict[str, Any]:
    existing = runner(
        [docker_bin, "image", "inspect", immutable_tag],
        capture_output=True,
        text=True,
        check=False,
    )
    if existing.returncode == 0:
        inspected = _parse_inspect(existing.stdout, immutable_tag)
        if inspected["Id"] != image_id:
            raise ImageExportError(
                f"immutable tag already points to a different image: {immutable_tag}"
            )
        return inspected
    missing_message = (existing.stderr or existing.stdout).lower()
    if "no such image" not in missing_message and "no such object" not in missing_message:
        raise ImageExportError(
            f"cannot inspect immutable tag {immutable_tag}: "
            f"{(existing.stderr or existing.stdout).strip()}"
        )
    tagged = runner(
        [docker_bin, "image", "tag", source_image, immutable_tag],
        capture_output=True,
        text=True,
        check=False,
    )
    if tagged.returncode != 0:
        raise ImageExportError(
            f"docker image tag failed: {(tagged.stderr or tagged.stdout).strip()}"
        )
    inspected = inspect_image(immutable_tag, docker_bin=docker_bin, runner=runner)
    if inspected["Id"] != image_id:
        raise ImageExportError("immutable tag changed image identity after tagging")
    return inspected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _terminate_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    except (OSError, ProcessLookupError):
        pass


def _normalized_inspect(inspect: dict[str, Any]) -> dict[str, Any]:
    config = inspect.get("Config") if isinstance(inspect.get("Config"), dict) else {}
    rootfs = inspect.get("RootFS") if isinstance(inspect.get("RootFS"), dict) else {}
    return {
        "id": inspect.get("Id"),
        "created": inspect.get("Created"),
        "repo_tags": inspect.get("RepoTags"),
        "repo_digests": inspect.get("RepoDigests"),
        "architecture": inspect.get("Architecture"),
        "os": inspect.get("Os"),
        "size_bytes": inspect.get("Size"),
        "rootfs_layers": rootfs.get("Layers"),
        "labels": config.get("Labels"),
        "entrypoint": config.get("Entrypoint"),
        "cmd": config.get("Cmd"),
    }


def validate_release_provenance(inspect: dict[str, Any]) -> None:
    config = inspect.get("Config") if isinstance(inspect.get("Config"), dict) else {}
    labels = config.get("Labels") if isinstance(config.get("Labels"), dict) else {}
    invalid = [
        name
        for name in _PROVENANCE_SHA256_LABELS
        if re.fullmatch(r"[0-9a-f]{64}", str(labels.get(name, ""))) is None
    ]
    if invalid:
        raise ImageExportError(
            "image is missing release-grade SHA256 provenance labels: " + ", ".join(invalid)
        )
    if labels.get(_RUNTIME_SOURCE_DIRTY_LABEL) != "false":
        raise ImageExportError("image is not release-grade because source provenance is not clean")


def export_runtime_image(
    image: str,
    output_dir: Path,
    *,
    repository: str | None = None,
    docker_bin: str = "docker",
    zstd_bin: str = "zstd",
    compression_level: int = 3,
    force: bool = False,
    require_provenance: bool = True,
    runner: Runner = subprocess.run,
    popen_factory: PopenFactory = subprocess.Popen,
) -> ImageExportResult:
    """Tag by full image ID, then stream docker-save output through zstd."""

    source_inspect = inspect_image(image, docker_bin=docker_bin, runner=runner)
    if require_provenance:
        validate_release_provenance(source_inspect)
    image_id = str(source_inspect["Id"])
    selected_repository = repository or repository_from_reference(image)
    immutable_tag = immutable_tag_for_image(selected_repository, image_id)
    pinned_inspect = _tag_immutable_image(
        image,
        immutable_tag,
        image_id,
        docker_bin=docker_bin,
        runner=runner,
    )
    save_command, zstd_command = export_commands(
        immutable_tag,
        docker_bin=docker_bin,
        zstd_bin=zstd_bin,
        compression_level=compression_level,
    )

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    image_hex = image_id.removeprefix("sha256:")
    repository_slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", selected_repository).strip("-")
    archive = destination / f"{repository_slug}-sha256-{image_hex}.docker.tar.zst"
    checksum_file = archive.with_suffix(archive.suffix + ".sha256")
    metadata_file = archive.with_suffix(archive.suffix + ".metadata.json")
    existing = [path for path in (archive, checksum_file, metadata_file) if path.exists()]
    if existing and not force:
        raise ImageExportError(
            "release output already exists; use force to replace: "
            + ", ".join(str(path) for path in existing)
        )
    partial = archive.with_name(f".{archive.name}.partial")
    partial.unlink(missing_ok=True)

    save_process: subprocess.Popen[bytes] | None = None
    compressor: subprocess.Popen[bytes] | None = None
    try:
        with partial.open("wb") as output, tempfile.TemporaryFile() as save_stderr:
            save_process = popen_factory(
                save_command,
                stdout=subprocess.PIPE,
                stderr=save_stderr,
            )
            if save_process.stdout is None:
                raise ImageExportError("docker image save did not provide a stdout pipe")
            compressor = popen_factory(
                zstd_command,
                stdin=save_process.stdout,
                stdout=output,
                stderr=subprocess.PIPE,
            )
            save_process.stdout.close()
            _compress_stdout, compress_stderr = compressor.communicate()
            save_returncode = save_process.wait()
            save_stderr.seek(0)
            save_error = save_stderr.read().decode("utf-8", errors="replace").strip()
            compress_error = (compress_stderr or b"").decode("utf-8", errors="replace").strip()
            if save_returncode != 0:
                raise ImageExportError(
                    f"docker image save failed with return code {save_returncode}: {save_error}"
                )
            if compressor.returncode != 0:
                raise ImageExportError(
                    f"zstd failed with return code {compressor.returncode}: {compress_error}"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, archive)
    except BaseException:
        _terminate_process(compressor)
        _terminate_process(save_process)
        partial.unlink(missing_ok=True)
        raise

    final_inspect = inspect_image(immutable_tag, docker_bin=docker_bin, runner=runner)
    if final_inspect["Id"] != image_id:
        archive.unlink(missing_ok=True)
        raise ImageExportError("immutable tag changed while the archive was being exported")
    archive_sha256 = _sha256_file(archive)
    atomic_write_text(checksum_file, f"{archive_sha256}  {archive.name}\n")
    metadata = {
        "schema": "aerith.runtime-image-export.v1",
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source_image": image,
        "immutable_tag": immutable_tag,
        "image_id": image_id,
        "archive": archive.name,
        "archive_format": "docker-image-save+zstd",
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "compression": {"tool": zstd_bin, "level": compression_level},
        "commands": {
            "save": save_command,
            "compress": zstd_command,
            "restore": {
                "decompress": [
                    zstd_bin,
                    "--decompress",
                    "--stdout",
                    archive.name,
                ],
                "load": [docker_bin, "image", "load"],
            },
        },
        "image_inspect": _normalized_inspect(pinned_inspect),
    }
    atomic_write_json(metadata_file, metadata)
    return ImageExportResult(
        immutable_tag=immutable_tag,
        image_id=image_id,
        archive=archive,
        archive_sha256=archive_sha256,
        checksum_file=checksum_file,
        metadata_file=metadata_file,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="aerith/fold-runtime:local")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository")
    parser.add_argument("--docker-bin", default="docker")
    parser.add_argument("--zstd-bin", default="zstd")
    parser.add_argument("--compression-level", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-unprovenanced", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = export_runtime_image(
            args.image,
            args.output_dir,
            repository=args.repository,
            docker_bin=args.docker_bin,
            zstd_bin=args.zstd_bin,
            compression_level=args.compression_level,
            force=args.force,
            require_provenance=not args.allow_unprovenanced,
        )
    except (ImageExportError, OSError, subprocess.SubprocessError) as exc:
        print(f"runtime image export error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "immutable_tag": result.immutable_tag,
                "image_id": result.image_id,
                "archive": str(result.archive),
                "archive_sha256": result.archive_sha256,
                "checksum_file": str(result.checksum_file),
                "metadata_file": str(result.metadata_file),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
