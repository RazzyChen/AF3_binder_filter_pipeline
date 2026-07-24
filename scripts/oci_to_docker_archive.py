#!/usr/bin/env python3
"""Add Docker save metadata to a single-image OCI layout tar archive."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any


class ConversionError(RuntimeError):
    """Raised when the source archive does not match the supported contract."""


def _read_json(archive: tarfile.TarFile, member_name: str) -> dict[str, Any]:
    try:
        member = archive.getmember(member_name)
    except KeyError as exc:
        raise ConversionError(f"archive member is missing: {member_name}") from exc
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ConversionError(f"archive member is not a regular file: {member_name}")
    try:
        payload = json.load(extracted)
    except json.JSONDecodeError as exc:
        raise ConversionError(f"archive member is not valid JSON: {member_name}") from exc
    if not isinstance(payload, dict):
        raise ConversionError(f"archive member must contain an object: {member_name}")
    return payload


def docker_manifest(path: Path, tag: str) -> bytes:
    if not tag or any(character.isspace() for character in tag):
        raise ConversionError("tag must be non-empty and contain no whitespace")
    with tarfile.open(path, mode="r:") as archive:
        try:
            archive.getmember("manifest.json")
        except KeyError:
            pass
        else:
            raise ConversionError("archive already contains manifest.json")

        index = _read_json(archive, "index.json")
        descriptors = index.get("manifests")
        if not isinstance(descriptors, list) or len(descriptors) != 1:
            raise ConversionError("OCI index must contain exactly one image manifest")
        descriptor = descriptors[0]
        if not isinstance(descriptor, dict):
            raise ConversionError("OCI image descriptor must be an object")
        manifest_digest = descriptor.get("digest")
        if not isinstance(manifest_digest, str) or not manifest_digest.startswith("sha256:"):
            raise ConversionError("OCI image manifest must use a sha256 digest")

        manifest = _read_json(
            archive,
            "blobs/sha256/" + manifest_digest.removeprefix("sha256:"),
        )
        config = manifest.get("config")
        layers = manifest.get("layers")
        if not isinstance(config, dict) or not isinstance(layers, list):
            raise ConversionError("OCI manifest must contain config and layers")
        config_digest = config.get("digest")
        if not isinstance(config_digest, str) or not config_digest.startswith("sha256:"):
            raise ConversionError("OCI config must use a sha256 digest")

        layer_paths: list[str] = []
        for layer in layers:
            if not isinstance(layer, dict):
                raise ConversionError("OCI layer descriptor must be an object")
            digest = layer.get("digest")
            if not isinstance(digest, str) or not digest.startswith("sha256:"):
                raise ConversionError("OCI layer must use a sha256 digest")
            layer_paths.append("blobs/sha256/" + digest.removeprefix("sha256:"))

        docker_payload = [
            {
                "Config": ("blobs/sha256/" + config_digest.removeprefix("sha256:")),
                "RepoTags": [tag],
                "Layers": layer_paths,
            }
        ]
        return json.dumps(
            docker_payload,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")


def append_manifest(path: Path, payload: bytes) -> None:
    member = tarfile.TarInfo("manifest.json")
    member.mode = 0o644
    member.mtime = 0
    member.size = len(payload)
    with tarfile.open(path, mode="a:") as archive:
        archive.addfile(member, io.BytesIO(payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    archive = args.archive.expanduser().resolve()
    if not archive.is_file():
        raise ConversionError(f"OCI archive does not exist: {archive}")
    payload = docker_manifest(archive, args.tag)
    append_manifest(archive, payload)
    print(f"added Docker manifest for {args.tag}: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
