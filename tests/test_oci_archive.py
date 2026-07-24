from __future__ import annotations

import io
import json
import runpy
import tarfile
from pathlib import Path

_CONVERTER = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "oci_to_docker_archive.py"))
append_manifest = _CONVERTER["append_manifest"]
docker_manifest = _CONVERTER["docker_manifest"]


def _add_json(archive: tarfile.TarFile, name: str, payload: object) -> None:
    encoded = json.dumps(payload).encode()
    member = tarfile.TarInfo(name)
    member.size = len(encoded)
    archive.addfile(member, io.BytesIO(encoded))


def test_oci_archive_gets_docker_save_manifest(tmp_path: Path) -> None:
    path = tmp_path / "image.tar"
    manifest_digest = "a" * 64
    config_digest = "b" * 64
    layer_digest = "c" * 64
    with tarfile.open(path, mode="w:") as archive:
        _add_json(
            archive,
            "index.json",
            {"manifests": [{"digest": f"sha256:{manifest_digest}"}]},
        )
        _add_json(
            archive,
            f"blobs/sha256/{manifest_digest}",
            {
                "config": {"digest": f"sha256:{config_digest}"},
                "layers": [{"digest": f"sha256:{layer_digest}"}],
            },
        )
        _add_json(archive, f"blobs/sha256/{config_digest}", {})
        _add_json(archive, f"blobs/sha256/{layer_digest}", {})

    payload = docker_manifest(path, "aerith/fold-runtime:local")
    append_manifest(path, payload)

    with tarfile.open(path, mode="r:") as archive:
        generated = json.load(archive.extractfile("manifest.json"))
    assert generated == [
        {
            "Config": f"blobs/sha256/{config_digest}",
            "RepoTags": ["aerith/fold-runtime:local"],
            "Layers": [f"blobs/sha256/{layer_digest}"],
        }
    ]
