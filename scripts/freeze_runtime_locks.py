#!/usr/bin/env python3
"""Freeze the verified host environments into Docker build lock files."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docker" / "runtime" / "locks"


def command(*arguments: str) -> str:
    completed = subprocess.run(
        arguments,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def explicit_conda(environment: str) -> None:
    value = command("conda", "list", "-n", environment, "--explicit")
    atomic_text(OUTPUT / f"{environment}.conda-linux-64.lock", value)


def pypi_packages(environment: str, *, excluded: set[str]) -> list[str]:
    packages = json.loads(command("conda", "list", "-n", environment, "--json"))
    return sorted(
        f"{item['name']}=={item['version']}"
        for item in packages
        if item.get("channel") == "pypi" and item["name"].lower() not in excluded
    )


def uv_packages(python: Path, *, excluded: set[str]) -> list[str]:
    packages = json.loads(
        command("uv", "pip", "list", "--python", str(python), "--format", "json")
    )
    return sorted(
        f"{item['name']}=={item['version']}"
        for item in packages
        if item["name"].lower() not in excluded
    )


def main() -> None:
    explicit_conda("protenix")
    explicit_conda("esm")
    atomic_text(
        OUTPUT / "protenix.pip.lock",
        "\n".join(pypi_packages("protenix", excluded={"protenix"})),
    )
    # fair-esm and OpenFold are installed from the pinned local ESM source tree.
    esm_excluded = {"dllogger", "fair-esm", "openfold"}
    esm_lines = pypi_packages("esm", excluded=esm_excluded)
    # DLLogger is a git-only dependency in the verified environment.
    esm_lines.append(
        "DLLogger @ git+https://github.com/NVIDIA/dllogger.git@"
        "0478734ff7be75adde8d160e04872664d1c62e5f"
    )
    atomic_text(OUTPUT / "esm.pip.lock", "\n".join(sorted(esm_lines)))
    atomic_text(
        OUTPUT / "opendde.pip.lock",
        "\n".join(
            uv_packages(
                Path("/home/structure/Software/OpenDDE/.venv/bin/python"),
                excluded={"opendde"},
            )
        ),
    )


if __name__ == "__main__":
    main()
