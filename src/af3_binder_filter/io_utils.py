"""Atomic, deterministic serialization helpers used by every pipeline stage."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _atomic_replace(path: Path, writer: Any, *, newline: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    _atomic_replace(path, lambda handle: handle.write(text))


def atomic_write_json(path: Path, payload: Any) -> None:
    def write(handle: Any) -> None:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    _atomic_replace(path, write)


def atomic_write_yaml(path: Path, payload: Any) -> None:
    import yaml

    def write(handle: Any) -> None:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)

    _atomic_replace(path, write)


def atomic_write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
    delimiter: str = ",",
) -> None:
    materialized = list(rows)
    if fieldnames is None:
        discovered: list[str] = []
        for row in materialized:
            for key in row:
                if key not in discovered:
                    discovered.append(key)
        fieldnames = discovered

    def write(handle: Any) -> None:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames or ()),
            extrasaction="ignore",
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(materialized)

    _atomic_replace(path, write, newline="")
