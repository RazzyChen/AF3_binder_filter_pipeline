"""Validated, run-local structure derivatives for downstream reuse.

Bundles are immutable and content-addressed by every scientific input that can
change their normalized residue/coordinate representation.  Writers serialize
through a content-ID lock, build in a sibling temporary directory, validate the
entire bundle, and publish it with one directory rename.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol, Sequence

import numpy as np

from af3_binder_filter.io_utils import atomic_write_csv, atomic_write_json
from af3_binder_filter.residue_format import parse_residue_positions


DERIVED_STRUCTURE_SCHEMA = "aerith.derived-structures.v2"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_FILENAMES: dict[str, str] = {
    "complex_pdb": "complex_ab.pdb",
    "target_pdb": "target.pdb",
    "binder_pdb": "binder.pdb",
    "residue_map": "residue_map.tsv",
    "coordinates": "coordinates.npz",
}
_MAP_FIELDS: tuple[str, ...] = (
    "pdb_chain",
    "pdb_residue_number",
    "original_chain",
    "original_res_id",
    "original_ins_code",
    "sequence_position",
    "res_name",
    "mapping_mode",
)
_AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class ResidueLike(Protocol):
    chain_id: str
    original_res_id: int
    original_ins_code: str
    sequence_position: int
    res_name: str
    atom_indexes: np.ndarray
    mapping_mode: str


class SourceModelChangedError(RuntimeError):
    """Raised when a source model no longer matches its pre-parse digest."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        observed_sha256: str,
        phase: str,
    ) -> None:
        self.path = path
        self.expected_sha256 = expected_sha256
        self.observed_sha256 = observed_sha256
        self.phase = phase
        super().__init__(
            f"source model changed {phase}: {path} "
            f"(expected {expected_sha256}, observed {observed_sha256})"
        )


@dataclass(frozen=True, slots=True)
class DerivedStructureArtifacts:
    content_id: str
    source_model_sha256: str
    job_id: str
    backend: str
    target_chain: str
    binder_chain: str
    target_sequence_sha256: str
    binder_sequence_sha256: str
    interface_distance_cutoff: float
    root: Path
    complex_pdb: Path
    target_pdb: Path
    binder_pdb: Path
    residue_map: Path
    coordinates: Path
    manifest: Path
    cache_hit: bool = False

    def as_row(self) -> dict[str, Any]:
        return {
            "derived_structure_status": "success",
            "derived_structure_error": "",
            "derived_structure_cache_hit": self.cache_hit,
            "derived_structure_id": self.content_id,
            "derived_source_model_sha256": self.source_model_sha256,
            "derived_target_sequence_sha256": self.target_sequence_sha256,
            "derived_binder_sequence_sha256": self.binder_sequence_sha256,
            "derived_interface_distance_cutoff": self.interface_distance_cutoff,
            "derived_structure_manifest_path": str(self.manifest),
            "normalized_complex_pdb_path": str(self.complex_pdb),
            "normalized_target_pdb_path": str(self.target_pdb),
            "normalized_binder_pdb_path": str(self.binder_pdb),
            "derived_residue_map_path": str(self.residue_map),
            "derived_coordinates_path": str(self.coordinates),
        }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source_model_sha256(
    path: Path,
    expected_sha256: str,
    *,
    phase: str,
) -> str:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise SourceModelChangedError(
            path,
            expected_sha256=expected_sha256,
            observed_sha256=observed,
            phase=phase,
        )
    return observed


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.strip().upper().encode("ascii")).hexdigest()


def _canonical_json_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def derived_content_id(identity: Mapping[str, Any]) -> str:
    """Return the digest of a complete canonical derivative identity."""

    return _canonical_json_digest(identity)


def _safe_component(value: str) -> str:
    normalized = _SAFE_COMPONENT.sub("_", value.strip()).strip("._")
    return normalized or "unknown"


def _artifact_paths(root: Path) -> dict[str, Path]:
    return {key: root / filename for key, filename in _ARTIFACT_FILENAMES.items()}


def canonical_residue_mapping(
    records: Sequence[ResidueLike],
) -> list[dict[str, Any]]:
    return [
        {
            "pdb_chain": str(record.chain_id),
            "pdb_residue_number": int(record.sequence_position),
            "original_chain": str(record.chain_id),
            "original_res_id": int(record.original_res_id),
            "original_ins_code": str(record.original_ins_code),
            "sequence_position": int(record.sequence_position),
            "res_name": str(record.res_name),
            "mapping_mode": str(record.mapping_mode),
        }
        for record in records
    ]


def _validated_mapping(
    *,
    target_records: Sequence[ResidueLike],
    binder_records: Sequence[ResidueLike],
    target_chain: str,
    binder_chain: str,
    target_sequence: str,
    binder_sequence: str,
) -> list[dict[str, Any]]:
    expected_by_chain = {
        target_chain: target_sequence.strip().upper(),
        binder_chain: binder_sequence.strip().upper(),
    }
    records_by_chain = {
        target_chain: tuple(target_records),
        binder_chain: tuple(binder_records),
    }
    for chain, records in records_by_chain.items():
        if not records:
            raise ValueError(f"chain {chain!r} has no mapped standard residues")
        positions = [int(record.sequence_position) for record in records]
        if len(positions) != len(set(positions)) or any(value <= 0 for value in positions):
            raise ValueError(
                f"chain {chain!r} must have unique positive sequence positions"
            )
        expected = expected_by_chain[chain]
        for record in records:
            position = int(record.sequence_position)
            if position > len(expected):
                raise ValueError(
                    f"chain {chain!r} position {position} exceeds input sequence"
                )
            residue = _AA3_TO_1.get(str(record.res_name))
            if residue is None or expected[position - 1] != residue:
                raise ValueError(
                    f"chain {chain!r} structure residue {record.res_name} at "
                    f"position {position} does not match input sequence"
                )
    mapping = canonical_residue_mapping(
        tuple(target_records) + tuple(binder_records)
    )
    if [row["pdb_chain"] for row in mapping] != (
        [target_chain] * len(target_records)
        + [binder_chain] * len(binder_records)
    ):
        raise ValueError("canonical mapping chain order is not target then binder")
    return mapping


def _identity_payload(
    *,
    source_model_sha256: str,
    job_id: str,
    backend: str,
    target_chain: str,
    binder_chain: str,
    target_sequence: str,
    binder_sequence: str,
    interface_distance_cutoff: float,
    target_interface_positions: Iterable[int],
    binder_interface_positions: Iterable[int],
    residue_mapping: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema": DERIVED_STRUCTURE_SCHEMA,
        "source_model_sha256": source_model_sha256,
        "job_id": job_id,
        "backend": backend,
        "target_chain": target_chain,
        "binder_chain": binder_chain,
        "target_sequence_sha256": sequence_sha256(target_sequence),
        "binder_sequence_sha256": sequence_sha256(binder_sequence),
        "target_sequence_length": len(target_sequence.strip()),
        "binder_sequence_length": len(binder_sequence.strip()),
        "interface_distance_cutoff": float(interface_distance_cutoff),
        "target_interface_positions": sorted(
            {int(value) for value in target_interface_positions}
        ),
        "binder_interface_positions": sorted(
            {int(value) for value in binder_interface_positions}
        ),
        "residue_mapping": [dict(row) for row in residue_mapping],
    }


def _artifacts_from_manifest(
    manifest_path: Path,
    payload: Mapping[str, Any],
    *,
    cache_hit: bool,
) -> DerivedStructureArtifacts:
    root = manifest_path.parent
    identity = payload["identity"]
    artifacts = payload["artifacts"]
    if not isinstance(identity, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("derived manifest identity/artifacts must be mappings")
    resolved: dict[str, Path] = {}
    root_resolved = root.resolve()
    for key in _ARTIFACT_FILENAMES:
        item = artifacts.get(key)
        if not isinstance(item, Mapping):
            raise ValueError(f"derived manifest is missing {key}")
        relative = Path(str(item.get("path", "")))
        if (
            relative.is_absolute()
            or len(relative.parts) != 1
            or relative.name != _ARTIFACT_FILENAMES[key]
        ):
            raise ValueError(f"invalid derived artifact path for {key}")
        path = (root / relative).resolve()
        if path.parent != root_resolved:
            raise ValueError(f"derived artifact escapes cache root: {path}")
        resolved[key] = path
    return DerivedStructureArtifacts(
        content_id=str(payload["content_id"]),
        source_model_sha256=str(identity["source_model_sha256"]),
        job_id=str(identity["job_id"]),
        backend=str(identity["backend"]),
        target_chain=str(identity["target_chain"]),
        binder_chain=str(identity["binder_chain"]),
        target_sequence_sha256=str(identity["target_sequence_sha256"]),
        binder_sequence_sha256=str(identity["binder_sequence_sha256"]),
        interface_distance_cutoff=float(identity["interface_distance_cutoff"]),
        root=root,
        complex_pdb=resolved["complex_pdb"],
        target_pdb=resolved["target_pdb"],
        binder_pdb=resolved["binder_pdb"],
        residue_map=resolved["residue_map"],
        coordinates=resolved["coordinates"],
        manifest=manifest_path,
        cache_hit=cache_hit,
    )


def _validate_mapping(identity: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    try:
        if (
            identity.get("schema") != DERIVED_STRUCTURE_SCHEMA
            or not str(identity.get("job_id", ""))
            or not str(identity.get("backend", ""))
            or _SHA256.fullmatch(str(identity.get("source_model_sha256", "")))
            is None
            or _SHA256.fullmatch(str(identity.get("target_sequence_sha256", "")))
            is None
            or _SHA256.fullmatch(str(identity.get("binder_sequence_sha256", "")))
            is None
        ):
            return None
        target_length = int(identity.get("target_sequence_length", 0))
        binder_length = int(identity.get("binder_sequence_length", 0))
        distance = float(identity.get("interface_distance_cutoff"))
        if (
            target_length <= 0
            or binder_length <= 0
            or not np.isfinite(distance)
            or distance <= 0
        ):
            return None
        interface_by_key: dict[str, list[int]] = {}
        for key in (
            "target_interface_positions",
            "binder_interface_positions",
        ):
            raw = identity.get(key)
            if not isinstance(raw, list) or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw
            ):
                return None
            positions = [int(value) for value in raw]
            if positions != sorted(set(positions)) or any(
                position <= 0 for position in positions
            ):
                return None
            interface_by_key[key] = positions
    except (TypeError, ValueError, OverflowError):
        return None
    mapping = identity.get("residue_mapping")
    if not isinstance(mapping, list) or not mapping:
        return None
    target_chain = str(identity.get("target_chain", ""))
    binder_chain = str(identity.get("binder_chain", ""))
    if not target_chain or not binder_chain or target_chain == binder_chain:
        return None
    limits = {target_chain: target_length, binder_chain: binder_length}
    seen: dict[str, set[int]] = {target_chain: set(), binder_chain: set()}
    previous_position: dict[str, int] = {target_chain: 0, binder_chain: 0}
    mapping_mode: dict[str, str | None] = {target_chain: None, binder_chain: None}
    normalized: list[dict[str, Any]] = []
    encountered_binder = False
    for item in mapping:
        if not isinstance(item, Mapping) or not all(field in item for field in _MAP_FIELDS):
            return None
        chain = str(item["pdb_chain"])
        if chain not in seen or str(item["original_chain"]) != chain:
            return None
        if chain == binder_chain:
            encountered_binder = True
        elif encountered_binder:
            return None
        position = int(item["sequence_position"])
        mode = str(item["mapping_mode"])
        if (
            int(item["pdb_residue_number"]) != position
            or position <= 0
            or position > limits[chain]
            or position in seen[chain]
            or position <= previous_position[chain]
            or str(item["res_name"]) not in _AA3_TO_1
            or mode
            not in {
                "author_residue_ids",
                "complete_sequence_order",
                "unique_exact_subsequence",
            }
            or mapping_mode[chain] not in {None, mode}
        ):
            return None
        seen[chain].add(position)
        previous_position[chain] = position
        mapping_mode[chain] = mode
        normalized.append(
            {
                field: (
                    int(item[field])
                    if field in {
                        "pdb_residue_number",
                        "original_res_id",
                        "sequence_position",
                    }
                    else str(item[field])
                )
                for field in _MAP_FIELDS
            }
        )
    if not seen[target_chain] or not seen[binder_chain]:
        return None
    if not set(interface_by_key["target_interface_positions"]).issubset(
        seen[target_chain]
    ) or not set(interface_by_key["binder_interface_positions"]).issubset(
        seen[binder_chain]
    ):
        return None
    return normalized


def _validate_npz(path: Path, identity: Mapping[str, Any], mapping: list[dict[str, Any]]) -> bool:
    with np.load(path, allow_pickle=False) as coordinates:
        required = {
            "ca_coord",
            "chain_id",
            "sequence_position",
            "res_name",
            "is_interface",
        }
        if set(coordinates.files) != required:
            return False
        ca_coord = coordinates["ca_coord"]
        chain_id = coordinates["chain_id"]
        positions = coordinates["sequence_position"]
        residue_names = coordinates["res_name"]
        interface = coordinates["is_interface"]
        count = len(mapping)
        if (
            ca_coord.shape != (count, 3)
            or not np.issubdtype(ca_coord.dtype, np.floating)
            or not np.all(np.isfinite(ca_coord))
            or chain_id.shape != (count,)
            or chain_id.dtype.kind not in {"U", "S"}
            or positions.shape != (count,)
            or not np.issubdtype(positions.dtype, np.integer)
            or np.any(positions <= 0)
            or residue_names.shape != (count,)
            or residue_names.dtype.kind not in {"U", "S"}
            or interface.shape != (count,)
            or not np.issubdtype(interface.dtype, np.bool_)
        ):
            return False
        expected_chains = [str(item["pdb_chain"]) for item in mapping]
        expected_positions = [int(item["sequence_position"]) for item in mapping]
        expected_names = [str(item["res_name"]) for item in mapping]
        if (
            chain_id.astype(str).tolist() != expected_chains
            or positions.astype(int).tolist() != expected_positions
            or residue_names.astype(str).tolist() != expected_names
        ):
            return False
        target_chain = str(identity["target_chain"])
        binder_chain = str(identity["binder_chain"])
        if set(expected_chains) != {target_chain, binder_chain}:
            return False
        target_interface = {
            int(value) for value in identity["target_interface_positions"]
        }
        binder_interface = {
            int(value) for value in identity["binder_interface_positions"]
        }
        expected_interface = [
            (
                int(item["sequence_position"])
                in (
                    target_interface
                    if item["pdb_chain"] == target_chain
                    else binder_interface
                )
            )
            for item in mapping
        ]
        return interface.astype(bool).tolist() == expected_interface


def _validate_residue_map(path: Path, mapping: list[dict[str, Any]]) -> bool:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = [{field: str(item[field]) for field in _MAP_FIELDS} for item in mapping]
    return rows == expected


def validate_derived_manifest(
    manifest_path: Path,
    *,
    content_id: str | None = None,
    source_model_sha256: str | None = None,
    job_id: str | None = None,
    backend: str | None = None,
    target_chain: str | None = None,
    binder_chain: str | None = None,
    target_sequence_sha256: str | None = None,
    binder_sequence_sha256: str | None = None,
    interface_distance_cutoff: float | None = None,
    target_interface_positions: Iterable[int] | None = None,
    binder_interface_positions: Iterable[int] | None = None,
    residue_mapping: Sequence[Mapping[str, Any]] | None = None,
) -> DerivedStructureArtifacts | None:
    """Return a semantically and cryptographically valid bundle, else ``None``."""

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != DERIVED_STRUCTURE_SCHEMA:
            return None
        identity = payload.get("identity")
        if not isinstance(identity, dict) or identity.get("schema") != DERIVED_STRUCTURE_SCHEMA:
            return None
        actual_content_id = derived_content_id(identity)
        if (
            _SHA256.fullmatch(actual_content_id) is None
            or payload.get("content_id") != actual_content_id
        ):
            return None
        duplicated_identity_fields = (
            "source_model_sha256",
            "job_id",
            "backend",
            "target_chain",
            "binder_chain",
            "target_sequence_sha256",
            "binder_sequence_sha256",
            "interface_distance_cutoff",
            "target_interface_positions",
            "binder_interface_positions",
            "residue_mapping",
        )
        if any(
            payload.get(key) != identity.get(key)
            for key in duplicated_identity_fields
        ):
            return None
        expected_scalars = {
            "content_id": (actual_content_id, content_id),
            "source_model_sha256": (identity.get("source_model_sha256"), source_model_sha256),
            "job_id": (identity.get("job_id"), job_id),
            "backend": (identity.get("backend"), backend),
            "target_chain": (identity.get("target_chain"), target_chain),
            "binder_chain": (identity.get("binder_chain"), binder_chain),
            "target_sequence_sha256": (
                identity.get("target_sequence_sha256"),
                target_sequence_sha256,
            ),
            "binder_sequence_sha256": (
                identity.get("binder_sequence_sha256"),
                binder_sequence_sha256,
            ),
        }
        if any(
            expected is not None and str(actual) != str(expected)
            for actual, expected in expected_scalars.values()
        ):
            return None
        if interface_distance_cutoff is not None and not np.isclose(
            float(identity.get("interface_distance_cutoff")),
            float(interface_distance_cutoff),
            rtol=0,
            atol=1e-12,
        ):
            return None
        for key, expected in (
            ("target_interface_positions", target_interface_positions),
            ("binder_interface_positions", binder_interface_positions),
        ):
            if expected is not None and list(identity.get(key, ())) != sorted(
                {int(value) for value in expected}
            ):
                return None
        mapping = _validate_mapping(identity)
        if mapping is None:
            return None
        if residue_mapping is not None and mapping != [dict(row) for row in residue_mapping]:
            return None
        result = _artifacts_from_manifest(manifest_path, payload, cache_hit=True)
        artifact_payload = payload["artifacts"]
        for key, path in (
            ("complex_pdb", result.complex_pdb),
            ("target_pdb", result.target_pdb),
            ("binder_pdb", result.binder_pdb),
            ("residue_map", result.residue_map),
            ("coordinates", result.coordinates),
        ):
            item = artifact_payload[key]
            if (
                not path.is_file()
                or path.stat().st_size <= 0
                or int(item.get("size", -1)) != path.stat().st_size
                or file_sha256(path) != item.get("sha256")
            ):
                return None
        if not _validate_residue_map(result.residue_map, mapping):
            return None
        if not _validate_npz(result.coordinates, identity, mapping):
            return None
        return result
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
        UnicodeError,
        OverflowError,
        EOFError,
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
    ):
        return None


def _strict_row_positions(value: Any, expected_chain: str) -> frozenset[int]:
    """Parse row residues while enforcing v2 chain qualifiers when supplied."""

    if isinstance(value, str):
        for raw in re.split(r"[;,]", value):
            token = raw.strip()
            if not token or ":" not in token:
                continue
            chain, _separator, _position = token.rpartition(":")
            if chain != expected_chain:
                raise ValueError(
                    f"residue {token!r} is not on expected chain {expected_chain!r}"
                )
    positions = parse_residue_positions(value)
    if any(position <= 0 for position in positions):
        raise ValueError("interface positions must be positive")
    return positions


def _row_value(row: Mapping[str, Any], prefix: str, name: str) -> Any:
    prefixed = f"{prefix}_{name}" if prefix else name
    if prefixed in row:
        return row.get(prefixed)
    return row.get(name)


def row_job_identifier(row: Mapping[str, Any]) -> str | None:
    """Return a scalar row identity, honoring an empty ``job_name`` fallback."""

    try:
        value = row.get("job_name") or row.get("job_id") or ""
    except (TypeError, ValueError):
        return None
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        return None
    normalized = str(value).strip()
    return normalized or None


def row_structure_is_eligible(
    row: Mapping[str, Any],
    *,
    prefix: str,
) -> bool:
    """Return whether a row may safely supply or fall back to a structure."""

    status = _row_value(row, prefix, "interface_status")
    provenance = _row_value(row, prefix, "source_model_provenance_status")
    return status == "success" and provenance != "changed"


def validated_artifacts_from_row(
    row: Mapping[str, Any],
    *,
    prefix: str = "effective",
) -> DerivedStructureArtifacts | None:
    """Strictly bind a selected row to its current derivative manifest."""

    try:
        job_id = row_job_identifier(row)
        if job_id is None:
            return None
        backend = str(_row_value(row, prefix, "backend") or "")
        target_chain = str(row.get("target_chain") or "")
        binder_chain = str(row.get("binder_chain") or "")
        target_sequence = str(row.get("target_sequence") or "")
        binder_sequence = str(row.get("binder_sequence") or "")
        manifest_value = _row_value(
            row,
            prefix,
            "derived_structure_manifest_path",
        )
        content_id = str(_row_value(row, prefix, "derived_structure_id") or "")
        source_sha = str(
            _row_value(row, prefix, "derived_source_model_sha256") or ""
        )
        distance = _row_value(row, prefix, "derived_interface_distance_cutoff")
        if not all(
            (
                job_id,
                backend,
                target_chain,
                binder_chain,
                target_sequence,
                binder_sequence,
                manifest_value,
                content_id,
                source_sha,
            )
        ) or distance in (None, ""):
            return None
        target_positions = _strict_row_positions(
            _row_value(row, prefix, "target_interface_residues"),
            target_chain,
        )
        binder_positions = _strict_row_positions(
            _row_value(row, prefix, "binder_interface_residues"),
            binder_chain,
        )
        source_path_value = _row_value(row, prefix, "best_model_path")
        if source_path_value not in (None, ""):
            source_path = Path(str(source_path_value))
            if not source_path.is_file() or file_sha256(source_path) != source_sha:
                return None
        artifacts = validate_derived_manifest(
            Path(str(manifest_value)),
            content_id=content_id,
            source_model_sha256=source_sha,
            job_id=job_id,
            backend=backend,
            target_chain=target_chain,
            binder_chain=binder_chain,
            target_sequence_sha256=sequence_sha256(target_sequence),
            binder_sequence_sha256=sequence_sha256(binder_sequence),
            interface_distance_cutoff=float(distance),
            target_interface_positions=target_positions,
            binder_interface_positions=binder_positions,
        )
        if artifacts is None:
            return None
        expected_paths = {
            "normalized_complex_pdb_path": artifacts.complex_pdb,
            "normalized_target_pdb_path": artifacts.target_pdb,
            "normalized_binder_pdb_path": artifacts.binder_pdb,
            "derived_residue_map_path": artifacts.residue_map,
            "derived_coordinates_path": artifacts.coordinates,
        }
        for name, expected in expected_paths.items():
            supplied = _row_value(row, prefix, name)
            if supplied in (None, "") or Path(str(supplied)).resolve() != expected.resolve():
                return None
        return artifacts
    except (OSError, ValueError, TypeError, UnicodeError):
        return None


def _atomic_save_structure(path: Path, array: Any) -> None:
    import biotite.structure.io as strucio

    strucio.save_structure(str(path), array)


def _save_coordinates(
    path: Path,
    *,
    ca_coord: np.ndarray,
    chain_id: np.ndarray,
    sequence_position: np.ndarray,
    res_name: np.ndarray,
    is_interface: np.ndarray,
) -> None:
    with path.open("wb") as handle:
        np.savez_compressed(
            handle,
            ca_coord=ca_coord,
            chain_id=chain_id,
            sequence_position=sequence_position,
            res_name=res_name,
            is_interface=is_interface,
        )
        handle.flush()
        os.fsync(handle.fileno())


def _normalized_array(
    array: Any,
    records: Sequence[ResidueLike],
) -> Any:
    keep_indexes = np.concatenate([record.atom_indexes for record in records])
    converted = array[keep_indexes].copy()
    cursor = 0
    categories = set(converted.get_annotation_categories())
    for record in records:
        count = len(record.atom_indexes)
        converted.res_id[cursor : cursor + count] = record.sequence_position
        if "ins_code" in categories:
            converted.ins_code[cursor : cursor + count] = ""
        if "hetero" in categories:
            converted.hetero[cursor : cursor + count] = False
        cursor += count
    return converted


def _coordinate_arrays(
    array: Any,
    records: Sequence[ResidueLike],
    *,
    target_chain: str,
    binder_chain: str,
    target_interface_positions: set[int],
    binder_interface_positions: set[int],
) -> dict[str, np.ndarray]:
    coordinates: list[np.ndarray] = []
    for record in records:
        residue = array[record.atom_indexes]
        ca = residue[residue.atom_name == "CA"]
        coordinate = ca.coord[0] if len(ca) else residue.coord.mean(axis=0)
        coordinates.append(np.asarray(coordinate, dtype=np.float32))
    return {
        "ca_coord": np.asarray(coordinates, dtype=np.float32).reshape((-1, 3)),
        "chain_id": np.asarray([record.chain_id for record in records], dtype="U16"),
        "sequence_position": np.asarray(
            [record.sequence_position for record in records],
            dtype=np.int32,
        ),
        "res_name": np.asarray([record.res_name for record in records], dtype="U8"),
        "is_interface": np.asarray(
            [
                record.sequence_position
                in (
                    target_interface_positions
                    if record.chain_id == target_chain
                    else binder_interface_positions
                )
                for record in records
            ],
            dtype=np.bool_,
        ),
    }


def _write_bundle(
    root: Path,
    *,
    array: Any,
    records: Sequence[ResidueLike],
    mapping: list[dict[str, Any]],
    identity: dict[str, Any],
    content_id: str,
    source_model_path: Path,
) -> None:
    root.mkdir(parents=False, exist_ok=False)
    paths = _artifact_paths(root)
    normalized = _normalized_array(array, records)
    target_chain = str(identity["target_chain"])
    binder_chain = str(identity["binder_chain"])
    _atomic_save_structure(paths["complex_pdb"], normalized)
    _atomic_save_structure(
        paths["target_pdb"],
        normalized[normalized.chain_id == target_chain],
    )
    _atomic_save_structure(
        paths["binder_pdb"],
        normalized[normalized.chain_id == binder_chain],
    )
    atomic_write_csv(
        paths["residue_map"],
        mapping,
        fieldnames=_MAP_FIELDS,
        delimiter="\t",
    )
    _save_coordinates(
        paths["coordinates"],
        **_coordinate_arrays(
            array,
            records,
            target_chain=target_chain,
            binder_chain=binder_chain,
            target_interface_positions=set(identity["target_interface_positions"]),
            binder_interface_positions=set(identity["binder_interface_positions"]),
        ),
    )
    artifact_payload = {
        key: {
            "path": path.name,
            "sha256": file_sha256(path),
            "size": path.stat().st_size,
        }
        for key, path in paths.items()
    }
    atomic_write_json(
        root / "manifest.json",
        {
            "schema": DERIVED_STRUCTURE_SCHEMA,
            "content_id": content_id,
            "source_model_path": str(source_model_path),
            **{
                key: identity[key]
                for key in (
                    "source_model_sha256",
                    "job_id",
                    "backend",
                    "target_chain",
                    "binder_chain",
                    "target_sequence_sha256",
                    "binder_sequence_sha256",
                    "interface_distance_cutoff",
                    "target_interface_positions",
                    "binder_interface_positions",
                    "residue_mapping",
                )
            },
            "identity": identity,
            "artifacts": artifact_payload,
        },
    )


def materialize_derived_structures(
    array: Any,
    *,
    source_model_path: Path,
    expected_source_model_sha256: str,
    target_chain: str,
    binder_chain: str,
    target_sequence: str,
    binder_sequence: str,
    interface_distance_cutoff: float,
    target_records: Sequence[ResidueLike],
    binder_records: Sequence[ResidueLike],
    artifacts_root: Path,
    backend: str,
    job_id: str,
    target_interface_positions: Iterable[int] = (),
    binder_interface_positions: Iterable[int] = (),
) -> DerivedStructureArtifacts:
    """Create or reuse one immutable, atomically published derivative bundle."""

    source_path = source_model_path.resolve()
    source_sha = _verify_source_model_sha256(
        source_path,
        expected_source_model_sha256,
        phase="after interface parsing",
    )
    target_positions = {int(value) for value in target_interface_positions}
    binder_positions = {int(value) for value in binder_interface_positions}
    mapping = _validated_mapping(
        target_records=target_records,
        binder_records=binder_records,
        target_chain=target_chain,
        binder_chain=binder_chain,
        target_sequence=target_sequence,
        binder_sequence=binder_sequence,
    )
    mapped_target = {
        int(row["sequence_position"])
        for row in mapping
        if row["pdb_chain"] == target_chain
    }
    mapped_binder = {
        int(row["sequence_position"])
        for row in mapping
        if row["pdb_chain"] == binder_chain
    }
    if not target_positions.issubset(mapped_target) or not binder_positions.issubset(
        mapped_binder
    ):
        raise ValueError("interface positions are not present in canonical mapping")
    identity = _identity_payload(
        source_model_sha256=source_sha,
        job_id=job_id,
        backend=backend,
        target_chain=target_chain,
        binder_chain=binder_chain,
        target_sequence=target_sequence,
        binder_sequence=binder_sequence,
        interface_distance_cutoff=interface_distance_cutoff,
        target_interface_positions=target_positions,
        binder_interface_positions=binder_positions,
        residue_mapping=mapping,
    )
    content_id = derived_content_id(identity)
    parent = (
        artifacts_root
        / _safe_component(backend)
        / _safe_component(job_id)
    )
    parent.mkdir(parents=True, exist_ok=True)
    root = parent / content_id
    lock_path = parent / f".{content_id}.lock"
    records = tuple(target_records) + tuple(binder_records)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        _verify_source_model_sha256(
            source_path,
            expected_source_model_sha256,
            phase="while waiting to publish derivatives",
        )
        cached = validate_derived_manifest(
            root / "manifest.json",
            content_id=content_id,
            source_model_sha256=source_sha,
            job_id=job_id,
            backend=backend,
            target_chain=target_chain,
            binder_chain=binder_chain,
            target_sequence_sha256=identity["target_sequence_sha256"],
            binder_sequence_sha256=identity["binder_sequence_sha256"],
            interface_distance_cutoff=interface_distance_cutoff,
            target_interface_positions=target_positions,
            binder_interface_positions=binder_positions,
            residue_mapping=mapping,
        )
        if cached is not None:
            _verify_source_model_sha256(
                source_path,
                expected_source_model_sha256,
                phase="while validating cached derivatives",
            )
            return cached

        temporary_parent = Path(
            tempfile.mkdtemp(prefix=f".{content_id}.building-", dir=parent)
        )
        temporary_root = temporary_parent / "bundle"
        try:
            _write_bundle(
                temporary_root,
                array=array,
                records=records,
                mapping=mapping,
                identity=identity,
                content_id=content_id,
                source_model_path=source_path,
            )
            _verify_source_model_sha256(
                source_path,
                expected_source_model_sha256,
                phase="while building derivatives",
            )
            built = validate_derived_manifest(
                temporary_root / "manifest.json",
                content_id=content_id,
                source_model_sha256=source_sha,
                job_id=job_id,
                backend=backend,
                target_chain=target_chain,
                binder_chain=binder_chain,
                target_sequence_sha256=identity["target_sequence_sha256"],
                binder_sequence_sha256=identity["binder_sequence_sha256"],
                interface_distance_cutoff=interface_distance_cutoff,
                target_interface_positions=target_positions,
                binder_interface_positions=binder_positions,
                residue_mapping=mapping,
            )
            if built is None:
                raise RuntimeError(
                    f"failed to validate derived structure build {temporary_root}"
                )
            if root.exists():
                shutil.rmtree(root)
            os.replace(temporary_root, root)
        finally:
            shutil.rmtree(temporary_parent, ignore_errors=True)

        published = validate_derived_manifest(
            root / "manifest.json",
            content_id=content_id,
            source_model_sha256=source_sha,
            job_id=job_id,
            backend=backend,
            target_chain=target_chain,
            binder_chain=binder_chain,
            target_sequence_sha256=identity["target_sequence_sha256"],
            binder_sequence_sha256=identity["binder_sequence_sha256"],
            interface_distance_cutoff=interface_distance_cutoff,
            target_interface_positions=target_positions,
            binder_interface_positions=binder_positions,
            residue_mapping=mapping,
        )
        if published is None:
            raise RuntimeError(f"failed to validate published structures at {root}")
        _verify_source_model_sha256(
            source_path,
            expected_source_model_sha256,
            phase="while publishing derivatives",
        )
        return DerivedStructureArtifacts(
            content_id=published.content_id,
            source_model_sha256=published.source_model_sha256,
            job_id=published.job_id,
            backend=published.backend,
            target_chain=published.target_chain,
            binder_chain=published.binder_chain,
            target_sequence_sha256=published.target_sequence_sha256,
            binder_sequence_sha256=published.binder_sequence_sha256,
            interface_distance_cutoff=published.interface_distance_cutoff,
            root=published.root,
            complex_pdb=published.complex_pdb,
            target_pdb=published.target_pdb,
            binder_pdb=published.binder_pdb,
            residue_map=published.residue_map,
            coordinates=published.coordinates,
            manifest=published.manifest,
            cache_hit=False,
        )
