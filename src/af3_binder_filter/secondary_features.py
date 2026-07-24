"""Adapt AF3 target features to the local Protenix/OpenDDE contract.

The adapter never performs a search.  It reuses the exact AF3 target MSA and
converts AF3's explicit residue mapping templates into the aligned A3M plus
mmCIF directory expected by the secondary predictors.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeatureBundle,
    FeatureError,
    feature_bundle_content_sha256,
)
from af3_binder_filter.io_utils import atomic_write_json, atomic_write_text
from af3_binder_filter.jobs import file_asset_identity, sequence_sha256

SECONDARY_FEATURE_MANIFEST_VERSION = 3


@dataclass(frozen=True, slots=True)
class SecondaryFeatureBundle:
    sequence_sha256: str
    cache_dir: Path
    non_pairing_a3m: Path
    pairing_a3m: None
    hmmsearch_a3m: Path | None
    template_mmcif_dir: Path
    fingerprint: str
    templates_enabled: bool
    template_count: int

    def validate(self) -> None:
        if not self.non_pairing_a3m.is_file():
            raise FeatureError(f"secondary target MSA is missing: {self.non_pairing_a3m}")
        if self.templates_enabled:
            if self.hmmsearch_a3m is None or not self.hmmsearch_a3m.is_file():
                raise FeatureError("secondary templates enabled without an aligned template A3M")
            if self.template_count < 1 or not any(self.template_mmcif_dir.glob("*.cif")):
                raise FeatureError("secondary templates enabled without staged mmCIF files")


def secondary_feature_bundle_artifact_identity(
    bundle: SecondaryFeatureBundle,
) -> dict[str, Any]:
    """Return exact identities for all adapted features consumed by prediction."""

    template_ids: set[str] = set()
    if bundle.hmmsearch_a3m is not None and bundle.hmmsearch_a3m.is_file():
        for line in bundle.hmmsearch_a3m.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.startswith(">"):
                continue
            identifier = line[1:].split()[0].split("/", 1)[0].lower()
            if identifier != "query" and len(identifier) >= 4:
                template_ids.add(identifier[:4])
    template_paths = {
        pdb_id: bundle.template_mmcif_dir / f"{pdb_id}.cif" for pdb_id in sorted(template_ids)
    }
    missing_templates = [path for path in template_paths.values() if not path.is_file()]
    if missing_templates:
        raise FeatureError(
            "secondary template A3M references missing mmCIF files: "
            + ", ".join(path.name for path in missing_templates)
        )
    return {
        "non_pairing_a3m": file_asset_identity(bundle.non_pairing_a3m),
        "hmmsearch_a3m": file_asset_identity(bundle.hmmsearch_a3m),
        "templates": {pdb_id: file_asset_identity(path) for pdb_id, path in template_paths.items()},
        "templates_enabled": bundle.templates_enabled,
        "template_count": bundle.template_count,
    }


def secondary_feature_bundle_content_sha256(
    bundle: SecondaryFeatureBundle,
) -> str:
    return sequence_sha256(
        json.dumps(
            secondary_feature_bundle_artifact_identity(bundle),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def _entry_id(path: Path) -> str:
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:30]:
        match = re.match(r"\s*data_([A-Za-z0-9]+)", line)
        if match:
            return match.group(1).lower()[:4]
    match = re.search(r"([0-9][A-Za-z0-9]{3})", path.stem)
    return match.group(1).lower() if match else "t" + sequence_sha256(path.name)[:3]


def _template_chain_sequence(path: Path, maximum_index: int) -> tuple[str, str]:
    import biotite.structure as struc
    import biotite.structure.io as strucio

    array = strucio.load_structure(str(path), model=1)
    if getattr(array, "stack_depth", lambda: 1)() > 1:
        array = array[0]
    candidates: list[tuple[int, str, str]] = []
    for chain in sorted(set(str(value) for value in array.chain_id.tolist())):
        chain_array = array[array.chain_id == chain]
        try:
            sequences, _starts = struc.to_sequence(chain_array, allow_hetero=True)
            sequence = str(sequences[0]) if sequences else ""
        except Exception:
            continue
        if len(sequence) > maximum_index:
            candidates.append((len(sequence), chain, sequence))
    if not candidates:
        raise FeatureError(f"no template chain in {path} covers template index {maximum_index}")
    _length, chain_id, sequence = min(candidates, key=lambda item: item[0])
    return chain_id, sequence


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(source.read_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def adapt_af3_features_for_secondary(
    bundle: AF3FeatureBundle,
    target_sequence: str,
    *,
    maximum_templates: int = 4,
    force: bool = False,
) -> SecondaryFeatureBundle:
    """Create a deterministic offline secondary-feature view of an AF3 cache."""

    bundle.validate()
    normalized = "".join(target_sequence.split()).upper()
    fingerprint = sequence_sha256(
        json.dumps(
            {
                "adapter_version": SECONDARY_FEATURE_MANIFEST_VERSION,
                "af3_feature_fingerprint": bundle.fingerprint,
                "af3_feature_content_sha256": feature_bundle_content_sha256(bundle),
                "target_sequence_sha256": sequence_sha256(normalized),
                "maximum_templates": maximum_templates,
                "paired_msa": False,
            },
            sort_keys=True,
        )
    )
    root = bundle.cache_dir / "secondary" / fingerprint[:16]
    target_msa = root / "non_pairing.a3m"
    template_dir = root / "mmcif"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("version") != SECONDARY_FEATURE_MANIFEST_VERSION:
                raise FeatureError("secondary feature manifest version mismatch")
            result = SecondaryFeatureBundle(
                sequence_sha256=sequence_sha256(normalized),
                cache_dir=root,
                non_pairing_a3m=target_msa,
                pairing_a3m=None,
                hmmsearch_a3m=(root / "templates.a3m")
                if payload.get("templates_enabled")
                else None,
                template_mmcif_dir=template_dir,
                fingerprint=fingerprint,
                templates_enabled=bool(payload.get("templates_enabled")),
                template_count=int(payload.get("template_count", 0)),
            )
            result.validate()
            if payload.get("artifact_identity") != secondary_feature_bundle_artifact_identity(
                result
            ):
                raise FeatureError("secondary feature artifact identity mismatch")
            return result
        except Exception:
            pass

    root.mkdir(parents=True, exist_ok=True)
    _atomic_copy(Path(str(bundle.features.unpaired_msa_path)), target_msa)
    template_dir.mkdir(parents=True, exist_ok=True)
    records: list[tuple[str, str]] = []
    rejected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, template in enumerate(bundle.features.templates):
        if len(records) >= maximum_templates:
            break
        try:
            query_indices = [int(value) for value in template.get("queryIndices", [])]
            template_indices = [int(value) for value in template.get("templateIndices", [])]
            if not query_indices or len(query_indices) != len(template_indices):
                raise FeatureError("missing or unequal queryIndices/templateIndices")
            if min(query_indices) < 0 or max(query_indices) >= len(normalized):
                raise FeatureError("queryIndices outside the target sequence")
            source = Path(str(template.get("mmcifPath") or ""))
            if not source.is_file():
                raise FeatureError(f"template mmCIF does not exist: {source}")
            chain_id, template_sequence = _template_chain_sequence(source, max(template_indices))
            aligned = np.full(len(normalized), "-", dtype="<U1")
            for query_index, template_index in zip(query_indices, template_indices, strict=True):
                residue = template_sequence[template_index].upper()
                aligned[query_index] = residue if residue.isalpha() else "X"
            pdb_id = _entry_id(source)
            identifier = f"{pdb_id}_{chain_id}"
            if identifier in used_ids:
                identifier = f"{pdb_id}_{chain_id}.{index + 1}"
            used_ids.add(identifier)
            destination = template_dir / f"{pdb_id}.cif"
            _atomic_copy(source, destination)
            records.append((identifier, "".join(aligned.tolist())))
        except Exception as exc:
            rejected.append({"index": index, "error": str(exc)})

    template_a3m: Path | None = None
    if records:
        template_a3m = root / "templates.a3m"
        atomic_write_text(
            template_a3m,
            f">query\n{normalized}\n"
            + "".join(f">{identifier}\n{aligned}\n" for identifier, aligned in records),
        )
    result = SecondaryFeatureBundle(
        sequence_sha256=sequence_sha256(normalized),
        cache_dir=root,
        non_pairing_a3m=target_msa,
        pairing_a3m=None,
        hmmsearch_a3m=template_a3m,
        template_mmcif_dir=template_dir,
        fingerprint=fingerprint,
        templates_enabled=bool(records),
        template_count=len(records),
    )
    result.validate()
    atomic_write_json(
        manifest_path,
        {
            "version": SECONDARY_FEATURE_MANIFEST_VERSION,
            "fingerprint": fingerprint,
            "source_af3_feature_fingerprint": bundle.fingerprint,
            "target_sequence_sha256": result.sequence_sha256,
            "paired_msa": False,
            "templates_enabled": result.templates_enabled,
            "template_count": result.template_count,
            "rejected_templates": rejected,
            "artifact_identity": secondary_feature_bundle_artifact_identity(result),
        },
    )
    return result


def _a3m_records_by_id(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    current: list[str] = []

    def flush() -> None:
        if not current or not current[0].startswith(">"):
            return
        identifier = current[0][1:].split()[0].split("/", 1)[0].lower()
        records.setdefault(identifier, "\n".join(current) + "\n")

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">") and current:
            flush()
            current = []
        if line.strip():
            current.append(line)
    flush()
    return records


def adapt_local_features_for_secondary(
    bundle: FeatureBundle,
    target_sequence: str,
    *,
    maximum_templates: int = 4,
    force: bool = False,
) -> SecondaryFeatureBundle:
    """Stage a self-contained MSA view and reuse selected local AF3 templates."""

    bundle.validate()
    normalized = "".join(target_sequence.split()).upper()
    source_mmcif_dir = bundle.source_mmcif_dir
    if source_mmcif_dir is None or not source_mmcif_dir.is_dir():
        raise FeatureError("local features do not identify the source mmCIF directory")
    fingerprint = sequence_sha256(
        json.dumps(
            {
                "adapter_version": SECONDARY_FEATURE_MANIFEST_VERSION,
                "local_feature_fingerprint": bundle.fingerprint,
                "local_feature_content_sha256": feature_bundle_content_sha256(bundle),
                "target_sequence_sha256": sequence_sha256(normalized),
                "maximum_templates": maximum_templates,
                "paired_msa": False,
            },
            sort_keys=True,
        )
    )
    root = bundle.cache_dir / "secondary" / fingerprint[:16]
    target_msa = root / "non_pairing.a3m"
    template_a3m = root / "templates.a3m"
    manifest_path = root / "manifest.json"
    if manifest_path.is_file() and not force:
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            if payload.get("version") != SECONDARY_FEATURE_MANIFEST_VERSION:
                raise FeatureError("secondary feature manifest version mismatch")
            templates_enabled = bool(payload.get("templates_enabled"))
            result = SecondaryFeatureBundle(
                sequence_sha256=sequence_sha256(normalized),
                cache_dir=root,
                non_pairing_a3m=target_msa,
                pairing_a3m=None,
                hmmsearch_a3m=template_a3m if templates_enabled else None,
                template_mmcif_dir=source_mmcif_dir,
                fingerprint=fingerprint,
                templates_enabled=templates_enabled,
                template_count=int(payload.get("template_count", 0)),
            )
            result.validate()
            if payload.get("artifact_identity") != secondary_feature_bundle_artifact_identity(
                result
            ):
                raise FeatureError("secondary feature artifact identity mismatch")
            return result
        except Exception:
            pass

    root.mkdir(parents=True, exist_ok=True)
    _atomic_copy(bundle.non_pairing_a3m, target_msa)
    source_records = _a3m_records_by_id(bundle.hmmsearch_a3m)
    template_payload = json.loads(bundle.af3_templates_json.read_text(encoding="utf-8"))
    selected_records: list[str] = []
    selected_templates: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    for index, template in enumerate(template_payload.get("templates", [])):
        if len(selected_records) >= maximum_templates:
            break
        pdb_id = str(template.get("pdbId", "")).lower()
        chain_id = str(template.get("authChainId", ""))
        identifier = f"{pdb_id}_{chain_id}".lower()
        source_cif = source_mmcif_dir / f"{pdb_id}.cif"
        record = source_records.get(identifier)
        if not pdb_id or not chain_id or record is None or not source_cif.is_file():
            rejected.append(
                {
                    "index": index,
                    "identifier": identifier,
                    "error": "matching HMMsearch record or source mmCIF is missing",
                }
            )
            continue
        selected_records.append(record)
        selected_templates.append(
            {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "mmcif": str(source_cif),
            }
        )

    if selected_records:
        atomic_write_text(
            template_a3m,
            f">query\n{normalized}\n" + "".join(selected_records),
        )
    else:
        template_a3m.unlink(missing_ok=True)
    result = SecondaryFeatureBundle(
        sequence_sha256=sequence_sha256(normalized),
        cache_dir=root,
        non_pairing_a3m=target_msa,
        pairing_a3m=None,
        hmmsearch_a3m=template_a3m if selected_records else None,
        template_mmcif_dir=source_mmcif_dir,
        fingerprint=fingerprint,
        templates_enabled=bool(selected_records),
        template_count=len(selected_records),
    )
    result.validate()
    atomic_write_json(
        manifest_path,
        {
            "version": SECONDARY_FEATURE_MANIFEST_VERSION,
            "fingerprint": fingerprint,
            "source_local_feature_fingerprint": bundle.fingerprint,
            "target_sequence_sha256": result.sequence_sha256,
            "paired_msa": False,
            "templates_enabled": result.templates_enabled,
            "template_count": result.template_count,
            "selected_templates": selected_templates,
            "rejected_templates": rejected,
            "artifact_identity": secondary_feature_bundle_artifact_identity(result),
        },
    )
    return result
