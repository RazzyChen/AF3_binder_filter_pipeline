"""Extract chain A MSA/template features from AF3 target data JSON."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from af3_binder_filter.af3_json import TargetFeatures, sanitize_job_name
from af3_binder_filter.io_utils import atomic_write_text


class TargetDataError(ValueError):
    """Raised when target AF3 data cannot be reused for complex inputs."""


def _first_a3m_sequence(path: Path) -> str | None:
    sequence: list[str] = []
    seen_header = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            if seen_header:
                break
            seen_header = True
        elif seen_header and line.strip():
            sequence.append(line.strip())
    return "".join(sequence).upper() if sequence else None


def _protein_for_chain(data: dict[str, Any], chain_id: str) -> dict[str, Any]:
    for entry in data.get("sequences", []):
        protein = entry.get("protein", {})
        ids = protein.get("id")
        if ids == chain_id or (isinstance(ids, list) and chain_id in ids):
            return protein
    raise TargetDataError(f"chain {chain_id!r} not found in target data JSON")


def _write_text_feature(
    *,
    text: str | None,
    existing_path: str | None,
    data_json_path: Path,
    output_root: Path,
    relative_path: str,
    force: bool,
) -> str | None:
    output_path = output_root / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if text:
        if force or not output_path.exists():
            atomic_write_text(output_path, text)
        return relative_path

    if existing_path:
        source = Path(existing_path)
        if not source.is_absolute():
            source = data_json_path.parent / existing_path
        if not source.exists():
            raise TargetDataError(f"referenced feature file does not exist: {source}")
        if force or not output_path.exists():
            shutil.copy2(source, output_path)
        return relative_path

    return None


def extract_target_features(
    target_data_json: Path,
    output_root: Path,
    *,
    chain_id: str = "A",
    prefix: str | None = None,
    expected_sequence: str | None = None,
    force: bool = False,
) -> TargetFeatures:
    """Externalize target chain MSA/templates next to complex input JSONs."""

    if not target_data_json.exists():
        raise TargetDataError(f"target data JSON does not exist: {target_data_json}")

    data = json.loads(target_data_json.read_text())
    if not isinstance(data, dict):
        raise TargetDataError(f"target data JSON must contain one AF3 object: {target_data_json}")

    protein = _protein_for_chain(data, chain_id)
    if expected_sequence is not None:
        actual_sequence = "".join(str(protein.get("sequence", "")).split()).upper()
        normalized_expected = "".join(expected_sequence.split()).upper()
        if actual_sequence != normalized_expected:
            raise TargetDataError(
                f"target feature sequence mismatch for chain {chain_id!r}: "
                f"expected {normalized_expected}, found {actual_sequence or '<missing>'}"
            )
    feature_prefix = sanitize_job_name(prefix or data.get("name") or target_data_json.stem)

    unpaired_path = _write_text_feature(
        text=protein.get("unpairedMsa"),
        existing_path=protein.get("unpairedMsaPath"),
        data_json_path=target_data_json,
        output_root=output_root,
        relative_path=f"msas/{feature_prefix}_{chain_id}_unpaired.a3m",
        force=force,
    )
    paired_path = _write_text_feature(
        text=protein.get("pairedMsa"),
        existing_path=protein.get("pairedMsaPath"),
        data_json_path=target_data_json,
        output_root=output_root,
        relative_path=f"msas/{feature_prefix}_{chain_id}_paired.a3m",
        force=force,
    )
    if expected_sequence is not None:
        normalized_expected = "".join(expected_sequence.split()).upper()
        for relative_path in (unpaired_path, paired_path):
            if relative_path is None:
                continue
            actual_query = _first_a3m_sequence(output_root / relative_path)
            if actual_query != normalized_expected:
                raise TargetDataError(
                    f"target MSA query mismatch: expected {normalized_expected}, "
                    f"found {actual_query or '<missing>'} in {relative_path}"
                )

    templates: list[dict[str, Any]] = []
    for index, template in enumerate(protein.get("templates") or []):
        template_copy = {key: value for key, value in template.items() if key != "mmcif"}
        relative_path = f"templates/{feature_prefix}_{chain_id}_template_{index}.cif"
        output_path = output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if "mmcif" in template and template["mmcif"]:
            if force or not output_path.exists():
                atomic_write_text(output_path, str(template["mmcif"]))
        elif template.get("mmcifPath"):
            source = Path(str(template["mmcifPath"]))
            if not source.is_absolute():
                source = target_data_json.parent / source
            if not source.exists():
                raise TargetDataError(f"referenced template file does not exist: {source}")
            if force or not output_path.exists():
                shutil.copy2(source, output_path)
        else:
            raise TargetDataError(f"template {index} has neither mmcif nor mmcifPath")

        template_copy["mmcifPath"] = relative_path
        templates.append(template_copy)

    return TargetFeatures(
        unpaired_msa_path=unpaired_path,
        paired_msa_path=paired_path,
        templates=templates,
    )
