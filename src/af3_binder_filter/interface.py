"""Biotite interface geometry, epitope scoring, and Rosetta input conversion."""

from __future__ import annotations

import json
import math
import os
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.derived_structures import (
    SourceModelChangedError,
    file_sha256,
    materialize_derived_structures,
)
from af3_binder_filter.io_utils import atomic_write_csv
from af3_binder_filter.jobs import JobSpec, parse_epitope_residues
from af3_binder_filter.residue_format import (
    format_contact_pairs,
    format_residue_list,
)

STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)

_AMINO_ACID_ONE_LETTER = {
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


class InterfaceError(RuntimeError):
    """Raised when a predicted complex cannot be analyzed geometrically."""


@dataclass(frozen=True, slots=True)
class ResidueRecord:
    chain_id: str
    original_res_id: int
    original_ins_code: str
    sequence_position: int
    res_name: str
    atom_indexes: np.ndarray
    mapping_mode: str


def load_protein_complex(path: Path, *, target_chain: str, binder_chain: str):
    try:
        import biotite.structure.io as strucio
    except ImportError as exc:
        raise InterfaceError("biotite is required for interface analysis") from exc
    if not path.is_file():
        raise InterfaceError(f"structure file does not exist: {path}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            array = strucio.load_structure(
                str(path),
                model=1,
                extra_fields=["b_factor"],
            )
        except Exception as exc:
            try:
                array = strucio.load_structure(str(path), model=1)
            except Exception:
                raise InterfaceError(f"failed to parse structure {path}: {exc}") from exc
    if getattr(array, "stack_depth", lambda: 1)() > 1:
        array = array[0]
    chain_mask = (array.chain_id == target_chain) | (array.chain_id == binder_chain)
    standard_mask = np.isin(array.res_name, list(STANDARD_AMINO_ACIDS))
    array = array[chain_mask & standard_mask]
    present = set(str(chain) for chain in array.chain_id.tolist())
    for chain in (target_chain, binder_chain):
        if chain not in present:
            raise InterfaceError(f"protein chain {chain!r} not found in {path}")
    return array


def structure_has_chains(path: Path, target_chain: str, binder_chain: str) -> bool:
    try:
        load_protein_complex(path, target_chain=target_chain, binder_chain=binder_chain)
        return True
    except InterfaceError:
        return False


def _annotation(array: Any, name: str, default: Any) -> np.ndarray:
    try:
        return np.asarray(array.get_annotation(name))
    except Exception:
        return np.full(array.array_length(), default)


def _exact_subsequence_positions(observed: str, expected: str) -> list[int]:
    """Return a unique exact-subsequence mapping, rejecting ambiguity."""

    earliest: list[int] = []
    cursor = 0
    for residue in observed:
        found = expected.find(residue, cursor)
        if found < 0:
            raise InterfaceError(
                "structure sequence is not an exact subsequence of the input sequence"
            )
        earliest.append(found + 1)
        cursor = found + 1

    latest_reversed: list[int] = []
    cursor = len(expected)
    for residue in reversed(observed):
        found = expected.rfind(residue, 0, cursor)
        if found < 0:
            raise InterfaceError(
                "structure sequence is not an exact subsequence of the input sequence"
            )
        latest_reversed.append(found + 1)
        cursor = found
    latest = list(reversed(latest_reversed))
    if earliest != latest:
        raise InterfaceError("missing-residue mapping is ambiguous against the input sequence")
    return earliest


def _chain_sequence_positions(
    *,
    observed_sequence: str,
    expected_sequence: str,
    author_residue_ids: list[int],
) -> tuple[list[int], str]:
    expected = expected_sequence.strip().upper()
    if not expected:
        raise InterfaceError("input sequence is empty")
    if not observed_sequence:
        raise InterfaceError("structure chain has no standard residues")
    if len(author_residue_ids) != len(observed_sequence):
        raise InterfaceError("structure residue identifiers do not match its residue count")
    if len(observed_sequence) > len(expected):
        raise InterfaceError("structure contains more standard residues than the input sequence")

    author_positions_valid = (
        len(set(author_residue_ids)) == len(author_residue_ids)
        and all(position > 0 for position in author_residue_ids)
        and author_residue_ids == sorted(author_residue_ids)
        and all(position <= len(expected) for position in author_residue_ids)
        and all(
            observed == expected[position - 1]
            for observed, position in zip(
                observed_sequence,
                author_residue_ids,
                strict=True,
            )
        )
    )
    if author_positions_valid:
        return author_residue_ids, "author_residue_ids"
    if len(observed_sequence) == len(expected):
        if observed_sequence != expected:
            raise InterfaceError("structure sequence does not match the input sequence")
        return list(range(1, len(expected) + 1)), "complete_sequence_order"
    return (
        _exact_subsequence_positions(observed_sequence, expected),
        "unique_exact_subsequence",
    )


def _residue_records(
    array: Any,
    chain_id: str,
    expected_sequence: str,
) -> list[ResidueRecord]:
    import biotite.structure as struc

    chain_indexes = np.where(array.chain_id == chain_id)[0]
    chain_array = array[chain_indexes]
    starts = struc.get_residue_starts(chain_array, add_exclusive_stop=True)
    insertion_codes = _annotation(chain_array, "ins_code", "")
    residue_ranges = list(zip(starts[:-1], starts[1:], strict=True))
    author_ids = [int(chain_array.res_id[start]) for start, _stop in residue_ranges]
    residue_names = [str(chain_array.res_name[start]) for start, _stop in residue_ranges]
    try:
        observed_sequence = "".join(_AMINO_ACID_ONE_LETTER[name] for name in residue_names)
    except KeyError as exc:
        raise InterfaceError(f"unsupported residue {exc.args[0]!r} in chain {chain_id!r}") from exc
    try:
        positions, mapping_mode = _chain_sequence_positions(
            observed_sequence=observed_sequence,
            expected_sequence=expected_sequence,
            author_residue_ids=author_ids,
        )
    except InterfaceError as exc:
        raise InterfaceError(f"chain {chain_id!r}: {exc}") from exc
    if len(positions) != len(set(positions)) or any(position <= 0 for position in positions):
        raise InterfaceError(
            f"chain {chain_id!r} did not map to unique positive sequence positions"
        )

    records: list[ResidueRecord] = []
    for (start, stop), res_id, res_name, sequence_position in zip(
        residue_ranges,
        author_ids,
        residue_names,
        positions,
        strict=True,
    ):
        records.append(
            ResidueRecord(
                chain_id=chain_id,
                original_res_id=res_id,
                original_ins_code=str(insertion_codes[start]).strip(),
                sequence_position=sequence_position,
                res_name=res_name,
                atom_indexes=chain_indexes[start:stop],
                mapping_mode=mapping_mode,
            )
        )
    return records


def _heavy_mask(array: Any) -> np.ndarray:
    element = np.char.upper(_annotation(array, "element", "").astype(str))
    atom_name = np.char.upper(array.atom_name.astype(str))
    return (element != "H") & ~np.char.startswith(atom_name, "H")


def _contact_metrics(
    array: Any,
    target_records: list[ResidueRecord],
    binder_records: list[ResidueRecord],
    *,
    distance: float,
) -> tuple[set[tuple[int, int]], float | None, np.ndarray]:
    heavy = _heavy_mask(array)
    target_atom_indexes = np.concatenate([record.atom_indexes for record in target_records])
    binder_atom_indexes = np.concatenate([record.atom_indexes for record in binder_records])
    target_atom_indexes = target_atom_indexes[heavy[target_atom_indexes]]
    binder_atom_indexes = binder_atom_indexes[heavy[binder_atom_indexes]]
    if target_atom_indexes.size == 0 or binder_atom_indexes.size == 0:
        raise InterfaceError("target or binder has no polymer heavy atoms")

    target_position_by_atom = {
        int(index): record.sequence_position
        for record in target_records
        for index in record.atom_indexes
    }
    binder_position_by_atom = {
        int(index): record.sequence_position
        for record in binder_records
        for index in record.atom_indexes
    }
    contacts: set[tuple[int, int]] = set()
    minimum = math.inf
    interface_atom_mask = np.zeros(array.array_length(), dtype=bool)
    target_coord = array.coord[target_atom_indexes]
    binder_coord = array.coord[binder_atom_indexes]
    # Chunking avoids materializing an unbounded N_target x N_binder matrix.
    chunk_size = max(1, min(512, len(target_coord)))
    for start in range(0, len(target_coord), chunk_size):
        stop = min(len(target_coord), start + chunk_size)
        delta = target_coord[start:stop, None, :] - binder_coord[None, :, :]
        distances = np.linalg.norm(delta, axis=-1)
        if distances.size:
            minimum = min(minimum, float(np.min(distances)))
        close_i, close_j = np.where(distances <= distance)
        for local_i, binder_i in zip(close_i.tolist(), close_j.tolist(), strict=True):
            target_index = int(target_atom_indexes[start + local_i])
            binder_index = int(binder_atom_indexes[binder_i])
            contacts.add(
                (
                    target_position_by_atom[target_index],
                    binder_position_by_atom[binder_index],
                )
            )
            interface_atom_mask[target_index] = True
            interface_atom_mask[binder_index] = True
    # Interface confidence is residue-based: include every atom belonging to a
    # residue that participates in at least one heavy-atom contact.
    target_contact_positions = {target for target, _binder in contacts}
    binder_contact_positions = {binder for _target, binder in contacts}
    for record in target_records:
        if record.sequence_position in target_contact_positions:
            interface_atom_mask[record.atom_indexes] = True
    for record in binder_records:
        if record.sequence_position in binder_contact_positions:
            interface_atom_mask[record.atom_indexes] = True
    return contacts, (minimum if math.isfinite(minimum) else None), interface_atom_mask


def _sasa_metrics(
    array: Any, target_chain: str, binder_chain: str, point_number: int
) -> dict[str, float]:
    import biotite.structure as struc

    target = array[array.chain_id == target_chain]
    binder = array[array.chain_id == binder_chain]
    complex_values = struc.sasa(array, point_number=point_number)
    target_values = struc.sasa(target, point_number=point_number)
    binder_values = struc.sasa(binder, point_number=point_number)
    sasa_complex = float(np.nansum(complex_values))
    sasa_target = float(np.nansum(target_values))
    sasa_binder = float(np.nansum(binder_values))
    bsa_total = sasa_target + sasa_binder - sasa_complex
    complex_target = float(np.nansum(complex_values[array.chain_id == target_chain]))
    complex_binder = float(np.nansum(complex_values[array.chain_id == binder_chain]))
    return {
        "biotite_sasa_target": sasa_target,
        "biotite_sasa_binder": sasa_binder,
        "biotite_sasa_complex": sasa_complex,
        "biotite_bsa_target": sasa_target - complex_target,
        "biotite_bsa_binder": sasa_binder - complex_binder,
        "biotite_bsa_total": bsa_total,
        # Compatibility aliases retain their established Biotite semantics.
        "sasa_target": sasa_target,
        "sasa_binder": sasa_binder,
        "sasa_complex": sasa_complex,
        "bsa": bsa_total,
        "bsa_interface": bsa_total,
    }


def _plddt_metrics(array: Any, interface_atom_mask: np.ndarray) -> dict[str, float | None]:
    b_factor = _annotation(array, "b_factor", np.nan).astype(float)
    values = b_factor[interface_atom_mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"interface_plddt_mean": None, "interface_plddt_min": None}
    if float(np.nanmax(values)) <= 1.0:
        values = values * 100.0
    return {
        "interface_plddt_mean": float(np.mean(values)),
        "interface_plddt_min": float(np.min(values)),
    }


def _pae_metrics(
    confidence_path: Path | None,
    *,
    job: JobSpec,
    target_positions: set[int],
    binder_positions: set[int],
) -> dict[str, float | None]:
    empty = {
        "interface_pae_target_to_binder_mean": None,
        "interface_pae_binder_to_target_mean": None,
        "interface_pae_mean": None,
        "ipae_A_to_B_mean": None,
        "ipae_B_to_A_mean": None,
    }
    if confidence_path is None or not confidence_path.is_file():
        return empty
    try:
        data = json.loads(confidence_path.read_text(encoding="utf-8"))
        pae_value = (
            data.get("pae") or data.get("predicted_aligned_error") or data.get("token_pair_pae")
        )
        pae = np.asarray(pae_value, dtype=float)
        if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
            return empty
        chains = (
            data.get("token_chain_ids") or data.get("token_asym_ids") or data.get("token_asym_id")
        )
        residue_ids = (
            data.get("token_res_ids") or data.get("token_residue_ids") or data.get("token_res_id")
        )
        if chains is None and pae.shape[0] == len(job.target_sequence) + len(job.binder_sequence):
            chains = [job.target_chain] * len(job.target_sequence) + [job.binder_chain] * len(
                job.binder_sequence
            )
            residue_ids = list(range(1, len(job.target_sequence) + 1)) + list(
                range(1, len(job.binder_sequence) + 1)
            )
        if chains is None:
            return empty
        chains_array = np.asarray(chains).astype(str)
        if not np.any(chains_array == job.target_chain) or not np.any(
            chains_array == job.binder_chain
        ):
            # Protenix/OpenDDE confidence JSON uses numeric asym IDs.  Protein
            # chains follow input order, which is target then binder.
            unique_asym_ids = list(dict.fromkeys(chains_array.tolist()))
            if len(unique_asym_ids) != 2:
                return empty
            chains_array = np.where(
                chains_array == unique_asym_ids[0],
                job.target_chain,
                job.binder_chain,
            )
        if residue_ids is None:
            residue_ids_array = np.zeros(len(chains_array), dtype=int)
            for chain in (job.target_chain, job.binder_chain):
                mask_indexes = np.where(chains_array == chain)[0]
                residue_ids_array[mask_indexes] = np.arange(1, len(mask_indexes) + 1)
        else:
            residue_ids_array = np.asarray(residue_ids, dtype=int)
        target_mask = (chains_array == job.target_chain) & np.isin(
            residue_ids_array, sorted(target_positions)
        )
        binder_mask = (chains_array == job.binder_chain) & np.isin(
            residue_ids_array, sorted(binder_positions)
        )
        ab = pae[np.ix_(target_mask, binder_mask)]
        ba = pae[np.ix_(binder_mask, target_mask)]
        if ab.size == 0 or ba.size == 0:
            return empty
        ab_mean = float(np.mean(ab))
        ba_mean = float(np.mean(ba))
        return {
            "interface_pae_target_to_binder_mean": ab_mean,
            "interface_pae_binder_to_target_mean": ba_mean,
            "interface_pae_mean": (ab_mean + ba_mean) / 2,
            "ipae_A_to_B_mean": ab_mean,
            "ipae_B_to_A_mean": ba_mean,
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return empty


def write_rosetta_pdb(
    array: Any,
    *,
    target_records: list[ResidueRecord],
    binder_records: list[ResidueRecord],
    pdb_path: Path,
    residue_map_path: Path,
) -> None:
    import biotite.structure.io as strucio

    records = target_records + binder_records
    keep_indexes = np.concatenate([record.atom_indexes for record in records])
    converted = array[keep_indexes].copy()
    cursor = 0
    map_rows: list[dict[str, Any]] = []
    for record in records:
        count = len(record.atom_indexes)
        converted.res_id[cursor : cursor + count] = record.sequence_position
        if "ins_code" in converted.get_annotation_categories():
            converted.ins_code[cursor : cursor + count] = ""
        if "hetero" in converted.get_annotation_categories():
            converted.hetero[cursor : cursor + count] = False
        cursor += count
        map_rows.append(
            {
                "pdb_chain": record.chain_id,
                "pdb_residue_number": record.sequence_position,
                "original_chain": record.chain_id,
                "original_res_id": record.original_res_id,
                "original_ins_code": record.original_ins_code,
                "sequence_position": record.sequence_position,
                "res_name": record.res_name,
            }
        )
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{pdb_path.name}.", suffix=".pdb", dir=pdb_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        strucio.save_structure(str(temporary_path), converted)
        os.replace(temporary_path, pdb_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    atomic_write_csv(
        residue_map_path,
        map_rows,
        fieldnames=[
            "pdb_chain",
            "pdb_residue_number",
            "original_chain",
            "original_res_id",
            "original_ins_code",
            "sequence_position",
            "res_name",
        ],
        delimiter="\t",
    )


def analyze_interface_geometry(
    job: JobSpec,
    prediction: UnifiedPrediction,
    *,
    distance: float = 5.0,
    epitope_residues: str | None = None,
    sasa_point_number: int = 1000,
    rosetta_input_dir: Path | None = None,
    derived_structure_dir: Path | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "job_name": job.job_id,
        "interface_status": "error",
        "interface_error": "",
        "derived_structure_status": "not_available",
        "derived_structure_error": "",
        "source_model_provenance_status": "not_checked",
        "source_model_sha256_preparse": "",
        "source_model_sha256_observed": "",
    }
    if prediction.status != "success" or prediction.best_model_path is None:
        result["interface_error"] = prediction.error or "prediction is not successful"
        return result
    try:
        source_model_sha256 = file_sha256(prediction.best_model_path)
        result["source_model_sha256_preparse"] = source_model_sha256
        array = load_protein_complex(
            prediction.best_model_path,
            target_chain=job.target_chain,
            binder_chain=job.binder_chain,
        )
        observed_source_sha256 = file_sha256(prediction.best_model_path)
        result["source_model_sha256_observed"] = observed_source_sha256
        if observed_source_sha256 != source_model_sha256:
            raise SourceModelChangedError(
                prediction.best_model_path,
                expected_sha256=source_model_sha256,
                observed_sha256=observed_source_sha256,
                phase="while it was being parsed",
            )
        result["source_model_provenance_status"] = "verified"
        target_records = _residue_records(
            array,
            job.target_chain,
            job.target_sequence,
        )
        binder_records = _residue_records(
            array,
            job.binder_chain,
            job.binder_sequence,
        )
        contacts, minimum_distance, interface_atom_mask = _contact_metrics(
            array,
            target_records,
            binder_records,
            distance=distance,
        )
        target_positions = {target for target, _binder in contacts}
        binder_positions = {binder for _target, binder in contacts}
        epitope = parse_epitope_residues(
            epitope_residues,
            target_length=len(job.target_sequence),
        )
        overlap = target_positions & epitope
        union = target_positions | epitope
        result.update(
            {
                "interface_status": "success",
                "interface_error": "",
                "interface_distance_cutoff": distance,
                "interface_contact_pair_count": len(contacts),
                "interface_minimum_distance": minimum_distance,
                "target_interface_residue_count": len(target_positions),
                "binder_interface_residue_count": len(binder_positions),
                "target_interface_residues": format_residue_list(
                    job.target_chain, target_positions
                ),
                "binder_interface_residues": format_residue_list(
                    job.binder_chain, binder_positions
                ),
                "interface_residue_pairs": format_contact_pairs(
                    job.target_chain, job.binder_chain, contacts
                ),
                "epitope_residues": format_residue_list(job.target_chain, epitope),
                "epitope_overlap_residues": format_residue_list(job.target_chain, overlap),
                "epitope_overlap_count": len(overlap),
                "epitope_coverage": len(overlap) / len(epitope) if epitope else None,
                "epitope_purity": len(overlap) / len(target_positions)
                if epitope and target_positions
                else None,
                "epitope_jaccard": len(overlap) / len(union) if epitope and union else None,
            }
        )
        result.update(_plddt_metrics(array, interface_atom_mask))
        result.update(
            _pae_metrics(
                prediction.confidence_path,
                job=job,
                target_positions=target_positions,
                binder_positions=binder_positions,
            )
        )
        try:
            result.update(
                _sasa_metrics(
                    array,
                    target_chain=job.target_chain,
                    binder_chain=job.binder_chain,
                    point_number=sasa_point_number,
                )
            )
            result["sasa_status"] = "success"
            result["sasa_error"] = ""
        except Exception as exc:
            result["sasa_status"] = "error"
            result["sasa_error"] = str(exc)
        artifact_root = derived_structure_dir
        if artifact_root is None and rosetta_input_dir is not None:
            artifact_root = rosetta_input_dir.parent / "derived_structures"
        if artifact_root is not None:
            try:
                derived = materialize_derived_structures(
                    array,
                    source_model_path=prediction.best_model_path,
                    expected_source_model_sha256=source_model_sha256,
                    target_chain=job.target_chain,
                    binder_chain=job.binder_chain,
                    target_sequence=job.target_sequence,
                    binder_sequence=job.binder_sequence,
                    interface_distance_cutoff=distance,
                    target_records=target_records,
                    binder_records=binder_records,
                    artifacts_root=artifact_root,
                    backend=prediction.backend,
                    job_id=job.job_id,
                    target_interface_positions=target_positions,
                    binder_interface_positions=binder_positions,
                )
                result.update(derived.as_row())
                # Rosetta consumes the same normalized AB structure; these
                # aliases preserve the existing interface-stage contract.
                result["rosetta_input_pdb"] = str(derived.complex_pdb)
                result["residue_map_path"] = str(derived.residue_map)
            except SourceModelChangedError as exc:
                # Geometry belongs to the stable in-memory parse, but the
                # backend must be ineligible once its on-disk provenance moves.
                result["interface_status"] = "error"
                result["interface_error"] = str(exc)
                result["source_model_provenance_status"] = "changed"
                result["source_model_sha256_observed"] = exc.observed_sha256
                result["derived_structure_status"] = "error"
                result["derived_structure_error"] = str(exc)
            except Exception as exc:
                # Geometry was already computed from the parsed structure.
                # Derivation is a downstream reuse optimization and must not
                # erase valid contact/SASA/PAE results.
                result["derived_structure_status"] = "error"
                result["derived_structure_error"] = str(exc)
        else:
            result["derived_structure_status"] = "not_requested"
    except SourceModelChangedError as exc:
        result["interface_status"] = "error"
        result["interface_error"] = str(exc)
        result["source_model_provenance_status"] = "changed"
        result["source_model_sha256_preparse"] = exc.expected_sha256
        result["source_model_sha256_observed"] = exc.observed_sha256
        result["derived_structure_status"] = "error"
        result["derived_structure_error"] = str(exc)
    except Exception as exc:
        result["interface_status"] = "error"
        result["interface_error"] = str(exc)
    return result


def apply_balanced_shortlist(
    rows: Iterable[dict[str, Any]],
    *,
    minimum_contact_pairs: int = 5,
    epitope_configured: bool = False,
    minimum_epitope_coverage: float = 0.30,
    minimum_epitope_purity: float | None = None,
) -> list[dict[str, Any]]:
    # Kept in the signature so older YAML remains loadable.  Purity is an
    # annotation only in output schema v2 and never participates in selection.
    _ = minimum_epitope_purity
    materialized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        geometry_pass = (
            row.get("interface_status") == "success"
            and int(row.get("interface_contact_pair_count") or 0) >= minimum_contact_pairs
        )
        epitope_pass = True
        if epitope_configured:
            coverage_pass = float(row.get("epitope_coverage") or 0) >= minimum_epitope_coverage
            epitope_pass = coverage_pass
        row["geometry_pass"] = geometry_pass
        row["epitope_pass"] = epitope_pass
        row["final_pass"] = geometry_pass and epitope_pass
        materialized.append(row)

    def descending(value: Any) -> float:
        try:
            number = float(value)
            return -number if math.isfinite(number) else math.inf
        except (TypeError, ValueError):
            return math.inf

    def ascending(value: Any) -> float:
        try:
            number = float(value)
            return number if math.isfinite(number) else math.inf
        except (TypeError, ValueError):
            return math.inf

    materialized.sort(
        key=lambda row: (
            not bool(row["final_pass"]),
            descending(row.get("epitope_coverage")),
            ascending(row.get("interface_pae_mean")),
            ascending(row.get("rosetta_dG_separated_per_dSASA_x100")),
            descending(row.get("rosetta_packstat")),
            descending(row.get("iptm")),
            descending(row.get("ranking_score")),
            str(row.get("job_name", "")),
        )
    )
    return materialized
