"""AF3/secondary epitope-and-pose consensus metrics and review flags."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from af3_binder_filter.clustering import jaccard, parse_residue_set
from af3_binder_filter.config import ConsensusSettings
from af3_binder_filter.interface import load_protein_complex
from af3_binder_filter.residue_format import parse_contact_pairs


def _ca_coordinates(array: Any, chain_id: str) -> np.ndarray:
    import biotite.structure as struc

    chain = array[array.chain_id == chain_id]
    starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    coordinates: list[np.ndarray] = []
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        residue = chain[start:stop]
        ca = residue[residue.atom_name == "CA"]
        coordinates.append(
            np.asarray(ca.coord[0] if len(ca) else residue.coord.mean(axis=0), dtype=float)
        )
    return np.asarray(coordinates, dtype=float)


def _kabsch(moving: np.ndarray, fixed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    moving_center = moving.mean(axis=0)
    fixed_center = fixed.mean(axis=0)
    covariance = (moving - moving_center).T @ (fixed - fixed_center)
    left, _values, right = np.linalg.svd(covariance)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    translation = fixed_center - moving_center @ rotation
    return rotation, translation


def _rmsd(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) == 0 or left.shape != right.shape:
        return None
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))


def _tm_score(distances: np.ndarray, length: int) -> float | None:
    if length < 1 or len(distances) == 0:
        return None
    d0 = max(0.5, 1.24 * max(length - 15, 1) ** (1 / 3) - 1.8)
    return float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / length)


def _robust_target_transform(
    moving: np.ndarray,
    fixed: np.ndarray,
    settings: ConsensusSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(len(moving), len(fixed))
    if count < settings.target_alignment_min_residues:
        raise ValueError(f"only {count} target residues are available for alignment")
    moving = moving[:count]
    fixed = fixed[:count]
    keep = np.ones(count, dtype=bool)
    minimum = max(
        settings.target_alignment_min_residues,
        int(math.ceil(count * settings.target_alignment_min_fraction)),
    )
    for _ in range(settings.target_alignment_max_iterations):
        rotation, translation = _kabsch(moving[keep], fixed[keep])
        distances = np.linalg.norm(moving @ rotation + translation - fixed, axis=1)
        median = float(np.median(distances[keep]))
        mad = float(np.median(np.abs(distances[keep] - median)))
        threshold = median + max(0.5, 3.0 * 1.4826 * mad)
        updated = distances <= threshold
        if updated.sum() < minimum or np.array_equal(updated, keep):
            break
        keep = updated
    rotation, translation = _kabsch(moving[keep], fixed[keep])
    return rotation, translation, keep


def _selected(coords: np.ndarray, residues: frozenset[int]) -> np.ndarray:
    indexes = [value - 1 for value in sorted(residues) if 1 <= value <= len(coords)]
    return coords[indexes] if indexes else np.empty((0, 3), dtype=float)


def structure_consensus_metrics(
    primary_path: Path,
    secondary_path: Path,
    *,
    target_chain: str,
    binder_chain: str,
    primary_target_contacts: frozenset[int],
    secondary_target_contacts: frozenset[int],
    primary_binder_contacts: frozenset[int],
    secondary_binder_contacts: frozenset[int],
    settings: ConsensusSettings,
) -> dict[str, Any]:
    primary = load_protein_complex(
        primary_path, target_chain=target_chain, binder_chain=binder_chain
    )
    secondary = load_protein_complex(
        secondary_path, target_chain=target_chain, binder_chain=binder_chain
    )
    target_primary = _ca_coordinates(primary, target_chain)
    target_secondary = _ca_coordinates(secondary, target_chain)
    binder_primary = _ca_coordinates(primary, binder_chain)
    binder_secondary = _ca_coordinates(secondary, binder_chain)
    rotation, translation, kept = _robust_target_transform(
        target_secondary, target_primary, settings
    )
    target_count = min(len(target_primary), len(target_secondary))
    aligned_target_secondary = target_secondary[:target_count] @ rotation + translation
    aligned_binder_secondary = binder_secondary @ rotation + translation
    binder_count = min(len(binder_primary), len(aligned_binder_secondary))
    fixed_distances = np.linalg.norm(
        binder_primary[:binder_count] - aligned_binder_secondary[:binder_count], axis=1
    )
    common_binder_interface = primary_binder_contacts & secondary_binder_contacts
    primary_interface = _selected(binder_primary, common_binder_interface)
    secondary_interface = _selected(aligned_binder_secondary, common_binder_interface)

    independent_count = min(len(binder_primary), len(binder_secondary))
    independent_rotation, independent_translation = _kabsch(
        binder_secondary[:independent_count], binder_primary[:independent_count]
    )
    independent_aligned = (
        binder_secondary[:independent_count] @ independent_rotation
        + independent_translation
    )
    independent_distances = np.linalg.norm(
        binder_primary[:independent_count] - independent_aligned, axis=1
    )

    common_target = primary_target_contacts & secondary_target_contacts
    common_binder = primary_binder_contacts & secondary_binder_contacts
    pair_deltas: list[float] = []
    for target_residue in sorted(common_target):
        if target_residue > target_count:
            continue
        for binder_residue in sorted(common_binder):
            if binder_residue > binder_count:
                continue
            first = np.linalg.norm(
                target_primary[target_residue - 1] - binder_primary[binder_residue - 1]
            )
            second = np.linalg.norm(
                aligned_target_secondary[target_residue - 1]
                - aligned_binder_secondary[binder_residue - 1]
            )
            if min(first, second) <= 15.0:
                pair_deltas.append(abs(float(first - second)))
    interface_lddt = None
    if pair_deltas:
        values = np.asarray(pair_deltas)
        interface_lddt = float(
            np.mean([(values < threshold).mean() for threshold in (0.5, 1.0, 2.0, 4.0)])
        )
    return {
        "consensus_status": "success",
        "consensus_target_alignment_residues": int(kept.sum()),
        "consensus_target_alignment_rmsd": _rmsd(
            target_primary[:target_count][kept], aligned_target_secondary[kept]
        ),
        "consensus_binder_fixed_frame_rmsd": _rmsd(
            binder_primary[:binder_count], aligned_binder_secondary[:binder_count]
        ),
        "consensus_interface_fixed_frame_rmsd": _rmsd(
            primary_interface, secondary_interface
        ),
        "consensus_binder_center_displacement": float(
            np.linalg.norm(
                binder_primary[:binder_count].mean(axis=0)
                - aligned_binder_secondary[:binder_count].mean(axis=0)
            )
        ) if binder_count else None,
        "consensus_interface_lddt": interface_lddt,
        "consensus_binder_fold_rmsd": _rmsd(
            binder_primary[:independent_count], independent_aligned
        ),
        "consensus_binder_fold_tm": _tm_score(
            independent_distances, independent_count
        ),
        "consensus_epitope_jaccard": jaccard(
            primary_target_contacts, secondary_target_contacts
        ),
        "consensus_error": "",
    }


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _pair_set(value: Any) -> frozenset[tuple[int, int]]:
    return parse_contact_pairs(value)


def _set_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def add_anomaly_flags(
    rows: list[dict[str, Any]], settings: ConsensusSettings
) -> list[dict[str, Any]]:
    metrics = (
        "consensus_binder_fixed_frame_rmsd",
        "consensus_interface_fixed_frame_rmsd",
        "consensus_binder_center_displacement",
        "consensus_epitope_disagreement",
        "consensus_fold_disagreement",
    )
    cohorts: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        cohorts.setdefault(str(row.get("secondary_backend", "none")), []).append(row)
    for cohort, members in cohorts.items():
        usable = [row for row in members if row.get("consensus_status") == "success"]
        statistics: dict[str, tuple[float, float]] = {}
        if settings.anomaly_detection and len(usable) >= settings.anomaly_min_samples:
            for metric in metrics:
                values = [_number(row.get(metric)) for row in usable]
                array = np.asarray([value for value in values if value is not None])
                if len(array):
                    median = float(np.median(array))
                    mad = float(np.median(np.abs(array - median)))
                    statistics[metric] = (median, mad)
        for row in members:
            flags: list[str] = []
            z_values: dict[str, float] = {}
            for metric, (median, mad) in statistics.items():
                value = _number(row.get(metric))
                if value is None:
                    continue
                if mad == 0:
                    z = math.inf if value != median else 0.0
                else:
                    z = abs(value - median) / (1.4826 * mad)
                z_values[metric] = z
                if z > settings.robust_z_threshold:
                    flags.append(metric)
            explicit = (
                len(parse_residue_set(row.get("primary_target_interface_residues")))
                >= settings.minimum_contact_residues_for_epitope_flag
                and len(parse_residue_set(row.get("secondary_target_interface_residues")))
                >= settings.minimum_contact_residues_for_epitope_flag
                and (_number(row.get("consensus_epitope_jaccard")) or 0.0)
                < settings.explicit_different_epitope_jaccard
            )
            pair_jaccard = _number(
                row.get("consensus_interface_pair_jaccard")
            )
            different_contact_pairs = (
                pair_jaccard is not None
                and pair_jaccard
                < settings.explicit_different_interface_pair_jaccard
            )
            fold_tm = _number(row.get("consensus_binder_fold_tm"))
            fixed_frame_rmsd = _number(
                row.get("consensus_binder_fixed_frame_rmsd")
            )
            different_fold = (
                fold_tm is not None
                and fold_tm < settings.same_fold_tm_threshold
            )
            different_pose = (
                fixed_frame_rmsd is not None
                and fixed_frame_rmsd > settings.different_pose_rmsd_threshold
            )
            row["consensus_anomaly_metric_count"] = len(flags)
            row["consensus_anomaly_metrics"] = ";".join(flags)
            row["consensus_robust_z"] = ";".join(
                f"{key}={value:.3g}" for key, value in sorted(z_values.items())
            )
            row["consensus_explicit_different_epitope"] = explicit
            row["consensus_different_interface_pairs"] = different_contact_pairs
            row["consensus_different_binder_fold"] = different_fold
            row["consensus_different_pose"] = different_pose
            robust_anomaly = len(flags) >= settings.minimum_anomaly_metrics
            row["manual_review"] = (
                explicit
                or different_contact_pairs
                or different_fold
                or different_pose
                or robust_anomaly
            )
            review_reasons: list[str] = []
            if explicit:
                review_reasons.append("different_epitope")
            if different_contact_pairs:
                review_reasons.append("contact_pair_disagreement")
            if different_fold:
                review_reasons.append("different_binder_fold")
            if different_pose:
                review_reasons.append("different_pose")
            if robust_anomaly:
                review_reasons.append("robust_multimetric_anomaly")
            row["manual_review_reason"] = ";".join(review_reasons)
            row["consensus_cohort"] = cohort
            row["consensus_cohort_size"] = len(usable)
    return rows


def consensus_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    secondary_rows: Sequence[Mapping[str, Any]],
    settings: ConsensusSettings,
) -> list[dict[str, Any]]:
    secondary_by_job = {str(row["job_name"]): row for row in secondary_rows}
    merged: list[dict[str, Any]] = []
    for primary in primary_rows:
        job_name = str(primary["job_name"])
        secondary = secondary_by_job.get(job_name)
        row = dict(primary)
        if secondary is None:
            row.update({"secondary_status": "not_selected", "consensus_status": "not_available"})
            merged.append(row)
            continue
        for key, value in primary.items():
            row[f"primary_{key}"] = value
        for key, value in secondary.items():
            row[f"secondary_{key}"] = value
        row["secondary_backend"] = secondary.get("backend")
        row["secondary_status"] = secondary.get("job_status")
        primary_contacts = parse_residue_set(primary.get("target_interface_residues"))
        secondary_contacts = parse_residue_set(secondary.get("target_interface_residues"))
        row["consensus_epitope_disagreement"] = 1.0 - jaccard(
            primary_contacts, secondary_contacts
        )
        row["consensus_interface_pair_jaccard"] = _set_jaccard(
            _pair_set(primary.get("interface_residue_pairs")),
            _pair_set(secondary.get("interface_residue_pairs")),
        )
        try:
            metrics = structure_consensus_metrics(
                Path(str(primary["best_model_path"])),
                Path(str(secondary["best_model_path"])),
                target_chain=str(primary["target_chain"]),
                binder_chain=str(primary["binder_chain"]),
                primary_target_contacts=primary_contacts,
                secondary_target_contacts=secondary_contacts,
                primary_binder_contacts=parse_residue_set(primary.get("binder_interface_residues")),
                secondary_binder_contacts=parse_residue_set(secondary.get("binder_interface_residues")),
                settings=settings,
            )
            row.update(metrics)
            tm = _number(row.get("consensus_binder_fold_tm"))
            row["consensus_fold_disagreement"] = 1.0 - tm if tm is not None else None
        except Exception as exc:
            row.update({"consensus_status": "error", "consensus_error": str(exc)})
        merged.append(row)
    return add_anomaly_flags(merged, settings)
