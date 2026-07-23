"""AF3/secondary epitope-and-pose consensus metrics and review flags."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from af3_binder_filter.clustering import jaccard, parse_residue_set
from af3_binder_filter.config import ConsensusSettings
from af3_binder_filter.derived_structures import (
    DerivedStructureArtifacts,
    validated_artifacts_from_row,
)
from af3_binder_filter.interface import load_protein_complex
from af3_binder_filter.residue_format import parse_contact_pairs


def _ca_coordinate_map(array: Any, chain_id: str) -> dict[int, np.ndarray]:
    import biotite.structure as struc

    chain = array[array.chain_id == chain_id]
    if chain.array_length() == 0:
        raise ValueError(f"chain {chain_id!r} has no coordinates")
    starts = struc.get_residue_starts(chain, add_exclusive_stop=True)
    categories = set(chain.get_annotation_categories())
    coordinates: dict[int, np.ndarray] = {}
    ordered_positions: list[int] = []
    for start, stop in zip(starts[:-1], starts[1:], strict=True):
        residue = chain[start:stop]
        position = int(residue.res_id[0])
        if position <= 0 or position in coordinates:
            raise ValueError(
                f"chain {chain_id!r} residue IDs must be unique positive positions"
            )
        if "ins_code" in categories and any(
            str(value).strip() for value in residue.ins_code.tolist()
        ):
            raise ValueError(
                f"chain {chain_id!r} residue {position} has an insertion code"
            )
        ca = residue[residue.atom_name == "CA"]
        coordinate = np.asarray(
            ca.coord[0] if len(ca) else residue.coord.mean(axis=0),
            dtype=float,
        )
        if coordinate.shape != (3,) or not np.all(np.isfinite(coordinate)):
            raise ValueError(
                f"chain {chain_id!r} residue {position} has invalid coordinates"
            )
        coordinates[position] = coordinate
        ordered_positions.append(position)
    if ordered_positions != sorted(ordered_positions):
        raise ValueError(f"chain {chain_id!r} residue IDs are not increasing")
    return coordinates


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


def _derived_coordinate_maps(
    artifacts: DerivedStructureArtifacts,
) -> dict[str, dict[int, np.ndarray]]:
    """Load a coordinate bundle already validated with its manifest."""

    with np.load(artifacts.coordinates, allow_pickle=False) as payload:
        coordinates = np.asarray(payload["ca_coord"], dtype=float)
        chains = payload["chain_id"].astype(str)
        positions = payload["sequence_position"].astype(int)
    result: dict[str, dict[int, np.ndarray]] = {}
    for chain, position, coordinate in zip(
        chains.tolist(),
        positions.tolist(),
        coordinates,
        strict=True,
    ):
        result.setdefault(chain, {})[position] = coordinate
    return result


def _paired_coordinate_arrays(
    fixed: Mapping[int, np.ndarray],
    moving: Mapping[int, np.ndarray],
    positions: frozenset[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    common = set(fixed) & set(moving)
    if positions is not None:
        common &= set(positions)
    ordered = tuple(sorted(common))
    return (
        np.asarray(
            [fixed[position] for position in ordered],
            dtype=float,
        ).reshape((-1, 3)),
        np.asarray(
            [moving[position] for position in ordered],
            dtype=float,
        ).reshape((-1, 3)),
        ordered,
    )


def _consensus_from_coordinate_maps(
    primary: Mapping[str, Mapping[int, np.ndarray]],
    secondary: Mapping[str, Mapping[int, np.ndarray]],
    *,
    target_chain: str,
    binder_chain: str,
    primary_target_contacts: frozenset[int],
    secondary_target_contacts: frozenset[int],
    primary_binder_contacts: frozenset[int],
    secondary_binder_contacts: frozenset[int],
    settings: ConsensusSettings,
    coordinate_source: str = "derived_cache",
) -> dict[str, Any]:
    target_primary, target_secondary, target_positions = _paired_coordinate_arrays(
        primary[target_chain],
        secondary[target_chain],
    )
    rotation, translation, kept = _robust_target_transform(
        target_secondary,
        target_primary,
        settings,
    )
    aligned_target_secondary = target_secondary @ rotation + translation

    binder_primary, binder_secondary, binder_positions = _paired_coordinate_arrays(
        primary[binder_chain],
        secondary[binder_chain],
    )
    if len(binder_primary) < 3:
        raise ValueError(
            f"only {len(binder_primary)} common binder residues are available"
        )
    aligned_binder_secondary = binder_secondary @ rotation + translation

    common_binder_interface = (
        primary_binder_contacts & secondary_binder_contacts
    )
    primary_interface, secondary_interface, _interface_positions = (
        _paired_coordinate_arrays(
            primary[binder_chain],
            secondary[binder_chain],
            common_binder_interface,
        )
    )
    secondary_interface = secondary_interface @ rotation + translation

    independent_rotation, independent_translation = _kabsch(
        binder_secondary,
        binder_primary,
    )
    independent_aligned = (
        binder_secondary @ independent_rotation + independent_translation
    )
    independent_distances = np.linalg.norm(
        binder_primary - independent_aligned,
        axis=1,
    )

    primary_target = primary[target_chain]
    secondary_target = secondary[target_chain]
    primary_binder = primary[binder_chain]
    secondary_binder = secondary[binder_chain]
    target_index = {position: index for index, position in enumerate(target_positions)}
    binder_index = {position: index for index, position in enumerate(binder_positions)}
    common_target = primary_target_contacts & secondary_target_contacts
    common_binder = primary_binder_contacts & secondary_binder_contacts
    pair_deltas: list[float] = []
    for target_position in sorted(common_target):
        if (
            target_position not in primary_target
            or target_position not in secondary_target
            or target_position not in target_index
        ):
            continue
        for binder_position in sorted(common_binder):
            if (
                binder_position not in primary_binder
                or binder_position not in secondary_binder
                or binder_position not in binder_index
            ):
                continue
            first = np.linalg.norm(
                primary_target[target_position] - primary_binder[binder_position]
            )
            second = np.linalg.norm(
                aligned_target_secondary[target_index[target_position]]
                - aligned_binder_secondary[binder_index[binder_position]]
            )
            if min(first, second) <= 15.0:
                pair_deltas.append(abs(float(first - second)))
    interface_lddt = None
    if pair_deltas:
        values = np.asarray(pair_deltas)
        interface_lddt = float(
            np.mean(
                [
                    (values < threshold).mean()
                    for threshold in (0.5, 1.0, 2.0, 4.0)
                ]
            )
        )
    return {
        "consensus_status": "success",
        "consensus_target_alignment_residues": int(kept.sum()),
        "consensus_target_alignment_rmsd": _rmsd(
            target_primary[kept],
            aligned_target_secondary[kept],
        ),
        "consensus_binder_fixed_frame_rmsd": _rmsd(
            binder_primary,
            aligned_binder_secondary,
        ),
        "consensus_interface_fixed_frame_rmsd": _rmsd(
            primary_interface,
            secondary_interface,
        ),
        "consensus_binder_center_displacement": (
            float(
                np.linalg.norm(
                    binder_primary.mean(axis=0)
                    - aligned_binder_secondary.mean(axis=0)
                )
            )
            if len(binder_primary)
            else None
        ),
        "consensus_interface_lddt": interface_lddt,
        "consensus_binder_fold_rmsd": _rmsd(
            binder_primary,
            independent_aligned,
        ),
        "consensus_binder_fold_tm": _tm_score(
            independent_distances,
            len(binder_primary),
        ),
        "consensus_epitope_jaccard": jaccard(
            primary_target_contacts,
            secondary_target_contacts,
        ),
        "consensus_error": "",
        "consensus_coordinate_source": coordinate_source,
    }


def structure_consensus_metrics_from_rows(
    primary_row: Mapping[str, Any],
    secondary_row: Mapping[str, Any],
    *,
    target_chain: str,
    binder_chain: str,
    primary_target_contacts: frozenset[int],
    secondary_target_contacts: frozenset[int],
    primary_binder_contacts: frozenset[int],
    secondary_binder_contacts: frozenset[int],
    settings: ConsensusSettings,
    primary_prefix: str = "",
    secondary_prefix: str = "",
) -> dict[str, Any]:
    """Use validated NPZ coordinates, falling back safely to source models."""

    primary_artifacts = validated_artifacts_from_row(
        primary_row,
        prefix=primary_prefix,
        require_declared=True,
    )
    secondary_artifacts = validated_artifacts_from_row(
        secondary_row,
        prefix=secondary_prefix,
        require_declared=True,
    )
    if primary_artifacts is not None and secondary_artifacts is not None:
        return _consensus_from_coordinate_maps(
            _derived_coordinate_maps(primary_artifacts),
            _derived_coordinate_maps(secondary_artifacts),
            target_chain=target_chain,
            binder_chain=binder_chain,
            primary_target_contacts=primary_target_contacts,
            secondary_target_contacts=secondary_target_contacts,
            primary_binder_contacts=primary_binder_contacts,
            secondary_binder_contacts=secondary_binder_contacts,
            settings=settings,
        )

    def row_path(row: Mapping[str, Any], prefix: str) -> Path:
        key = f"{prefix}_best_model_path" if prefix else "best_model_path"
        return Path(str(row[key]))

    result = structure_consensus_metrics(
        row_path(primary_row, primary_prefix),
        row_path(secondary_row, secondary_prefix),
        target_chain=target_chain,
        binder_chain=binder_chain,
        primary_target_contacts=primary_target_contacts,
        secondary_target_contacts=secondary_target_contacts,
        primary_binder_contacts=primary_binder_contacts,
        secondary_binder_contacts=secondary_binder_contacts,
        settings=settings,
    )
    result["consensus_coordinate_source"] = "raw_structure_fallback"
    return result


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
    return _consensus_from_coordinate_maps(
        {
            target_chain: _ca_coordinate_map(primary, target_chain),
            binder_chain: _ca_coordinate_map(primary, binder_chain),
        },
        {
            target_chain: _ca_coordinate_map(secondary, target_chain),
            binder_chain: _ca_coordinate_map(secondary, binder_chain),
        },
        target_chain=target_chain,
        binder_chain=binder_chain,
        primary_target_contacts=primary_target_contacts,
        secondary_target_contacts=secondary_target_contacts,
        primary_binder_contacts=primary_binder_contacts,
        secondary_binder_contacts=secondary_binder_contacts,
        settings=settings,
        coordinate_source="raw_structure",
    )


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
        for key, value in primary.items():
            row[f"primary_{key}"] = value
        if secondary is None:
            row.update({"secondary_status": "not_selected", "consensus_status": "not_available"})
            merged.append(row)
            continue
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
            metrics = structure_consensus_metrics_from_rows(
                primary,
                secondary,
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
