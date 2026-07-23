"""Versioned decision and backend-review CSV projections for Aerith runs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from af3_binder_filter.effective import apply_effective_backend
from af3_binder_filter.io_utils import atomic_write_csv
from af3_binder_filter.output_layout import RunOutputLayout
from af3_binder_filter.residue_format import (
    normalize_contact_pairs,
    normalize_residue_list,
)


DECISION_COLUMNS: tuple[str, ...] = (
    "job_id",
    "sample_no",
    "run_name",
    "source_row_number",
    "target_chain",
    "binder_chain",
    "target_sequence",
    "binder_sequence",
    "configured_epitope_residues",
    "candidate_pass",
    "selection_reasons",
    "manual_review",
    "manual_review_reason",
    "consensus_status",
    "effective_backend",
    "effective_selection_reason",
    "effective_status",
    "effective_pass",
    "effective_ranking_score",
    "effective_iptm",
    "effective_ptm",
    "effective_plddt_global_mean",
    "effective_best_model_path",
    "effective_interface_status",
    "effective_interface_contact_pair_count",
    "effective_target_interface_residues",
    "effective_binder_interface_residues",
    "effective_interface_residue_pairs",
    "effective_epitope_overlap_residues",
    "effective_epitope_overlap_count",
    "effective_epitope_coverage",
    "effective_interface_pae_mean",
    "effective_biotite_bsa_total",
    "effective_rosetta_status",
    "effective_rosetta_dG_separated_per_dSASA_x100",
    "effective_rosetta_packstat",
    "esmfold_status",
    "esmfold_plddt",
    "esmfold_effective_binder_tm",
    "esm_if_status",
    "esm_if_log_likelihood",
    "esm_if_perplexity",
    "clustering_status",
    "binder_cluster_id",
    "binder_cluster_size",
    "is_binder_quality_representative",
    "complex_cluster_id",
    "complex_cluster_size",
    "is_complex_quality_representative",
    "epitope_cluster_id",
    "epitope_cluster_size",
    "is_epitope_quality_representative",
    "diversity_cell_id",
    "is_final_representative",
    "final_rank",
)

_BACKEND_FIELDS: tuple[str, ...] = (
    "backend",
    "ranking_score",
    "iptm",
    "ptm",
    "plddt_global_mean",
    "best_model_path",
    "interface_status",
    "interface_contact_pair_count",
    "target_interface_residues",
    "binder_interface_residues",
    "interface_residue_pairs",
    "epitope_overlap_residues",
    "epitope_overlap_count",
    "epitope_coverage",
    "interface_pae_mean",
    "biotite_bsa_total",
    "rosetta_status",
    "rosetta_dG_separated_per_dSASA_x100",
    "rosetta_packstat",
)

_PRIMARY_REVIEW_COLUMNS: tuple[str, ...] = (
    "primary_backend",
    "primary_status",
    *(f"primary_{field}" for field in _BACKEND_FIELDS if field != "backend"),
)
_SECONDARY_REVIEW_COLUMNS: tuple[str, ...] = (
    "secondary_backend",
    "secondary_status",
    *(f"secondary_{field}" for field in _BACKEND_FIELDS if field != "backend"),
)

REVIEW_ONLY_COLUMNS: tuple[str, ...] = (
    "primary_pass",
    "secondary_gate_pass",
    "secondary_pass",
    *_PRIMARY_REVIEW_COLUMNS,
    *_SECONDARY_REVIEW_COLUMNS,
    "consensus_target_alignment_rmsd",
    "consensus_binder_fixed_frame_rmsd",
    "consensus_interface_fixed_frame_rmsd",
    "consensus_interface_lddt",
    "consensus_binder_fold_tm",
    "consensus_epitope_jaccard",
    "consensus_interface_pair_jaccard",
    "consensus_different_pose",
    "esmfold_primary_binder_tm",
    "esmfold_secondary_binder_tm",
)

BACKEND_REVIEW_COLUMNS: tuple[str, ...] = DECISION_COLUMNS + REVIEW_ONLY_COLUMNS

# Compatibility alias: callers importing PUBLIC_COLUMNS now receive the compact
# decision schema used by all_results/candidates/final_shortlist.
PUBLIC_COLUMNS: tuple[str, ...] = DECISION_COLUMNS


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _optional_bool(value: Any) -> bool | None:
    return None if value in (None, "") else _truthy(value)


def _value(row: Mapping[str, Any], prefix: str, field: str) -> Any:
    key = f"{prefix}_{field}"
    if key in row:
        return row.get(key)
    return row.get(field) if prefix == "primary" else None


def _backend_projection(
    row: Mapping[str, Any], prefix: str, target_chain: str, binder_chain: str
) -> dict[str, Any]:
    projected = {
        f"{prefix}_{field}": _value(row, prefix, field)
        for field in _BACKEND_FIELDS
    }
    projected[f"{prefix}_status"] = (
        row.get(f"{prefix}_job_status")
        if f"{prefix}_job_status" in row
        else row.get(
            f"{prefix}_status",
            row.get("job_status", row.get("status"))
            if prefix == "primary"
            else None,
        )
    )
    for field, chain in (
        ("target_interface_residues", target_chain),
        ("binder_interface_residues", binder_chain),
        ("epitope_overlap_residues", target_chain),
    ):
        value = projected.get(f"{prefix}_{field}")
        projected[f"{prefix}_{field}"] = (
            normalize_residue_list(value, chain) if value not in (None, "") else ""
        )
    pair_value = projected.get(f"{prefix}_interface_residue_pairs")
    projected[f"{prefix}_interface_residue_pairs"] = (
        normalize_contact_pairs(pair_value, target_chain, binder_chain)
        if pair_value not in (None, "")
        else ""
    )
    return projected


def _selection_reasons(row: Mapping[str, Any]) -> str:
    explicit = row.get("selection_reasons")
    if explicit not in (None, ""):
        return str(explicit)
    primary_pass = _truthy(row.get("primary_final_pass", row.get("final_pass")))
    secondary_gate = _truthy(row.get("secondary_gate_pass"))
    secondary_pass = _truthy(row.get("secondary_final_pass"))
    candidate = _truthy(row.get("candidate_pool", row.get("candidate_pass")))
    reasons: list[str] = []
    if primary_pass:
        reasons.append("primary_pass")
    if secondary_pass:
        reasons.append("secondary_pass")
    if secondary_pass and not primary_pass:
        reasons.append("secondary_rescue")
    if not candidate:
        secondary_backend = row.get("secondary_backend")
        if row.get("secondary_status") == "not_selected" or (
            secondary_backend not in (None, "", "none") and not secondary_gate
        ):
            reasons.append("secondary_not_gated")
        elif secondary_gate and row.get("secondary_status") not in (
            "success",
            None,
            "",
        ):
            reasons.append("secondary_failed")
        elif not primary_pass:
            geometry = row.get("primary_geometry_pass", row.get("geometry_pass"))
            epitope = row.get("primary_epitope_pass", row.get("epitope_pass"))
            if not _truthy(geometry):
                reasons.append("primary_geometry_failed")
            elif not _truthy(epitope):
                reasons.append("primary_coverage_failed")
            else:
                reasons.append("selection_failed")
    return ";".join(dict.fromkeys(reasons))


def _cluster_annotations(
    member_rows: Sequence[Mapping[str, Any]],
    representative_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    memberships: dict[str, dict[str, Any]] = {}
    sizes: dict[str, Counter[str]] = {
        "binder": Counter(),
        "complex": Counter(),
        "epitope": Counter(),
    }
    for row in member_rows:
        job_id = str(row.get("job_name", row.get("job_id", "")))
        current = memberships.setdefault(job_id, {})
        for layer in sizes:
            cluster_id = row.get(f"{layer}_cluster")
            current[f"{layer}_cluster_id"] = cluster_id
            if cluster_id not in (None, ""):
                sizes[layer][str(cluster_id)] += 1
    quality = {
        (str(row.get("layer", "")), str(row.get("cluster_id", ""))): str(
            row.get("quality_representative", "")
        )
        for row in representative_rows
    }
    for job_id, current in memberships.items():
        for layer in sizes:
            cluster_id = current.get(f"{layer}_cluster_id")
            current[f"{layer}_cluster_size"] = (
                sizes[layer][str(cluster_id)] if cluster_id not in (None, "") else None
            )
            current[f"is_{layer}_quality_representative"] = (
                quality.get((layer, str(cluster_id))) == job_id
                if cluster_id not in (None, "")
                else False
            )
        cell = tuple(current.get(f"{layer}_cluster_id") for layer in sizes)
        current["diversity_cell_id"] = (
            "|".join(str(value) for value in cell)
            if all(value not in (None, "") for value in cell)
            else ""
        )
    return memberships


def _effective_source(source: Mapping[str, Any]) -> Mapping[str, Any]:
    """Add the effective projection when an upstream caller has not done so."""

    if "effective_backend" in source:
        return source
    return apply_effective_backend(source)


def _esmfold_tm(source: Mapping[str, Any], prefix: str) -> Any:
    key = f"esmfold_{prefix}_binder_tm"
    value = source.get(key)
    if value not in (None, ""):
        return value
    if prefix == "primary":
        return source.get("esmfold_af3_binder_tm")
    return None


def _common_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    member_rows: Sequence[Mapping[str, Any]] = (),
    representative_rows: Sequence[Mapping[str, Any]] = (),
    final_job_ids: Sequence[str] = (),
    clustering_status: str = "not_run",
) -> list[dict[str, Any]]:
    cluster_by_job = _cluster_annotations(member_rows, representative_rows)
    normalized_final_ids = tuple(str(job_id) for job_id in final_job_ids)
    final_rank = {
        job_id: index for index, job_id in enumerate(normalized_final_ids, start=1)
    }
    common: list[dict[str, Any]] = []
    for raw_source in rows:
        source = _effective_source(raw_source)
        job_id = str(source.get("job_name", source.get("job_id", "")))
        target_chain = str(source.get("target_chain") or "A")
        binder_chain = str(source.get("binder_chain") or "B")
        configured = _value(source, "primary", "epitope_residues")
        primary = _backend_projection(source, "primary", target_chain, binder_chain)
        secondary = _backend_projection(source, "secondary", target_chain, binder_chain)
        effective = _backend_projection(source, "effective", target_chain, binder_chain)
        effective_tm = source.get("esmfold_effective_binder_tm")
        if effective_tm in (None, ""):
            effective_backend = source.get("effective_backend")
            primary_backend = primary.get("primary_backend")
            if effective_backend not in (None, "") and effective_backend == primary_backend:
                effective_tm = _esmfold_tm(source, "primary")
            elif effective_backend not in (None, ""):
                effective_tm = _esmfold_tm(source, "secondary")
        row: dict[str, Any] = {
            "job_id": job_id,
            "sample_no": source.get("sample_no"),
            "run_name": source.get("run_name"),
            "source_row_number": source.get("source_row_number"),
            "target_chain": target_chain,
            "binder_chain": binder_chain,
            "target_sequence": source.get("target_sequence"),
            "binder_sequence": source.get("binder_sequence"),
            "configured_epitope_residues": (
                normalize_residue_list(configured, target_chain)
                if configured not in (None, "")
                else ""
            ),
            "candidate_pass": _truthy(
                source.get("candidate_pool", source.get("candidate_pass"))
            ),
            "selection_reasons": _selection_reasons(source),
            "manual_review": _truthy(source.get("manual_review")),
            "manual_review_reason": source.get("manual_review_reason", ""),
            "consensus_status": source.get("consensus_status"),
            "effective_selection_reason": source.get(
                "effective_selection_reason"
            ),
            "effective_pass": _optional_bool(source.get("effective_pass")),
            "esmfold_status": source.get("esmfold_status"),
            "esmfold_plddt": source.get("esmfold_plddt"),
            "esmfold_effective_binder_tm": effective_tm,
            "esm_if_status": source.get("esm_if_status"),
            "esm_if_log_likelihood": source.get("esm_if_log_likelihood"),
            "esm_if_perplexity": source.get("esm_if_perplexity"),
            "clustering_status": source.get(
                "clustering_status", clustering_status
            ),
            "primary_pass": _truthy(
                source.get("primary_final_pass", source.get("final_pass"))
            ),
            "secondary_gate_pass": _truthy(source.get("secondary_gate_pass")),
            "secondary_pass": _truthy(source.get("secondary_final_pass")),
            "consensus_target_alignment_rmsd": source.get(
                "consensus_target_alignment_rmsd"
            ),
            "consensus_binder_fixed_frame_rmsd": source.get(
                "consensus_binder_fixed_frame_rmsd"
            ),
            "consensus_interface_fixed_frame_rmsd": source.get(
                "consensus_interface_fixed_frame_rmsd"
            ),
            "consensus_interface_lddt": source.get("consensus_interface_lddt"),
            "consensus_binder_fold_tm": source.get("consensus_binder_fold_tm"),
            "consensus_epitope_jaccard": source.get(
                "consensus_epitope_jaccard"
            ),
            "consensus_interface_pair_jaccard": source.get(
                "consensus_interface_pair_jaccard"
            ),
            "consensus_different_pose": source.get("consensus_different_pose"),
            "esmfold_primary_binder_tm": _esmfold_tm(source, "primary"),
            "esmfold_secondary_binder_tm": _esmfold_tm(source, "secondary"),
            **primary,
            **secondary,
            **effective,
            **cluster_by_job.get(job_id, {}),
            "is_final_representative": job_id in final_rank,
            "final_rank": final_rank.get(job_id),
        }
        common.append(row)
    return common


def _project_rows(
    rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> list[dict[str, Any]]:
    return [{column: row.get(column, "") for column in columns} for row in rows]


def build_public_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    member_rows: Sequence[Mapping[str, Any]] = (),
    representative_rows: Sequence[Mapping[str, Any]] = (),
    final_job_ids: Sequence[str] = (),
    clustering_status: str = "not_run",
) -> list[dict[str, Any]]:
    """Build the compact decision projection used by the three public CSVs."""

    common = _common_rows(
        rows,
        member_rows=member_rows,
        representative_rows=representative_rows,
        final_job_ids=final_job_ids,
        clustering_status=clustering_status,
    )
    return _project_rows(common, DECISION_COLUMNS)


def build_backend_review_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    member_rows: Sequence[Mapping[str, Any]] = (),
    representative_rows: Sequence[Mapping[str, Any]] = (),
    final_job_ids: Sequence[str] = (),
    clustering_status: str = "not_run",
) -> list[dict[str, Any]]:
    """Build the full primary/secondary/effective audit projection."""

    common = _common_rows(
        rows,
        member_rows=member_rows,
        representative_rows=representative_rows,
        final_job_ids=final_job_ids,
        clustering_status=clustering_status,
    )
    return _project_rows(common, BACKEND_REVIEW_COLUMNS)


def write_public_reports(
    layout: RunOutputLayout,
    rows: Sequence[Mapping[str, Any]],
    *,
    member_rows: Sequence[Mapping[str, Any]] = (),
    representative_rows: Sequence[Mapping[str, Any]] = (),
    final_job_ids: Sequence[str] = (),
    clustering_status: str = "not_run",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Write three decision CSVs plus the complete backend-review CSV.

    The historical three-item return value is intentionally preserved for
    workflow callers; ``backend_review.csv`` is an additional on-disk audit
    artifact and contains every input job.
    """

    common = _common_rows(
        rows,
        member_rows=member_rows,
        representative_rows=representative_rows,
        final_job_ids=final_job_ids,
        clustering_status=clustering_status,
    )
    public = _project_rows(common, DECISION_COLUMNS)
    review = _project_rows(common, BACKEND_REVIEW_COLUMNS)
    candidates = [row for row in public if row["candidate_pass"]]
    final_by_id = {str(row["job_id"]): row for row in public}
    final = [
        final_by_id[str(job_id)]
        for job_id in final_job_ids
        if str(job_id) in final_by_id
    ]
    atomic_write_csv(layout.all_results, public, fieldnames=DECISION_COLUMNS)
    atomic_write_csv(layout.candidates, candidates, fieldnames=DECISION_COLUMNS)
    atomic_write_csv(layout.final_shortlist, final, fieldnames=DECISION_COLUMNS)
    atomic_write_csv(
        layout.backend_review,
        review,
        fieldnames=BACKEND_REVIEW_COLUMNS,
    )
    return public, candidates, final
