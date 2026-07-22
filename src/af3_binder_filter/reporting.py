"""Versioned, compact public CSV projection for Aerith runs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from af3_binder_filter.io_utils import atomic_write_csv
from af3_binder_filter.output_layout import RunOutputLayout
from af3_binder_filter.residue_format import (
    normalize_contact_pairs,
    normalize_residue_list,
)


PUBLIC_COLUMNS: tuple[str, ...] = (
    "job_id", "sample_no", "run_name", "source_row_number",
    "target_chain", "binder_chain", "target_sequence", "binder_sequence",
    "configured_epitope_residues",
    "primary_pass", "secondary_gate_pass", "secondary_pass", "candidate_pass",
    "selection_reasons", "manual_review", "manual_review_reason",
    "primary_backend", "primary_status", "primary_ranking_score", "primary_iptm",
    "primary_ptm", "primary_plddt_global_mean", "primary_best_model_path",
    "primary_interface_status", "primary_interface_contact_pair_count",
    "primary_target_interface_residues", "primary_binder_interface_residues",
    "primary_interface_residue_pairs", "primary_epitope_overlap_residues",
    "primary_epitope_overlap_count", "primary_epitope_coverage",
    "primary_interface_pae_mean", "primary_biotite_bsa_total",
    "primary_rosetta_status", "primary_rosetta_dG_separated_per_dSASA_x100",
    "primary_rosetta_packstat",
    "secondary_backend", "secondary_status", "secondary_ranking_score",
    "secondary_iptm", "secondary_ptm", "secondary_plddt_global_mean",
    "secondary_best_model_path", "secondary_interface_status",
    "secondary_interface_contact_pair_count", "secondary_target_interface_residues",
    "secondary_binder_interface_residues", "secondary_interface_residue_pairs",
    "secondary_epitope_overlap_residues", "secondary_epitope_overlap_count",
    "secondary_epitope_coverage", "secondary_interface_pae_mean",
    "secondary_biotite_bsa_total", "secondary_rosetta_status",
    "secondary_rosetta_dG_separated_per_dSASA_x100", "secondary_rosetta_packstat",
    "esmfold_status", "esmfold_plddt", "esmfold_af3_binder_tm",
    "esm_if_status", "esm_if_log_likelihood", "esm_if_perplexity",
    "consensus_status", "consensus_target_alignment_rmsd",
    "consensus_binder_fixed_frame_rmsd", "consensus_interface_fixed_frame_rmsd",
    "consensus_interface_lddt", "consensus_binder_fold_tm",
    "consensus_epitope_jaccard", "consensus_different_pose",
    "clustering_status", "binder_cluster_id", "binder_cluster_size",
    "is_binder_quality_representative", "complex_cluster_id", "complex_cluster_size",
    "is_complex_quality_representative", "epitope_cluster_id", "epitope_cluster_size",
    "is_epitope_quality_representative", "diversity_cell_id",
    "is_final_representative", "final_rank",
)

_BACKEND_FIELDS: tuple[str, ...] = (
    "backend", "ranking_score", "iptm", "ptm", "plddt_global_mean",
    "best_model_path", "interface_status", "interface_contact_pair_count",
    "target_interface_residues", "binder_interface_residues",
    "interface_residue_pairs", "epitope_overlap_residues",
    "epitope_overlap_count", "epitope_coverage", "interface_pae_mean",
    "biotite_bsa_total", "rosetta_status",
    "rosetta_dG_separated_per_dSASA_x100", "rosetta_packstat",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


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
            row.get("job_status") if prefix == "primary" else None,
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
    primary_pass = _truthy(row.get("primary_final_pass", row.get("final_pass")))
    secondary_gate = _truthy(row.get("secondary_gate_pass"))
    secondary_pass = _truthy(row.get("secondary_final_pass"))
    candidate = _truthy(row.get("candidate_pool"))
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
        elif secondary_gate and row.get("secondary_status") not in ("success", None, ""):
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
        "binder": Counter(), "complex": Counter(), "epitope": Counter()
    }
    for row in member_rows:
        job_id = str(row.get("job_name", ""))
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
                if cluster_id not in (None, "") else False
            )
        cell = tuple(current.get(f"{layer}_cluster_id") for layer in sizes)
        current["diversity_cell_id"] = (
            "|".join(str(value) for value in cell)
            if all(value not in (None, "") for value in cell) else ""
        )
    return memberships


def build_public_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    member_rows: Sequence[Mapping[str, Any]] = (),
    representative_rows: Sequence[Mapping[str, Any]] = (),
    final_job_ids: Sequence[str] = (),
    clustering_status: str = "not_run",
) -> list[dict[str, Any]]:
    cluster_by_job = _cluster_annotations(member_rows, representative_rows)
    final_rank = {job_id: index for index, job_id in enumerate(final_job_ids, start=1)}
    public: list[dict[str, Any]] = []
    for source in rows:
        job_id = str(source.get("job_name", source.get("job_id", "")))
        target_chain = str(source.get("target_chain", "A"))
        binder_chain = str(source.get("binder_chain", "B"))
        configured = _value(source, "primary", "epitope_residues")
        row: dict[str, Any] = {
            "job_id": job_id,
            "sample_no": source.get("sample_no"),
            "run_name": source.get("run_name"),
            "source_row_number": source.get("source_row_number"),
            "target_chain": target_chain,
            "binder_chain": binder_chain,
            "target_sequence": source.get("target_sequence"),
            "binder_sequence": source.get("binder_sequence"),
            "configured_epitope_residues": normalize_residue_list(
                configured, target_chain
            ) if configured not in (None, "") else "",
            "primary_pass": _truthy(
                source.get("primary_final_pass", source.get("final_pass"))
            ),
            "secondary_gate_pass": _truthy(source.get("secondary_gate_pass")),
            "secondary_pass": _truthy(source.get("secondary_final_pass")),
            "candidate_pass": _truthy(source.get("candidate_pool")),
            "selection_reasons": _selection_reasons(source),
            "manual_review": _truthy(source.get("manual_review")),
            "manual_review_reason": source.get("manual_review_reason", ""),
            "esmfold_status": source.get("esmfold_status"),
            "esmfold_plddt": source.get("esmfold_plddt"),
            "esmfold_af3_binder_tm": source.get("esmfold_af3_binder_tm"),
            "esm_if_status": source.get("esm_if_status"),
            "esm_if_log_likelihood": source.get("esm_if_log_likelihood"),
            "esm_if_perplexity": source.get("esm_if_perplexity"),
            "consensus_status": source.get("consensus_status"),
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
            "consensus_epitope_jaccard": source.get("consensus_epitope_jaccard"),
            "consensus_different_pose": source.get("consensus_different_pose"),
            "clustering_status": clustering_status,
            **_backend_projection(source, "primary", target_chain, binder_chain),
            **_backend_projection(source, "secondary", target_chain, binder_chain),
            **cluster_by_job.get(job_id, {}),
            "is_final_representative": job_id in final_rank,
            "final_rank": final_rank.get(job_id),
        }
        public.append({column: row.get(column, "") for column in PUBLIC_COLUMNS})
    return public


def write_public_reports(
    layout: RunOutputLayout,
    rows: Sequence[Mapping[str, Any]],
    *,
    member_rows: Sequence[Mapping[str, Any]] = (),
    representative_rows: Sequence[Mapping[str, Any]] = (),
    final_job_ids: Sequence[str] = (),
    clustering_status: str = "not_run",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    public = build_public_rows(
        rows,
        member_rows=member_rows,
        representative_rows=representative_rows,
        final_job_ids=final_job_ids,
        clustering_status=clustering_status,
    )
    candidates = [row for row in public if row["candidate_pass"]]
    final_by_id = {str(row["job_id"]): row for row in public}
    final = [final_by_id[job_id] for job_id in final_job_ids if job_id in final_by_id]
    atomic_write_csv(layout.all_results, public, fieldnames=PUBLIC_COLUMNS)
    atomic_write_csv(layout.candidates, candidates, fieldnames=PUBLIC_COLUMNS)
    atomic_write_csv(layout.final_shortlist, final, fieldnames=PUBLIC_COLUMNS)
    return public, candidates, final
