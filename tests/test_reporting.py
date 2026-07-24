from __future__ import annotations

import csv
from pathlib import Path

from af3_binder_filter.output_layout import (
    OUTPUT_SCHEMA_VERSION,
    STAGE_DIRECTORIES,
    RunOutputLayout,
)
from af3_binder_filter.reporting import (
    BACKEND_REVIEW_COLUMNS,
    DECISION_COLUMNS,
    PUBLIC_COLUMNS,
    REVIEW_ONLY_COLUMNS,
    write_public_reports,
)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def test_v3_column_contract_is_exact_and_disjoint() -> None:
    assert OUTPUT_SCHEMA_VERSION == 3
    assert len(DECISION_COLUMNS) == 55
    assert len(REVIEW_ONLY_COLUMNS) == 53
    assert len(BACKEND_REVIEW_COLUMNS) == 108
    assert PUBLIC_COLUMNS == DECISION_COLUMNS
    assert BACKEND_REVIEW_COLUMNS == DECISION_COLUMNS + REVIEW_ONLY_COLUMNS
    assert len(set(BACKEND_REVIEW_COLUMNS)) == len(BACKEND_REVIEW_COLUMNS)
    assert "primary_iptm" not in DECISION_COLUMNS
    assert "secondary_iptm" not in DECISION_COLUMNS
    assert "consensus_interface_pair_jaccard" not in DECISION_COLUMNS
    assert "effective_iptm" in DECISION_COLUMNS
    assert "primary_iptm" in REVIEW_ONLY_COLUMNS
    assert "secondary_iptm" in REVIEW_ONLY_COLUMNS


def test_decision_and_review_reports_have_v3_schemas_and_normalized_contacts(
    tmp_path: Path,
) -> None:
    layout = RunOutputLayout(tmp_path).ensure()
    rows = [
        {
            "job_name": "job_1",
            "sample_no": "1",
            "run_name": "screen",
            "source_row_number": 2,
            "target_chain": "T",
            "binder_chain": "X",
            "target_sequence": "AAAA",
            "binder_sequence": "CCCC",
            "backend": "alphafold3",
            "job_status": "success",
            "best_model_path": "/run/primary.cif",
            "final_pass": False,
            "target_interface_residues": "1,2",
            "binder_interface_residues": "3,4",
            "interface_residue_pairs": "1:3,2:4",
            "epitope_residues": "1,2,3",
            "epitope_overlap_residues": "1",
            "primary_interface_status": "success",
            "primary_iptm": 0.72,
            "secondary_backend": "opendde",
            "secondary_status": "success",
            "secondary_best_model_path": "/run/secondary.cif",
            "secondary_interface_status": "success",
            "secondary_target_interface_residues": "2,3",
            "secondary_binder_interface_residues": "4,5",
            "secondary_interface_residue_pairs": "2:4,3:5",
            "secondary_epitope_overlap_residues": "2,3",
            "secondary_iptm": 0.81,
            "secondary_gate_pass": True,
            "secondary_final_pass": True,
            "candidate_pool": True,
            "manual_review": True,
            "manual_review_reason": "secondary_rescue",
            "consensus_status": "success",
            "consensus_interface_pair_jaccard": 0.25,
            "effective_backend": "opendde",
            "effective_selection_reason": "quality:pass",
            "effective_status": "success",
            "effective_pass": True,
            "effective_best_model_path": "/run/secondary.cif",
            "effective_interface_status": "success",
            "effective_target_interface_residues": "2,3",
            "effective_binder_interface_residues": "4,5",
            "effective_interface_residue_pairs": "2:4,3:5",
            "effective_epitope_overlap_residues": "2,3",
            "effective_iptm": 0.81,
            "esmfold_effective_binder_tm": 0.74,
            "esmfold_primary_binder_tm": 0.61,
            "esmfold_secondary_binder_tm": 0.74,
        },
        {
            "job_name": "job_2",
            "target_chain": "T",
            "binder_chain": "X",
            "final_pass": False,
            "candidate_pool": False,
            "effective_backend": None,
            "effective_selection_reason": "no_eligible_backend",
            "effective_pass": None,
        },
    ]
    members = [
        {
            "job_name": "job_1",
            "binder_cluster": "binder_0001",
            "complex_cluster": "complex_0001",
            "epitope_cluster": "epitope_0001",
        }
    ]
    representatives = [
        {
            "layer": layer,
            "cluster_id": f"{layer}_0001",
            "member_count": 1,
            "quality_representative": "job_1",
        }
        for layer in ("binder", "complex", "epitope")
    ]

    returned = write_public_reports(
        layout,
        rows,
        member_rows=members,
        representative_rows=representatives,
        final_job_ids=("job_1",),
        clustering_status="success",
    )

    assert len(returned) == 3
    all_header, all_rows = _read(layout.all_results)
    candidate_header, candidate_rows = _read(layout.candidates)
    final_header, final_rows = _read(layout.final_shortlist)
    review_header, review_rows = _read(layout.backend_review)
    assert all_header == candidate_header == final_header == list(DECISION_COLUMNS)
    assert review_header == list(BACKEND_REVIEW_COLUMNS)
    assert len(all_rows) == len(review_rows) == 2
    assert [row["job_id"] for row in candidate_rows] == ["job_1"]
    assert [row["job_id"] for row in final_rows] == ["job_1"]

    assert all_rows[0]["effective_backend"] == "opendde"
    assert all_rows[0]["effective_target_interface_residues"] == "T:2;T:3"
    assert all_rows[0]["effective_binder_interface_residues"] == "X:4;X:5"
    assert all_rows[0]["effective_interface_residue_pairs"] == ("T:2-X:4;T:3-X:5")
    assert all_rows[0]["configured_epitope_residues"] == "T:1;T:2;T:3"
    assert all_rows[0]["diversity_cell_id"] == ("binder_0001|complex_0001|epitope_0001")
    assert all_rows[1]["effective_iptm"] == ""
    assert all_rows[1]["effective_pass"] == ""

    assert review_rows[0]["primary_target_interface_residues"] == "T:1;T:2"
    assert review_rows[0]["secondary_target_interface_residues"] == "T:2;T:3"
    assert review_rows[0]["primary_interface_residue_pairs"] == ("T:1-X:3;T:2-X:4")
    assert review_rows[0]["secondary_interface_residue_pairs"] == ("T:2-X:4;T:3-X:5")
    assert review_rows[0]["consensus_interface_pair_jaccard"] == "0.25"
    assert review_rows[0]["esmfold_primary_binder_tm"] == "0.61"
    assert review_rows[0]["esmfold_secondary_binder_tm"] == "0.74"
    assert "purity" not in "\n".join(review_header).lower()


def test_layout_creates_all_numbered_stage_folders(tmp_path: Path) -> None:
    assert list(STAGE_DIRECTORIES.values()) == [
        "01_preflight",
        "02_features",
        "03_primary_prediction",
        "04_primary_interface",
        "05_secondary_features",
        "06_secondary_prediction",
        "07_secondary_interface",
        "08_consensus",
        "09_esm",
        "10_clustering",
    ]
    layout = RunOutputLayout(tmp_path).ensure()
    assert {path.name for path in layout.stages_root.iterdir()} == set(STAGE_DIRECTORIES.values())
    for name in STAGE_DIRECTORIES:
        stage = layout.stage(name)
        assert stage.logs.is_dir()
        assert stage.tables.is_dir()
        assert stage.artifacts.is_dir()
