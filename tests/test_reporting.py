from __future__ import annotations

import csv
from pathlib import Path

from af3_binder_filter.output_layout import RunOutputLayout, STAGE_DIRECTORIES
from af3_binder_filter.reporting import PUBLIC_COLUMNS, write_public_reports


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or ()), list(reader)


def test_public_reports_have_one_schema_and_chain_qualified_contacts(
    tmp_path: Path,
) -> None:
    layout = RunOutputLayout(tmp_path).ensure()
    rows = [
        {
            "job_name": "job_1",
            "sample_no": "1",
            "run_name": "screen",
            "source_row_number": 2,
            "target_chain": "A",
            "binder_chain": "B",
            "target_sequence": "AAAA",
            "binder_sequence": "CCCC",
            "backend": "alphafold3",
            "job_status": "success",
            "final_pass": True,
            "candidate_pool": True,
            "target_interface_residues": "1,2",
            "binder_interface_residues": "3,4",
            "interface_residue_pairs": "1:3,2:4",
            "epitope_residues": "1,2,3",
            "epitope_overlap_residues": "1,2",
            "epitope_coverage": 2 / 3,
            "epitope_purity": 0.01,
        },
        {
            "job_name": "job_2",
            "target_chain": "A",
            "binder_chain": "B",
            "final_pass": False,
            "candidate_pool": False,
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

    write_public_reports(
        layout,
        rows,
        member_rows=members,
        representative_rows=representatives,
        final_job_ids=("job_1",),
        clustering_status="success",
    )

    all_header, all_rows = _read(layout.all_results)
    candidate_header, candidate_rows = _read(layout.candidates)
    final_header, final_rows = _read(layout.final_shortlist)
    assert all_header == candidate_header == final_header == list(PUBLIC_COLUMNS)
    assert len(all_rows) == 2
    assert [row["job_id"] for row in candidate_rows] == ["job_1"]
    assert [row["job_id"] for row in final_rows] == ["job_1"]
    assert all_rows[0]["primary_target_interface_residues"] == "A:1;A:2"
    assert all_rows[0]["primary_binder_interface_residues"] == "B:3;B:4"
    assert all_rows[0]["primary_interface_residue_pairs"] == "A:1-B:3;A:2-B:4"
    assert all_rows[0]["configured_epitope_residues"] == "A:1;A:2;A:3"
    assert "purity" not in "\n".join(all_header).lower()
    assert all_rows[0]["secondary_iptm"] == ""
    assert all_rows[0]["diversity_cell_id"] == (
        "binder_0001|complex_0001|epitope_0001"
    )


def test_layout_creates_all_numbered_stage_folders(tmp_path: Path) -> None:
    layout = RunOutputLayout(tmp_path).ensure()
    assert {path.name for path in layout.stages_root.iterdir()} == set(
        STAGE_DIRECTORIES.values()
    )
    for name in STAGE_DIRECTORIES:
        stage = layout.stage(name)
        assert stage.logs.is_dir()
        assert stage.tables.is_dir()
        assert stage.artifacts.is_dir()
