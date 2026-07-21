from pathlib import Path
import csv

import pytest

from af3_binder_filter.clustering import (
    ClusteringError,
    build_foldseek_container_command,
    greedy_epitope_clusters,
    parse_foldseek_clusters,
    select_quality_representatives,
    write_cluster_outputs,
)
from af3_binder_filter.config import ClusteringSettings
from af3_binder_filter.jobs import JobSpec


def _job(job_id: str, sequence: str) -> JobSpec:
    return JobSpec(job_id, job_id, "run", "AAAA", sequence, "A", "B", 2, 42, "x", "x")


def test_epitope_clusters_ignore_binder_fold_and_are_deterministic() -> None:
    membership, representatives = greedy_epitope_clusters(
        {"different_fold": "1,2,3", "same_pose": "1,2", "other": "9,10"},
        threshold=0.5,
    )

    assert membership["different_fold"] == membership["same_pose"]
    assert membership["other"] != membership["same_pose"]
    assert representatives[membership["different_fold"]] == "different_fold"

    qualified, _ = greedy_epitope_clusters(
        {"a": "A:1;A:2;A:3", "b": "A:1;A:2", "c": "A:9;A:10"},
        threshold=0.5,
    )
    assert qualified["a"] == qualified["b"]
    assert qualified["a"] != qualified["c"]


def test_foldseek_parser_keeps_raw_representative_and_singletons(tmp_path: Path) -> None:
    tsv = tmp_path / "cluster.tsv"
    tsv.write_text("a.pdb\ta.pdb\na.pdb\tb.pdb\n")

    membership, raw = parse_foldseek_clusters(
        tsv,
        all_job_ids=["a", "b", "c"],
        prefix="binder",
    )

    assert membership["a"] == membership["b"]
    assert raw[membership["a"]] == "a"
    assert membership["c"] != membership["a"]


def test_foldseek_clustering_uses_in_image_gpu_binary(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    execution_dir = tmp_path / "execution"
    input_dir.mkdir()
    execution_dir.mkdir()

    command = build_foldseek_container_command(
        ClusteringSettings(),
        layer="binder",
        docker_bin="docker",
        image="aerith/fold-runtime:local",
        gpu_index=2,
        input_dir=input_dir,
        execution_dir=execution_dir,
        container_name="aerith-foldseek-gpu2",
    )

    assert command[command.index("--gpus") + 1] == "device=2"
    assert "aerith/fold-runtime:local" in command
    assert command[command.index("aerith/fold-runtime:local") + 1] == "foldseek"
    assert command[command.index("--gpu") + 1] == "1"


def test_foldseek_clustering_rejects_host_binary_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    execution_dir = tmp_path / "execution"
    input_dir.mkdir()
    execution_dir.mkdir()

    with pytest.raises(ClusteringError, match="host binaries are disabled"):
        build_foldseek_container_command(
            ClusteringSettings(foldseek_binary="/host/bin/foldseek"),
            layer="binder",
            docker_bin="docker",
            image="aerith/fold-runtime:local",
            gpu_index=0,
            input_dir=input_dir,
            execution_dir=execution_dir,
            container_name="aerith-foldseek-gpu0",
        )


def test_quality_representative_and_all_cluster_outputs(tmp_path: Path) -> None:
    jobs = [_job("a", "ACDE"), _job("b", "FGHI")]
    rows = [
        {"job_name": "a", "final_pass": False, "iptm": 0.9, "target_interface_residues": "1,2"},
        {"job_name": "b", "final_pass": True, "iptm": 0.8, "target_interface_residues": "1,2"},
    ]
    membership = {"a": "cluster_1", "b": "cluster_1"}

    quality = select_quality_representatives(
        {row["job_name"]: row for row in rows},
        membership,
    )
    assert quality["cluster_1"] == "b"

    write_cluster_outputs(
        results_dir=tmp_path,
        jobs=jobs,
        rows=rows,
        binder_membership=membership,
        binder_raw_representatives={"cluster_1": "a"},
        complex_membership=membership,
        complex_raw_representatives={"cluster_1": "a"},
        epitope_membership=membership,
        epitope_raw_representatives={"cluster_1": "a"},
    )

    for name in (
        "binder_clusters.tsv",
        "binder_representatives.fasta",
        "complex_clusters.tsv",
        "complex_cluster_report.tsv",
        "epitope_clusters.tsv",
        "cluster_members.csv",
        "cluster_representatives.csv",
        "final_shortlist.csv",
    ):
        assert (tmp_path / name).exists()
    assert ">b\nFGHI" in (tmp_path / "binder_representatives.fasta").read_text()


def test_secondary_only_rescue_is_kept_in_final_shortlist(tmp_path: Path) -> None:
    jobs = [_job("secondary_rescue", "ACDE")]
    rows = [
        {
            "job_name": "secondary_rescue",
            # The unprefixed field is the primary-backend result.
            "final_pass": False,
            "secondary_final_pass": True,
            "candidate_pool": True,
            "epitope_coverage": 0.0,
            "secondary_epitope_coverage": 1 / 3,
            "target_interface_residues": "1,2",
        }
    ]
    membership = {"secondary_rescue": "cluster_1"}

    write_cluster_outputs(
        results_dir=tmp_path,
        jobs=jobs,
        rows=rows,
        binder_membership=membership,
        binder_raw_representatives={"cluster_1": "secondary_rescue"},
        complex_membership=membership,
        complex_raw_representatives={"cluster_1": "secondary_rescue"},
        epitope_membership=membership,
        epitope_raw_representatives={"cluster_1": "secondary_rescue"},
    )

    with (tmp_path / "final_shortlist.csv").open(newline="") as handle:
        shortlist = list(csv.DictReader(handle))
    assert [row["job_name"] for row in shortlist] == ["secondary_rescue"]
