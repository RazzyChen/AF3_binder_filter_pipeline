import csv

from af3_binder_filter.aggregate import aggregate_results


def test_aggregate_adds_design_chain_pi_and_esmfold_summary(tmp_path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "sample_no,run_name,binder_sequence,target_seq\n"
        "1,run,ACDEFGHIKLMNPQRSTVWY,ACDEFGHIKLMNPQRSTVWY\n"
    )
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    (score_dir / "esmfold_scores_summary.csv").write_text(
        "job_name,esmfold_status,esmfold_plddt_mean\n"
        "sample_1_binder_candiate_complex_pred,success,88.5\n"
    )

    rows = aggregate_results(
        csv_path=csv_path,
        af_output_dir=tmp_path / "af_output",
        results_dir=tmp_path / "results",
        score_dir=score_dir,
    )

    assert 2.0 < rows[0]["design_chain_pi"] < 13.0
    assert rows[0]["esmfold_status"] == "success"
    assert rows[0]["esmfold_plddt_mean"] == "88.5"

    with (tmp_path / "results" / "aggregate_results.csv").open(newline="") as handle:
        header = next(csv.reader(handle))
    assert "design_chain_pi" in header
    assert "esmfold_plddt_mean" in header
