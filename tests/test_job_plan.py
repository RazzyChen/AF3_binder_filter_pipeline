from pathlib import Path

import pytest

from af3_binder_filter.config import AerithConfig
from af3_binder_filter.csv_input import CsvInputError, read_target_sequence
from af3_binder_filter.jobs import (
    build_job_plan,
    job_fingerprint,
    parse_epitope_residues,
)


def _write(path: Path, rows: str) -> Path:
    path.write_text("sample_no,run_name,binder_sequence,target_seq\n" + rows)
    return path


def test_plan_validates_target_before_applying_limit(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "input.csv",
        "1,one,ACDE,LMNP\n2,two,FGHI,QRST\n",
    )
    config = AerithConfig()
    config.project.csv_path = str(csv_path)
    config.project.limit = 1

    with pytest.raises(CsvInputError, match="share one target"):
        build_job_plan(config)
    with pytest.raises(CsvInputError, match="same target"):
        read_target_sequence(csv_path)


def test_limit_is_applied_to_single_immutable_plan(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "input.csv",
        "\n1,one,ACDE,LMNP\n\n2,two,FGHI,LMNP\n",
    )
    config = AerithConfig()
    config.project.csv_path = str(csv_path)
    config.project.limit = 1

    plan = build_job_plan(config)

    assert len(plan.jobs) == 1
    assert plan.total_csv_jobs == 2
    assert plan.jobs[0].source_row_number == 3


def test_duplicate_sanitized_names_are_rejected(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "input.csv",
        "x,a,ACDE,LMNP\nx,b,FGHI,LMNP\n",
    )
    config = AerithConfig()
    config.project.csv_path = str(csv_path)

    with pytest.raises(CsvInputError, match="duplicate sanitized job name"):
        build_job_plan(config)


def test_fingerprint_changes_with_backend_or_sequence(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    config = AerithConfig()
    config.project.csv_path = str(csv_path)
    job = build_job_plan(config).jobs[0]
    original = job_fingerprint(job, config, feature_fingerprint="features")
    config.backend.name = "opendde"
    changed_backend = job_fingerprint(job, config, feature_fingerprint="features")

    assert original != changed_backend
    assert original != job_fingerprint(job, config, feature_fingerprint="new-features")


def test_fingerprint_changes_with_secondary_checkpoint(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    general = tmp_path / "opendde.pt"
    abag = tmp_path / "opendde_abag.pt"
    general.write_bytes(b"general")
    abag.write_bytes(b"abag")
    config = AerithConfig()
    config.project.csv_path = str(csv_path)
    config.secondary_backend.name = "opendde"
    config.secondary_backend.checkpoint_path = str(general)
    job = build_job_plan(config).jobs[0]
    original = job_fingerprint(job, config, feature_fingerprint="features")

    config.secondary_backend.checkpoint_path = str(abag)

    assert original != job_fingerprint(job, config, feature_fingerprint="features")


def test_epitope_parser_checks_one_based_range() -> None:
    assert parse_epitope_residues("1-3,5", target_length=5) == {1, 2, 3, 5}
    with pytest.raises(ValueError, match="outside target sequence"):
        parse_epitope_residues("0,6", target_length=5)
