from pathlib import Path

import pytest

from af3_binder_filter.csv_input import CsvInputError, read_binder_csv


def write_csv(path: Path, header: str) -> None:
    path.write_text(
        f"{header}\n1,run_a,ACDEFGHIK,LMNPQRSTV\n",
        encoding="utf-8",
    )


def test_read_binder_csv_accepts_utf8_bom_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, "\ufeffsample_no,run_name,binder_sequence,target_seq")

    rows = read_binder_csv(csv_path)

    assert len(rows) == 1
    assert rows[0].sample_no == "1"
    assert rows[0].run_name == "run_a"


def test_read_binder_csv_strips_header_whitespace(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, " sample_no , run_name , binder_sequence , target_seq ")

    rows = read_binder_csv(csv_path)

    assert len(rows) == 1
    assert rows[0].binder_sequence == "ACDEFGHIK"


def test_read_binder_csv_reports_available_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "input.csv"
    write_csv(csv_path, "run_name,binder_sequence,target_seq")

    with pytest.raises(CsvInputError, match="Available columns"):
        read_binder_csv(csv_path)
