"""Input CSV parsing and validation."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import ValidationError

from af3_binder_filter.models import BinderCsvRow

REQUIRED_COLUMNS = ("sample_no", "run_name", "binder_sequence", "target_seq")


class CsvInputError(ValueError):
    """Raised when the binder CSV cannot be parsed or validated."""


def _normalize_column_name(column: str) -> str:
    return column.strip().removeprefix("\ufeff")


def read_binder_csv(csv_path: Path, *, limit: int | None = None) -> list[BinderCsvRow]:
    """Read and validate binder-target rows from the project CSV schema."""

    if not csv_path.exists():
        raise CsvInputError(f"CSV does not exist: {csv_path}")

    records: list[BinderCsvRow] = []
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CsvInputError(f"CSV has no header: {csv_path}")
        reader.fieldnames = [_normalize_column_name(column) for column in reader.fieldnames]
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            available = ", ".join(repr(column) for column in reader.fieldnames)
            raise CsvInputError(
                f"CSV missing required columns: {', '.join(missing)}. "
                f"Available columns: {available}"
            )

        for raw_row in reader:
            # DictReader.line_num tracks physical input lines, including blank
            # lines skipped by the CSV iterator.
            row_number = reader.line_num
            if limit is not None and len(records) >= limit:
                break
            if not any((value or "").strip() for value in raw_row.values()):
                continue

            payload = {column: raw_row.get(column, "") for column in REQUIRED_COLUMNS}
            payload["source_row_number"] = row_number
            try:
                records.append(BinderCsvRow.model_validate(payload))
            except ValidationError as exc:
                errors = []
                for error in exc.errors():
                    column = ".".join(str(part) for part in error["loc"])
                    errors.append(f"row {row_number} column {column}: {error['msg']}")
                raise CsvInputError("; ".join(errors)) from exc

    return records


def read_target_sequence(csv_path: Path) -> str:
    """Return the target sequence shared by the input CSV."""

    rows = read_binder_csv(csv_path)
    if not rows:
        raise CsvInputError(f"CSV has no binder rows: {csv_path}")
    targets = {row.target_seq for row in rows}
    if len(targets) != 1:
        mismatched = ", ".join(
            str(row.source_row_number) for row in rows if row.target_seq != rows[0].target_seq
        )
        raise CsvInputError(
            "all rows in one run must use the same target sequence; "
            f"mismatch at CSV row(s): {mismatched}"
        )
    return rows[0].target_seq
