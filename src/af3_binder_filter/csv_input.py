"""Input CSV parsing and validation."""

from __future__ import annotations

import csv
from pathlib import Path

from pydantic import ValidationError

from af3_binder_filter.models import BinderCsvRow


REQUIRED_COLUMNS = ("sample_no", "run_name", "binder_sequence", "target_seq")


class CsvInputError(ValueError):
    """Raised when the binder CSV cannot be parsed or validated."""


def read_binder_csv(csv_path: Path, *, limit: int | None = None) -> list[BinderCsvRow]:
    """Read and validate binder-target rows from the project CSV schema."""

    if not csv_path.exists():
        raise CsvInputError(f"CSV does not exist: {csv_path}")

    records: list[BinderCsvRow] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CsvInputError(f"CSV has no header: {csv_path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise CsvInputError(f"CSV missing required columns: {', '.join(missing)}")

        for row_number, raw_row in enumerate(reader, start=2):
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

    rows = read_binder_csv(csv_path, limit=1)
    if not rows:
        raise CsvInputError(f"CSV has no binder rows: {csv_path}")
    return rows[0].target_seq
