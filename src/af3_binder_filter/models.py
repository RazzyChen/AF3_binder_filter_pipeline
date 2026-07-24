"""Validated input models used by the production pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class BinderCsvRow(BaseModel):
    """Validated binder design row from the input CSV."""

    model_config = ConfigDict(extra="forbid")

    sample_no: str
    run_name: str
    binder_sequence: str
    target_seq: str
    source_row_number: int = Field(ge=2)

    @field_validator("sample_no", "run_name")
    @classmethod
    def nonempty_text(cls, value: str, info: Any) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @field_validator("binder_sequence", "target_seq")
    @classmethod
    def normalize_sequence(cls, value: str, info: Any) -> str:
        sequence = "".join(str(value).split()).upper()
        if not sequence:
            raise ValueError(f"{info.field_name} must not be empty")
        allowed = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = sorted(set(sequence) - allowed)
        if invalid:
            raise ValueError(
                f"{info.field_name} contains unsupported amino-acid letters: {''.join(invalid)}"
            )
        return sequence
