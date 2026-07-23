"""Select the one backend whose structure drives downstream decisions.

The primary and secondary projections remain available for audit, but every
single-structure consumer (ESM-IF, Foldseek, contact clustering, and quality
representatives) must read the ``effective_*`` projection created here.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping


EFFECTIVE_BACKEND_FIELDS: tuple[str, ...] = (
    "backend",
    "status",
    "ranking_score",
    "iptm",
    "ptm",
    "plddt_global_mean",
    "best_model_path",
    "interface_status",
    "interface_contact_pair_count",
    "target_interface_residues",
    "binder_interface_residues",
    "interface_residue_pairs",
    "epitope_overlap_residues",
    "epitope_overlap_count",
    "epitope_coverage",
    "interface_pae_mean",
    "biotite_bsa_total",
    "rosetta_status",
    "rosetta_dG_separated_per_dSASA_x100",
    "rosetta_packstat",
)


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _backend_value(row: Mapping[str, Any], prefix: str, field: str) -> Any:
    key = f"{prefix}_{field}"
    if key in row:
        return row.get(key)
    if prefix != "primary":
        return None
    if field == "status":
        return row.get("job_status", row.get("status"))
    return row.get(field)


def _backend_pass(row: Mapping[str, Any], prefix: str) -> bool:
    if prefix == "primary":
        return _truthy(row.get("primary_final_pass", row.get("final_pass")))
    return _truthy(row.get("secondary_final_pass"))


def _eligible(row: Mapping[str, Any], prefix: str) -> bool:
    status = _backend_value(row, prefix, "status")
    interface_status = _backend_value(row, prefix, "interface_status")
    model_path = _backend_value(row, prefix, "best_model_path")
    return (
        status == "success"
        and interface_status == "success"
        and model_path not in (None, "")
    )


def _descending(value: Any) -> tuple[int, float]:
    number = _number(value)
    return (1, 0.0) if number is None else (0, -number)


def _ascending(value: Any) -> tuple[int, float]:
    number = _number(value)
    return (1, 0.0) if number is None else (0, number)


def backend_quality_key(row: Mapping[str, Any], prefix: str) -> tuple[Any, ...]:
    """Return the documented deterministic cross-backend quality key."""

    return (
        not _backend_pass(row, prefix),
        _descending(_backend_value(row, prefix, "epitope_coverage")),
        _ascending(_backend_value(row, prefix, "interface_pae_mean")),
        _ascending(
            _backend_value(
                row, prefix, "rosetta_dG_separated_per_dSASA_x100"
            )
        ),
        _descending(_backend_value(row, prefix, "rosetta_packstat")),
        _descending(_backend_value(row, prefix, "iptm")),
        # A complete tie intentionally prefers the independent secondary model.
        0 if prefix == "secondary" else 1,
    )


def _selection_reason(
    row: Mapping[str, Any], selected: str, other: str
) -> str:
    labels = (
        ("pass", not _backend_pass(row, selected), not _backend_pass(row, other)),
        (
            "epitope_coverage",
            _descending(_backend_value(row, selected, "epitope_coverage")),
            _descending(_backend_value(row, other, "epitope_coverage")),
        ),
        (
            "interface_pae_mean",
            _ascending(_backend_value(row, selected, "interface_pae_mean")),
            _ascending(_backend_value(row, other, "interface_pae_mean")),
        ),
        (
            "rosetta_normalized_dg",
            _ascending(
                _backend_value(
                    row, selected, "rosetta_dG_separated_per_dSASA_x100"
                )
            ),
            _ascending(
                _backend_value(
                    row, other, "rosetta_dG_separated_per_dSASA_x100"
                )
            ),
        ),
        (
            "rosetta_packstat",
            _descending(_backend_value(row, selected, "rosetta_packstat")),
            _descending(_backend_value(row, other, "rosetta_packstat")),
        ),
        (
            "iptm",
            _descending(_backend_value(row, selected, "iptm")),
            _descending(_backend_value(row, other, "iptm")),
        ),
    )
    for label, selected_value, other_value in labels:
        if selected_value != other_value:
            return f"quality:{label}"
    return "quality:secondary_tie_break"


def apply_effective_backend(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return *row* with a complete, auditable ``effective_*`` projection."""

    projected = dict(row)
    eligible = [
        prefix for prefix in ("primary", "secondary") if _eligible(row, prefix)
    ]
    if not eligible:
        projected.update(
            {
                "effective_backend": None,
                "effective_selection_reason": "no_eligible_backend",
                "effective_pass": None,
                **{
                    f"effective_{field}": None
                    for field in EFFECTIVE_BACKEND_FIELDS
                    if field != "backend"
                },
            }
        )
        return projected

    if len(eligible) == 1:
        selected = eligible[0]
        reason = f"only_{selected}_eligible"
    else:
        selected = min(eligible, key=lambda prefix: backend_quality_key(row, prefix))
        other = "secondary" if selected == "primary" else "primary"
        reason = _selection_reason(row, selected, other)

    projected["effective_selection_reason"] = reason
    projected["effective_pass"] = _backend_pass(row, selected)
    for field in EFFECTIVE_BACKEND_FIELDS:
        projected[f"effective_{field}"] = _backend_value(row, selected, field)
    # Normalize the source label even when an adapter omitted its backend key.
    projected["effective_backend"] = _backend_value(row, selected, "backend") or selected
    return projected


def effective_model_paths(
    rows: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
) -> dict[str, Path]:
    """Return validated non-empty effective model paths keyed by job name."""

    paths: dict[str, Path] = {}
    for row in rows:
        value = row.get("effective_best_model_path")
        if value not in (None, ""):
            paths[str(row.get("job_name", row.get("job_id", "")))] = Path(str(value))
    return paths
