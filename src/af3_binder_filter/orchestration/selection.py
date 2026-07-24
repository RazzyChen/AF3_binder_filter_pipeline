"""Cohesive selection orchestration boundary."""

from __future__ import annotations

import math
from pathlib import Path
from typing import (
    Any,
    Sequence,
)

from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.jobs import JobSpec


def merge_rows_by_job(
    rows: Sequence[dict[str, Any]], additions: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_job = {str(row["job_name"]): row for row in additions}
    return [{**row, **by_job.get(str(row["job_name"]), {})} for row in rows]


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def effective_predictions_from_rows(
    jobs: Sequence[JobSpec], rows: Sequence[dict[str, Any]]
) -> list[UnifiedPrediction]:
    """Materialize the selected backend as the one-structure adapter contract."""

    by_job = {str(row["job_name"]): row for row in rows}
    predictions: list[UnifiedPrediction] = []
    for job in jobs:
        row = by_job[job.job_id]
        status = row.get("effective_status")
        path_value = row.get("effective_best_model_path")
        predictions.append(
            UnifiedPrediction(
                job_id=job.job_id,
                backend=str(row.get("effective_backend") or "none"),
                status="success" if status == "success" and path_value else "missing",
                best_model_path=Path(str(path_value)) if path_value else None,
                ranking_score=_optional_float(row.get("effective_ranking_score")),
                iptm=_optional_float(row.get("effective_iptm")),
                ptm=_optional_float(row.get("effective_ptm")),
                plddt=_optional_float(row.get("effective_plddt_global_mean")),
                error=None if path_value else "no eligible effective backend",
                fingerprint_valid=bool(path_value),
            )
        )
    return predictions


def final_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    def number(name: str, default: float) -> float:
        try:
            value = float(row.get(name))
            return value if value == value else default
        except (TypeError, ValueError):
            return default

    return (
        0 if row.get("candidate_pool") else 1,
        -number("effective_epitope_coverage", -1),
        -number("consensus_epitope_jaccard", -1),
        -number("consensus_interface_pair_jaccard", -1),
        -number("consensus_interface_lddt", -1),
        number("consensus_interface_fixed_frame_rmsd", math.inf),
        number("effective_interface_pae_mean", math.inf),
        number("effective_rosetta_dG_separated_per_dSASA_x100", math.inf),
        -number("effective_rosetta_packstat", -math.inf),
        -number("effective_iptm", -math.inf),
        -number("esm_if_log_likelihood", -math.inf),
        -number("esmfold_plddt", -math.inf),
        -number("effective_ranking_score", -math.inf),
        str(row.get("job_name", "")),
    )


def secondary_gate_job_ids(predictions: Sequence[UnifiedPrediction], threshold: float) -> set[str]:
    """Gate on fingerprint-valid AF3 metrics, not AF3 structure/geometry success."""

    return {
        prediction.job_id
        for prediction in predictions
        if prediction.fingerprint_valid
        and prediction.iptm is not None
        and prediction.iptm >= threshold
    }
