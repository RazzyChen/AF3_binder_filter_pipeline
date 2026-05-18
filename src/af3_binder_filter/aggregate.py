"""Aggregate AF3, sequence, ESM, ESMFold, ipSAE, and SASA/BSA metrics."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from af3_binder_filter.af3_json import format_job_name
from af3_binder_filter.config import DEFAULT_COMPLEX_JOB_NAME_TEMPLATE
from af3_binder_filter.csv_input import read_binder_csv
from af3_binder_filter.models import BinderCsvRow
from af3_binder_filter.sasa import calculate_sasa_metrics
from af3_binder_filter.sequence_metrics import calculate_protein_pi


SUMMARY_FIELDS = (
    "ranking_score",
    "iptm",
    "ptm",
    "fraction_disordered",
    "has_clash",
    "chain_iptm",
    "chain_pair_iptm",
    "chain_pair_pae_min",
    "chain_ptm",
)


def _json_string(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        number = float(value)
        if math.isnan(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _normalize_plddt(value: Any) -> float | None:
    number = _safe_float(value)
    if number is None:
        return None
    return number / 100.0 if number > 1.0 else number


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _ranking_best(ranking_csv: Path) -> dict[str, Any]:
    if not ranking_csv.exists():
        return {}
    best: dict[str, Any] | None = None
    with ranking_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            score = _safe_float(row.get("ranking_score"))
            if score is None:
                continue
            if best is None or score > float(best["ranking_score"]):
                best = {
                    "best_seed": row.get("seed"),
                    "best_sample": row.get("sample"),
                    "ranking_score": score,
                }
    return best or {}


def _candidate_paths(job_dir: Path, job_name: str, best: dict[str, Any]) -> dict[str, Path]:
    paths = {
        "summary": job_dir / f"{job_name}_summary_confidences.json",
        "confidences": job_dir / f"{job_name}_confidences.json",
        "model": job_dir / f"{job_name}_model.cif",
    }
    seed = best.get("best_seed")
    sample = best.get("best_sample")
    if seed is not None and sample is not None:
        sample_dir = job_dir / f"seed-{seed}_sample-{sample}"
        sample_paths = {
            "summary": sample_dir / f"{job_name}_seed-{seed}_sample-{sample}_summary_confidences.json",
            "confidences": sample_dir / f"{job_name}_seed-{seed}_sample-{sample}_confidences.json",
            "model": sample_dir / f"{job_name}_seed-{seed}_sample-{sample}_model.cif",
        }
        for key, path in sample_paths.items():
            if not paths[key].exists() and path.exists():
                paths[key] = path
    return paths


def _array_stats(values: np.ndarray) -> tuple[float | None, float | None]:
    if values.size == 0:
        return None, None
    return float(np.mean(values)), float(np.min(values))


def _confidence_metrics(confidences: dict[str, Any], *, target_chain: str, binder_chain: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}

    if "atom_plddts" in confidences and "atom_chain_ids" in confidences:
        plddts = np.asarray(confidences["atom_plddts"], dtype=float)
        chains = np.asarray(confidences["atom_chain_ids"])
    elif "plddt" in confidences and "token_chain_ids" in confidences:
        plddts = np.asarray(confidences["plddt"], dtype=float)
        chains = np.asarray(confidences["token_chain_ids"])
    else:
        plddts = np.asarray([], dtype=float)
        chains = np.asarray([])

    for label, mask in {
        "global": np.ones(plddts.shape, dtype=bool),
        "chain_A": chains == target_chain,
        "chain_B": chains == binder_chain,
    }.items():
        mean_value, min_value = _array_stats(plddts[mask])
        metrics[f"plddt_{label}_mean"] = mean_value
        metrics[f"plddt_{label}_min"] = min_value

    metrics["normalized_plddt_global_mean"] = _normalize_plddt(metrics.get("plddt_global_mean"))

    if "pae" in confidences and "token_chain_ids" in confidences:
        pae = np.asarray(confidences["pae"], dtype=float)
        token_chains = np.asarray(confidences["token_chain_ids"])
        a_mask = token_chains == target_chain
        b_mask = token_chains == binder_chain
        ab = pae[np.ix_(a_mask, b_mask)]
        ba = pae[np.ix_(b_mask, a_mask)]
        metrics["ipae_A_to_B_mean"], metrics["ipae_A_to_B_min"] = _array_stats(ab)
        metrics["ipae_B_to_A_mean"], metrics["ipae_B_to_A_min"] = _array_stats(ba)
    else:
        metrics["ipae_A_to_B_mean"] = None
        metrics["ipae_A_to_B_min"] = None
        metrics["ipae_B_to_A_mean"] = None
        metrics["ipae_B_to_A_min"] = None

    return metrics


def _copy_best_model(
    model_path: Path,
    best_models_dir: Path,
    *,
    job_name: str,
    best_seed: Any,
    best_sample: Any,
) -> Path:
    seed = str(best_seed if best_seed is not None else "na")
    sample = str(best_sample if best_sample is not None else "na")
    destination = best_models_dir / f"{job_name}_seed-{seed}_sample-{sample}.cif"
    best_models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, destination)
    return destination


def aggregate_one_job(
    row: BinderCsvRow,
    *,
    job_name_template: str,
    af_output_dir: Path,
    best_models_dir: Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    sasa_point_number: int = 1000,
    design_chain_pi: float | None = None,
) -> dict[str, Any]:
    job_name = format_job_name(row, job_name_template)
    result: dict[str, Any] = {
        "sample_no": row.sample_no,
        "run_name": row.run_name,
        "job_name": job_name,
        "job_status": "pending",
        "job_error": "",
        "design_chain_pi": design_chain_pi if design_chain_pi is not None else calculate_protein_pi(row.binder_sequence),
    }

    job_dir = af_output_dir / job_name
    if not job_dir.exists():
        result["job_status"] = "missing"
        result["job_error"] = f"missing AF3 output directory: {job_dir}"
        return result

    try:
        best = _ranking_best(job_dir / f"{job_name}_ranking_scores.csv")
        result.update(best)
        paths = _candidate_paths(job_dir, job_name, best)
        missing = [name for name, path in paths.items() if not path.exists()]
        if missing:
            result["job_status"] = "missing"
            result["job_error"] = "missing output files: " + ", ".join(missing)
            return result

        summary = _load_json(paths["summary"])
        for field in SUMMARY_FIELDS:
            result[field] = _json_string(summary.get(field))

        confidences = _load_json(paths["confidences"])
        result.update(
            _confidence_metrics(confidences, target_chain=target_chain, binder_chain=binder_chain)
        )
        try:
            result.update(
                calculate_sasa_metrics(
                    paths["model"],
                    target_chain=target_chain,
                    binder_chain=binder_chain,
                    point_number=sasa_point_number,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep AF3 aggregation resumable
            result["sasa_status"] = "error"
            result["sasa_error"] = str(exc)
        copied_model = _copy_best_model(
            paths["model"],
            best_models_dir,
            job_name=job_name,
            best_seed=result.get("best_seed"),
            best_sample=result.get("best_sample"),
        )
        result["best_model_path"] = str(paths["model"])
        result["best_model_copy_path"] = str(copied_model)
        result["job_status"] = "success"
        return result
    except Exception as exc:  # noqa: BLE001 - aggregation must keep going per job
        result["job_status"] = "error"
        result["job_error"] = str(exc)
        return result


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def _candidate_passes(row: dict[str, Any]) -> bool:
    ipae_a_to_b = _safe_float(row.get("ipae_A_to_B_mean"))
    ipae_b_to_a = _safe_float(row.get("ipae_B_to_A_mean"))
    iptm = _safe_float(row.get("iptm"))
    normalized_plddt = _safe_float(row.get("normalized_plddt_global_mean"))
    esm_log_likelihood = _safe_float(row.get("esm_log_likelihood"))
    if None in (ipae_a_to_b, ipae_b_to_a, iptm, normalized_plddt, esm_log_likelihood):
        return False

    ipae_directional_mean = (ipae_a_to_b + ipae_b_to_a) / 2.0
    try:
        esm_perplexity = math.exp(-esm_log_likelihood)
    except OverflowError:
        return False

    return (
        ipae_directional_mean <= 1.9
        and iptm >= 0.80
        and normalized_plddt >= 0.85
        and esm_perplexity < 10
    )


def write_candidate_csv(
    aggregate_csv: Path,
    candidate_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Write candiate.csv by filtering aggregate_results.csv without changing headers/order."""

    output_path = candidate_csv or aggregate_csv.with_name("candiate.csv")
    with aggregate_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        candidates = [row for row in reader if _candidate_passes(row)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(candidates)
    return candidates


def _read_summary_by_job(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {
            row["job_name"]: row
            for row in csv.DictReader(handle)
            if row.get("job_name")
        }



def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

def aggregate_results(
    *,
    csv_path: Path,
    af_output_dir: Path,
    results_dir: Path = Path("."),
    score_dir: Path | None = None,
    job_name_template: str = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE,
    target_chain: str = "A",
    binder_chain: str = "B",
    sasa_point_number: int = 1000,
) -> list[dict[str, Any]]:
    rows = read_binder_csv(csv_path)
    best_models_dir = results_dir / "best_models"
    esm_by_job = _read_summary_by_job(score_dir / "esm_scores_summary.csv") if score_dir else {}
    esmfold_by_job = _read_summary_by_job(score_dir / "esmfold_scores_summary.csv") if score_dir else {}
    ipsae_by_job = _read_summary_by_job(score_dir / "ipsae_scores_summary.csv") if score_dir else {}

    design_chain_pi_values: list[float] = []
    with _progress() as progress:
        pi_task = progress.add_task("Calculating design-chain pI", total=len(rows))
        for row in rows:
            design_chain_pi_values.append(calculate_protein_pi(row.binder_sequence))
            progress.advance(pi_task)

    aggregated: list[dict[str, Any]] = []
    with _progress() as progress:
        aggregate_task = progress.add_task("Aggregating AF3 + SASA/BSA", total=len(rows))
        for row, design_chain_pi in zip(rows, design_chain_pi_values, strict=True):
            aggregated.append(
                aggregate_one_job(
                    row,
                    job_name_template=job_name_template,
                    af_output_dir=af_output_dir,
                    best_models_dir=best_models_dir,
                    target_chain=target_chain,
                    binder_chain=binder_chain,
                    sasa_point_number=sasa_point_number,
                    design_chain_pi=design_chain_pi,
                )
            )
            progress.advance(aggregate_task)

    with _progress() as progress:
        merge_task = progress.add_task("Merging score summaries", total=len(aggregated))
        for row in aggregated:
            job_name = row["job_name"]
            row.update(esm_by_job.get(job_name, {}))
            row.update(esmfold_by_job.get(job_name, {}))
            row.update(ipsae_by_job.get(job_name, {}))
            progress.advance(merge_task)

    aggregate_csv = results_dir / "aggregate_results.csv"
    write_csv(aggregate_csv, aggregated)
    write_candidate_csv(aggregate_csv, results_dir / "candiate.csv")

    input_rows = []
    with csv_path.open(newline="") as handle:
        for raw_row, metrics in zip(csv.DictReader(handle), aggregated, strict=False):
            if not any((value or "").strip() for value in raw_row.values()):
                continue
            input_rows.append({**raw_row, **metrics})
    write_csv(results_dir / "input_with_af3_metrics.csv", input_rows)
    return aggregated
