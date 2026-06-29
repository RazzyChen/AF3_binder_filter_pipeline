"""ESM inverse folding scoring helpers."""

from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path
from typing import Any

from af3_binder_filter.af3_json import get_chain_sequence, load_af3_input
from af3_binder_filter.config import ESMConfig


class ESMScoreError(RuntimeError):
    """Raised when ESM scoring setup or parsing fails."""


def write_chain_fasta(input_json: Path, fasta_path: Path, *, chain_id: str = "B") -> Path:
    """Write a temporary FASTA for a chain from an AF3 complex input JSON."""

    payload = load_af3_input(input_json)
    job_name = str(payload.get("name") or input_json.stem)
    sequence = get_chain_sequence(payload, chain_id)
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    fasta_path.write_text(f">{job_name}|chain_{chain_id}\n{sequence}\n")
    return fasta_path


def build_esm_command(
    *,
    model_cif: Path,
    fasta_path: Path,
    score_csv: Path,
    chain_id: str,
    config: ESMConfig,
) -> list[str]:
    return [
        config.conda_bin,
        "run",
        "-n",
        config.conda_env,
        "python",
        str(config.scorer_path),
        str(model_cif),
        str(fasta_path),
        "--chain",
        chain_id,
        "--outpath",
        str(score_csv),
    ]


def parse_esm_score_csv(score_csv: Path) -> dict[str, float | str | None]:
    """Parse ESM score output and normalize log likelihood/perplexity fields."""

    if not score_csv.exists():
        raise ESMScoreError(f"ESM score CSV does not exist: {score_csv}")
    with score_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ESMScoreError(f"ESM score CSV has no rows: {score_csv}")

    row = rows[0]
    lower_map = {key.lower(): key for key in row}
    ll_key = next(
        (lower_map[key] for key in lower_map if "log" in key and "likelihood" in key),
        None,
    )
    if ll_key is None:
        raise ESMScoreError(f"ESM score CSV has no log likelihood column: {score_csv}")

    log_likelihood = float(row[ll_key])
    perplexity_key = next((lower_map[key] for key in lower_map if "perplex" in key), None)
    if perplexity_key is not None and row.get(perplexity_key) not in (None, ""):
        perplexity = float(row[perplexity_key])
    else:
        perplexity = math.exp(-log_likelihood)

    return {
        "esm_log_likelihood": log_likelihood,
        "esm_perplexity": perplexity,
    }


def score_one_esm(
    *,
    job_name: str,
    input_json: Path,
    model_cif: Path,
    output_dir: Path,
    chain_id: str = "B",
    config: ESMConfig,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Generate FASTA, optionally run ESM scorer, and parse score output."""

    job_dir = output_dir / job_name
    fasta_path = job_dir / f"{job_name}_chain_{chain_id}.fasta"
    score_csv = job_dir / f"{job_name}_esm_scores.csv"
    stdout_path = job_dir / "esm.stdout.log"
    stderr_path = job_dir / "esm.stderr.log"
    job_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "job_name": job_name,
        "esm_score_status": "pending",
        "esm_fasta_path": str(fasta_path),
        "esm_score_csv": str(score_csv),
        "esm_error": "",
    }

    try:
        write_chain_fasta(input_json, fasta_path, chain_id=chain_id)
        command = build_esm_command(
            model_cif=model_cif,
            fasta_path=fasta_path,
            score_csv=score_csv,
            chain_id=chain_id,
            config=config,
        )
        result["esm_command"] = " ".join(command)

        if dry_run:
            result["esm_score_status"] = "skipped"
            result["esm_error"] = "dry-run"
            return result

        if not model_cif.exists():
            result["esm_score_status"] = "missing"
            result["esm_error"] = f"missing model CIF: {model_cif}"
            return result

        if force or not score_csv.exists():
            with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
                completed = subprocess.run(
                    command,
                    shell=False,
                    check=False,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                )
            if completed.returncode != 0:
                result["esm_score_status"] = "error"
                result["esm_error"] = f"ESM scorer failed with code {completed.returncode}"
                return result

        result.update(parse_esm_score_csv(score_csv))
        result["esm_score_status"] = "success"
        return result
    except Exception as exc:  # noqa: BLE001 - scorer summaries should retain per-job failures
        result["esm_score_status"] = "error"
        result["esm_error"] = str(exc)
        return result


def write_esm_summary(summary_csv: Path, rows: list[dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with summary_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_esm_inputs(
    *,
    input_dir: Path,
    input_jsons: list[Path] | None = None,
    af_output_dir: Path,
    score_dir: Path,
    chain_id: str,
    config: ESMConfig,
    dry_run: bool = False,
    force: bool = False,
    use_ray: bool = True,
) -> list[dict[str, Any]]:
    """Score all complex input JSONs with top-level AF3 best model CIFs."""

    af_output_dir = af_output_dir.resolve()
    score_dir = score_dir.resolve()
    jobs: list[dict[str, Any]] = []
    for input_json in (list(input_jsons) if input_jsons is not None else sorted(input_dir.glob("*.json"))):
        input_json = input_json.resolve()
        payload = load_af3_input(input_json)
        job_name = str(payload.get("name") or input_json.stem)
        model_cif = af_output_dir / job_name / f"{job_name}_model.cif"
        jobs.append(
            {
                "job_name": job_name,
                "input_json": input_json,
                "model_cif": model_cif,
            }
        )

    if use_ray and not dry_run and jobs:
        try:
            import ray
        except ImportError as exc:
            raise ESMScoreError("Ray is required for score-esm unless --no-ray is set") from exc
        try:
            if not ray.is_initialized():
                ray.init()
        except Exception as exc:  # noqa: BLE001 - present clear Ray setup failures
            raise ESMScoreError(f"Ray initialization failed for ESM scoring: {exc}") from exc

        @ray.remote(num_gpus=1)
        def _score_job(job: dict[str, Any]) -> dict[str, Any]:
            return score_one_esm(
                job_name=job["job_name"],
                input_json=job["input_json"],
                model_cif=job["model_cif"],
                output_dir=score_dir / "esm",
                chain_id=chain_id,
                config=config,
                dry_run=False,
                force=force,
            )

        rows = ray.get([_score_job.remote(job) for job in jobs])
    else:
        rows = [
            score_one_esm(
                job_name=job["job_name"],
                input_json=job["input_json"],
                model_cif=job["model_cif"],
                output_dir=score_dir / "esm",
                chain_id=chain_id,
                config=config,
                dry_run=dry_run,
                force=force,
            )
            for job in jobs
        ]

    write_esm_summary(score_dir / "esm_scores_summary.csv", rows)
    return rows
