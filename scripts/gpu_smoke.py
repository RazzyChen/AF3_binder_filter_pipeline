#!/usr/bin/env python3
"""Run one external golden Binder through an Aerith GPU cross-validation contract."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CONTRACT_SCHEMA_VERSION = 1


class SmokeError(RuntimeError):
    """Raised when a golden GPU smoke contract is invalid or fails."""


def _as_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SmokeError(f"{label} must be an object")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SmokeError(f"cannot read smoke contract {path}: {exc}") from exc
    contract = _as_mapping(payload, "smoke contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        raise SmokeError(
            f"unsupported smoke contract schema_version: {contract.get('schema_version')!r}"
        )
    backends = _as_mapping(contract.get("backends"), "smoke contract backends")
    if not backends:
        raise SmokeError("smoke contract backends must not be empty")
    return contract


def _string_list(value: object, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise SmokeError(f"{label} must be a list of column names")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    return _as_mapping(value, label)


def _bounds(value: object, label: str) -> tuple[float | None, float | None]:
    if isinstance(value, list) and len(value) == 2:
        lower, upper = value
    elif isinstance(value, dict):
        lower, upper = value.get("min"), value.get("max")
    else:
        raise SmokeError(f"{label} must be [min, max] or an object with min/max")

    def numeric(bound: object, bound_name: str) -> float | None:
        if bound is None:
            return None
        if not isinstance(bound, int | float) or isinstance(bound, bool):
            raise SmokeError(f"{label}.{bound_name} must be numeric or null")
        result = float(bound)
        if not math.isfinite(result):
            raise SmokeError(f"{label}.{bound_name} must be finite")
        return result

    minimum = numeric(lower, "min")
    maximum = numeric(upper, "max")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise SmokeError(f"{label}.min must be <= {label}.max")
    return minimum, maximum


def _merged_mapping(contract: dict[str, Any], suite: dict[str, Any], key: str) -> dict[str, Any]:
    return {
        **_mapping(contract.get(key), f"smoke contract {key}"),
        **_mapping(suite.get(key), f"smoke suite {key}"),
    }


def _selected_row(
    rows: list[dict[str, str]],
    *,
    job_id: str | None,
    review_csv: Path,
) -> dict[str, str]:
    if job_id is not None:
        matches = [row for row in rows if row.get("job_id") == job_id]
        if len(matches) != 1:
            raise SmokeError(
                f"golden job_id {job_id!r} matched {len(matches)} rows in {review_csv}"
            )
        return matches[0]
    if len(rows) != 1:
        raise SmokeError(
            f"smoke contract does not declare job_id and {review_csv} has {len(rows)} rows"
        )
    return rows[0]


def validate_backend_review(
    review_csv: Path,
    contract: dict[str, Any],
    backend: str,
) -> dict[str, str]:
    """Validate one backend-specific golden contract against ``backend_review.csv``."""

    suites = _as_mapping(contract["backends"], "smoke contract backends")
    if backend not in suites:
        raise SmokeError(f"smoke contract has no {backend!r} backend suite")
    suite = _as_mapping(suites[backend], f"smoke suite {backend}")
    try:
        with review_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as exc:
        raise SmokeError(f"cannot read backend review CSV {review_csv}: {exc}") from exc
    if not rows:
        raise SmokeError(f"backend review CSV is empty: {review_csv}")

    job_id = suite.get("job_id", contract.get("job_id"))
    if job_id is not None and not isinstance(job_id, str):
        raise SmokeError("smoke contract job_id must be a string")
    row = _selected_row(rows, job_id=job_id, review_csv=review_csv)
    errors: list[str] = []

    required_columns = [
        *_string_list(contract.get("required_nonempty"), "smoke contract required_nonempty"),
        *_string_list(suite.get("required_nonempty"), "smoke suite required_nonempty"),
    ]
    for column in required_columns:
        value = row.get(column)
        if value is None or value.strip().lower() in {"", "none", "nan", "null"}:
            errors.append(f"{column} must be present and nonempty")

    expected = _merged_mapping(contract, suite, "equals")
    for column, expected_value in expected.items():
        actual = row.get(column)
        if actual is None:
            errors.append(f"required column is absent: {column}")
        elif actual != str(expected_value):
            errors.append(f"{column} expected {expected_value!r}, found {actual!r}")

    ranges = _merged_mapping(contract, suite, "ranges")
    for column, range_spec in ranges.items():
        actual = row.get(column)
        if actual is None:
            errors.append(f"required numeric column is absent: {column}")
            continue
        try:
            value = float(actual)
        except ValueError:
            errors.append(f"{column} is not numeric: {actual!r}")
            continue
        if not math.isfinite(value):
            errors.append(f"{column} is not finite: {actual!r}")
            continue
        minimum, maximum = _bounds(range_spec, f"ranges.{column}")
        if minimum is not None and value < minimum:
            errors.append(f"{column}={value} is below minimum {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{column}={value} is above maximum {maximum}")

    if errors:
        raise SmokeError("golden contract failed:\n- " + "\n- ".join(errors))
    return row


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def pipeline_command(args: argparse.Namespace, run_id: str) -> tuple[list[str], Path]:
    root = args.work_root.expanduser().resolve()
    results_root = root / "results"
    run_root = results_root / run_id
    overrides = (
        f"project.work_dir={root / 'work'}",
        f"project.output_dir={root / 'outputs'}",
        f"project.results_dir={results_root}",
        f"project.run_id={run_id}",
        "project.allow_partial=false",
        f"backend.image={args.image}",
        f"secondary_backend.image={args.image}",
        f"features.image={args.image}",
    )
    command = [
        *shlex.split(args.aerith_command),
        "pipeline",
        "--config",
        str(args.config.expanduser().resolve()),
        "--secondary-backend",
        args.backend,
    ]
    for override in overrides:
        command.extend(["--override", override])
    return command, run_root


def _default_run_id(backend: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"gpu-smoke-{backend}-{stamp}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--backend", choices=("opendde", "protenix"), required=True)
    parser.add_argument("--image", required=True, help="immutable or candidate fold-runtime image")
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--run-id", help="unique smoke run identifier")
    parser.add_argument(
        "--aerith-command",
        default="uv run aerith",
        help="command used to invoke Aerith (default: 'uv run aerith')",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        contract = load_contract(args.contract.expanduser().resolve())
        run_id = args.run_id or _default_run_id(args.backend)
        command, run_root = pipeline_command(args, run_id)
        print(shlex.join(command), flush=True)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SmokeError(f"Aerith pipeline exited with return code {completed.returncode}")
        review_csv = run_root / "backend_review.csv"
        row = validate_backend_review(review_csv, contract, args.backend)
        summary = {
            "backend": args.backend,
            "image": args.image,
            "job_id": row.get("job_id"),
            "review_csv": str(review_csv),
            "run_id": run_id,
            "status": "passed",
        }
        _atomic_json(run_root / "gpu_smoke_summary.json", summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    except (OSError, SmokeError, subprocess.SubprocessError) as exc:
        print(f"GPU smoke error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
