from __future__ import annotations

import csv
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

_SMOKE = runpy.run_path(str(Path(__file__).parents[1] / "scripts" / "gpu_smoke.py"))
SmokeError = _SMOKE["SmokeError"]
load_contract = _SMOKE["load_contract"]
validate_backend_review = _SMOKE["validate_backend_review"]


def _review_csv(path: Path) -> Path:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["job_id", "primary_status", "secondary_status", "primary_iptm"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "job_id": "golden-1",
                "primary_status": "success",
                "secondary_status": "success",
                "primary_iptm": "0.82",
            }
        )
    return path


def _contract(path: Path, *, maximum: float = 0.9) -> Path:
    path.write_text(
        """{
  "schema_version": 1,
  "job_id": "golden-1",
  "required_nonempty": ["primary_iptm"],
  "backends": {
    "opendde": {
      "equals": {"primary_status": "success", "secondary_status": "success"},
      "ranges": {"primary_iptm": [0.7, REPLACE_MAX]}
    }
  }
}
""".replace("REPLACE_MAX", str(maximum))
    )
    return path


def test_gpu_smoke_contract_validates_ranges_and_statuses(tmp_path: Path) -> None:
    review = _review_csv(tmp_path / "backend_review.csv")
    contract = load_contract(_contract(tmp_path / "contract.json"))

    row = validate_backend_review(review, contract, "opendde")

    assert row["job_id"] == "golden-1"


def test_gpu_smoke_contract_reports_failed_ranges(tmp_path: Path) -> None:
    review = _review_csv(tmp_path / "backend_review.csv")
    contract = load_contract(_contract(tmp_path / "contract.json", maximum=0.8))

    with pytest.raises(SmokeError, match="above maximum"):
        validate_backend_review(review, contract, "opendde")


@pytest.mark.integration
def test_external_golden_gpu_smoke_contract() -> None:
    required = {
        "config": os.getenv("AERITH_GPU_SMOKE_CONFIG"),
        "contract": os.getenv("AERITH_GPU_SMOKE_CONTRACT"),
        "backend": os.getenv("AERITH_GPU_SMOKE_BACKEND"),
        "image": os.getenv("AERITH_GPU_SMOKE_IMAGE"),
        "work_root": os.getenv("AERITH_GPU_SMOKE_ROOT"),
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        pytest.skip("external GPU smoke is not configured: " + ", ".join(missing))

    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts" / "gpu_smoke.py"),
        "--config",
        required["config"],
        "--contract",
        required["contract"],
        "--backend",
        required["backend"],
        "--image",
        required["image"],
        "--work-root",
        required["work_root"],
    ]
    aerith_command = os.getenv("AERITH_GPU_SMOKE_AERITH_COMMAND")
    if aerith_command:
        command.extend(["--aerith-command", aerith_command])

    completed = subprocess.run(command, check=False)
    assert completed.returncode == 0
