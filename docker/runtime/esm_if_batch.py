"""Offline ESM-IF batch scorer used inside the unified runtime image."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    progress_dir = output.parent / ".aerith_progress" / "esm_if"
    progress_dir.mkdir(parents=True, exist_ok=True)
    for stale_marker in progress_dir.glob("*.json"):
        stale_marker.unlink()

    import torch
    import esm
    from esm.inverse_folding.util import (
        extract_coords_from_structure,
        load_structure,
        score_sequence,
    )

    jobs = json.loads(Path(args.manifest).read_text())
    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval().cuda() if torch.cuda.is_available() else model.eval()
    rows = []
    for job in jobs:
        row = {"job_name": job["job_name"], "esm_if_status": "error", "esm_if_error": ""}
        try:
            structure = load_structure(job["structure_path"], job["chain_id"])
            coords, sequence = extract_coords_from_structure(structure)
            ll_full, ll_with_coord = score_sequence(model, alphabet, coords, sequence)
            row.update(
                {
                    "esm_if_status": "success",
                    "esm_if_log_likelihood": float(ll_full),
                    "esm_if_log_likelihood_with_coord": float(ll_with_coord),
                    "esm_if_perplexity": math.exp(-float(ll_full)),
                }
            )
        except Exception as exc:
            row["esm_if_error"] = str(exc)
        rows.append(row)
        marker_name = hashlib.sha256(
            str(job["job_name"]).encode("utf-8")
        ).hexdigest()
        marker = progress_dir / f"{marker_name}.json"
        temporary_marker = marker.with_suffix(".json.tmp")
        temporary_marker.write_text(
            json.dumps(
                {
                    "job_name": job["job_name"],
                    "status": row["esm_if_status"],
                },
                sort_keys=True,
            )
            + "\n"
        )
        temporary_marker.replace(marker)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
