"""Importable ipSAE-style scoring for AF3 models.

This ports the AF3/mmCIF path of Roland Dunbrack's ipsae.py v4 script into
callable code for this pipeline.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RESIDUE_SET = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "DA",
    "DC",
    "DT",
    "DG",
    "A",
    "C",
    "U",
    "G",
}
NUC_RESIDUE_SET = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}


class IPSAEError(RuntimeError):
    """Raised when ipSAE scoring fails."""


@dataclass(frozen=True)
class ParsedResidues:
    residues: list[dict[str, Any]]
    cb_residues: list[dict[str, Any]]
    chains: np.ndarray
    token_mask: np.ndarray


def ptm_func(values: np.ndarray | float, d0: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + (values / d0) ** 2.0)


def calc_d0(length: float, pair_type: str) -> float:
    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
    d0 = 1.24 * (float(length) - 15.0) ** (1.0 / 3.0) - 1.8 if length > 27 else 1.0
    return max(min_value, d0)


def calc_d0_array(lengths: np.ndarray, pair_type: str) -> np.ndarray:
    min_value = 2.0 if pair_type == "nucleic_acid" else 1.0
    lengths = np.maximum(26, np.asarray(lengths, dtype=float))
    return np.maximum(min_value, 1.24 * (lengths - 15.0) ** (1.0 / 3.0) - 1.8)


def _parse_cif_atom(line: str, fields: dict[str, int]) -> dict[str, Any] | None:
    parts = line.split()
    residue_seq_num = parts[fields["label_seq_id"]]
    if residue_seq_num == ".":
        return None
    chain_field = "auth_asym_id" if "auth_asym_id" in fields else "label_asym_id"
    return {
        "atom_num": int(parts[fields["id"]]),
        "atom_name": parts[fields["label_atom_id"]],
        "residue_name": parts[fields["label_comp_id"]],
        "chain_id": parts[fields[chain_field]],
        "residue_seq_num": int(residue_seq_num),
        "x": float(parts[fields["Cartn_x"]]),
        "y": float(parts[fields["Cartn_y"]]),
        "z": float(parts[fields["Cartn_z"]]),
    }


def parse_af3_cif(cif_path: Path) -> ParsedResidues:
    fields: dict[str, int] = {}
    residues: list[dict[str, Any]] = []
    cb_residues: list[dict[str, Any]] = []
    chains: list[str] = []
    token_mask: list[int] = []

    with cif_path.open() as handle:
        for line in handle:
            if line.startswith("_atom_site."):
                field_name = line.strip().split(".", 1)[1]
                fields[field_name] = len(fields)
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            atom = _parse_cif_atom(line, fields)
            if atom is None:
                token_mask.append(0)
                continue

            atom_name = atom["atom_name"]
            residue_name = atom["residue_name"]
            is_token_atom = atom_name == "CA" or "C1" in atom_name
            if is_token_atom:
                token_mask.append(1)
                residues.append(
                    {
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": residue_name,
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": f"{residue_name:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    }
                )
                chains.append(atom["chain_id"])

            if atom_name == "CB" or "C3" in atom_name or (residue_name == "GLY" and atom_name == "CA"):
                cb_residues.append(
                    {
                        "atom_num": atom["atom_num"],
                        "coor": np.array([atom["x"], atom["y"], atom["z"]]),
                        "res": residue_name,
                        "chainid": atom["chain_id"],
                        "resnum": atom["residue_seq_num"],
                        "residue": f"{residue_name:3}   {atom['chain_id']:3} {atom['residue_seq_num']:4}",
                    }
                )

            if not is_token_atom and residue_name not in RESIDUE_SET:
                token_mask.append(0)

    if len(residues) != len(cb_residues):
        raise IPSAEError(
            f"CA/token residue count ({len(residues)}) differs from CB distance residue count ({len(cb_residues)})"
        )
    return ParsedResidues(residues, cb_residues, np.asarray(chains), np.asarray(token_mask))


def _chain_pair_type(chains: np.ndarray, residue_types: np.ndarray) -> dict[str, dict[str, str]]:
    unique_chains = ordered_unique(chains)
    chain_types: dict[str, str] = {}
    for chain in unique_chains:
        residues = residue_types[chains == chain]
        chain_types[chain] = "nucleic_acid" if any(res in NUC_RESIDUE_SET for res in residues) else "protein"
    return {
        chain1: {
            chain2: (
                "nucleic_acid"
                if chain_types[chain1] == "nucleic_acid" or chain_types[chain2] == "nucleic_acid"
                else "protein"
            )
            for chain2 in unique_chains
            if chain1 != chain2
        }
        for chain1 in unique_chains
    }


def ordered_unique(values: np.ndarray) -> list[str]:
    seen: list[str] = []
    for value in values:
        value = str(value)
        if value not in seen:
            seen.append(value)
    return seen


def _summary_iptm(confidences_path: Path, chains: list[str]) -> dict[str, dict[str, float]]:
    summary_path = Path(str(confidences_path).replace("confidences", "summary_confidences"))
    values = {a: {b: 0.0 for b in chains if a != b} for a in chains}
    if not summary_path.exists():
        return values
    data = json.loads(summary_path.read_text())
    matrix = data.get("chain_pair_iptm") or []
    for i, chain1 in enumerate(chains):
        for j, chain2 in enumerate(chains):
            if chain1 != chain2 and i < len(matrix) and j < len(matrix[i]):
                values[chain1][chain2] = float(matrix[i][j])
    return values


def calculate_ipsae(
    *,
    confidences_json: Path,
    model_cif: Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
) -> dict[str, Any]:
    """Calculate A->B, B->A, and max ipSAE metrics for an AF3 complex."""

    parsed = parse_af3_cif(model_cif)
    data = json.loads(confidences_json.read_text())
    residues = parsed.residues
    numres = len(residues)
    chains = parsed.chains
    unique_chains = ordered_unique(chains)
    if target_chain not in unique_chains or binder_chain not in unique_chains:
        raise IPSAEError(f"required chains {target_chain}/{binder_chain} not found in {model_cif}")

    coordinates = np.asarray([res["coor"] for res in parsed.cb_residues])
    distances = np.sqrt(((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(axis=2))
    residue_types = np.asarray([res["res"] for res in residues])
    chain_pair_type = _chain_pair_type(chains, residue_types)

    atom_plddts = np.asarray(data.get("atom_plddts", []), dtype=float)
    if atom_plddts.size:
        ca_atom_num = np.asarray([res["atom_num"] - 1 for res in residues])
        cb_atom_num = np.asarray([res["atom_num"] - 1 for res in parsed.cb_residues])
        plddt = atom_plddts[ca_atom_num]
        cb_plddt = atom_plddts[cb_atom_num]
    else:
        plddt = np.zeros(numres)
        cb_plddt = np.zeros(numres)

    pae_full = np.asarray(data["pae"], dtype=float)
    if parsed.token_mask.size == pae_full.shape[0]:
        pae = pae_full[np.ix_(parsed.token_mask.astype(bool), parsed.token_mask.astype(bool))]
    else:
        pae = pae_full
    if pae.shape != (numres, numres):
        raise IPSAEError(f"PAE matrix shape {pae.shape} does not match parsed residue count {numres}")

    iptm_af = _summary_iptm(confidences_json, unique_chains)
    pair_metrics: dict[tuple[str, str], dict[str, float]] = {}

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            pair_type = chain_pair_type[chain1][chain2]
            chain1_mask = chains == chain1
            chain2_mask = chains == chain2
            n0chn = int(np.sum(chain1_mask) + np.sum(chain2_mask))
            d0chn = calc_d0(n0chn, pair_type)
            ptm_d0chn = ptm_func(pae, d0chn)
            valid_matrix = np.outer(chain1_mask, chain2_mask) & (pae < pae_cutoff)

            unique_1: set[int] = set()
            unique_2: set[int] = set()
            dist_unique_1: set[int] = set()
            dist_unique_2: set[int] = set()
            iptm_byres = np.zeros(numres)
            ipsae_d0chn_byres = np.zeros(numres)

            for i in np.where(chain1_mask)[0]:
                iptm_byres[i] = ptm_d0chn[i, chain2_mask].mean() if np.any(chain2_mask) else 0.0
                valid_ipsae = valid_matrix[i]
                ipsae_d0chn_byres[i] = ptm_d0chn[i, valid_ipsae].mean() if np.any(valid_ipsae) else 0.0
                if np.any(valid_ipsae):
                    unique_1.add(residues[i]["resnum"])
                    for j in np.where(valid_ipsae)[0]:
                        unique_2.add(residues[j]["resnum"])
                dist_valid = chain2_mask & (pae[i] < pae_cutoff) & (distances[i] < dist_cutoff)
                if np.any(dist_valid):
                    dist_unique_1.add(residues[i]["resnum"])
                    for j in np.where(dist_valid)[0]:
                        dist_unique_2.add(residues[j]["resnum"])

            n0dom = len(unique_1) + len(unique_2)
            d0dom = calc_d0(n0dom, pair_type)
            ptm_d0dom = ptm_func(pae, d0dom)
            n0res_byres = np.sum(valid_matrix, axis=1)
            d0res_byres = calc_d0_array(n0res_byres, pair_type)
            ipsae_d0dom_byres = np.zeros(numres)
            ipsae_d0res_byres = np.zeros(numres)
            for i in np.where(chain1_mask)[0]:
                valid = valid_matrix[i]
                ipsae_d0dom_byres[i] = ptm_d0dom[i, valid].mean() if np.any(valid) else 0.0
                ptm_row_d0res = ptm_func(pae[i], d0res_byres[i])
                ipsae_d0res_byres[i] = ptm_row_d0res[valid].mean() if np.any(valid) else 0.0

            iptm_index = int(np.argmax(iptm_byres))
            ipsae_index = int(np.argmax(ipsae_d0res_byres))

            p_dockq, p_dockq2 = _pdockq_scores(
                chains=chains,
                distances=distances,
                pae=pae,
                cb_plddt=cb_plddt,
                chain1=chain1,
                chain2=chain2,
            )
            lis = _lis_score(chains=chains, pae=pae, chain1=chain1, chain2=chain2)
            pair_metrics[(chain1, chain2)] = {
                "ipSAE": float(ipsae_d0res_byres[ipsae_index]),
                "ipSAE_d0chn": float(ipsae_d0chn_byres[int(np.argmax(ipsae_d0chn_byres))]),
                "ipSAE_d0dom": float(ipsae_d0dom_byres[int(np.argmax(ipsae_d0dom_byres))]),
                "ipTM_af": float(iptm_af[chain1][chain2]),
                "ipTM_d0chn": float(iptm_byres[iptm_index]),
                "pDockQ": float(p_dockq),
                "pDockQ2": float(p_dockq2),
                "LIS": float(lis),
                "n0res": float(n0res_byres[ipsae_index]),
                "n0chn": float(n0chn),
                "n0dom": float(n0dom),
                "d0res": float(d0res_byres[ipsae_index]),
                "d0chn": float(d0chn),
                "d0dom": float(d0dom),
                "nres1": float(len(unique_1)),
                "nres2": float(len(unique_2)),
                "dist1": float(len(dist_unique_1)),
                "dist2": float(len(dist_unique_2)),
            }

    ab = pair_metrics[(target_chain, binder_chain)]
    ba = pair_metrics[(binder_chain, target_chain)]
    result: dict[str, Any] = {}
    for prefix, metrics in (("A_to_B", ab), ("B_to_A", ba)):
        for key, value in metrics.items():
            result[f"{key}_{prefix}"] = value
    for key in ("ipSAE", "ipSAE_d0chn", "ipSAE_d0dom", "ipTM_af", "ipTM_d0chn", "pDockQ", "pDockQ2"):
        result[f"{key}_max"] = max(float(ab[key]), float(ba[key]))
    result["LIS_max"] = max(float(ab["LIS"]), float(ba["LIS"]))
    return result


def _pdockq_scores(
    *,
    chains: np.ndarray,
    distances: np.ndarray,
    pae: np.ndarray,
    cb_plddt: np.ndarray,
    chain1: str,
    chain2: str,
) -> tuple[float, float]:
    cutoff = 8.0
    contact_residues: set[int] = set()
    npairs = 0
    ptm_sum = 0.0
    for i in np.where(chains == chain1)[0]:
        valid = (chains == chain2) & (distances[i] <= cutoff)
        if np.any(valid):
            npairs += int(np.sum(valid))
            contact_residues.add(int(i))
            for j in np.where(valid)[0]:
                contact_residues.add(int(j))
            ptm_sum += float(np.sum(ptm_func(pae[i][valid], 10.0)))
    if npairs == 0:
        return 0.0, 0.0
    mean_plddt = float(cb_plddt[list(contact_residues)].mean())
    x = mean_plddt * math.log10(npairs)
    pdockq = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
    x2 = mean_plddt * (ptm_sum / npairs)
    pdockq2 = 1.31 / (1 + math.exp(-0.075 * (x2 - 84.733))) + 0.005
    return pdockq, pdockq2


def _lis_score(*, chains: np.ndarray, pae: np.ndarray, chain1: str, chain2: str) -> float:
    selected = pae[(chains[:, None] == chain1) & (chains[None, :] == chain2)]
    valid = selected[selected < 12]
    if valid.size == 0:
        return 0.0
    return float(np.mean((12 - valid) / 12))


def score_one_ipsae(
    *,
    job_name: str,
    confidences_json: Path,
    model_cif: Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "job_name": job_name,
        "ipsae_score_status": "pending",
        "ipsae_error": "",
    }
    if not confidences_json.exists() or not model_cif.exists():
        result["ipsae_score_status"] = "missing"
        result["ipsae_error"] = f"missing confidences/model: {confidences_json}, {model_cif}"
        return result
    try:
        result.update(
            calculate_ipsae(
                confidences_json=confidences_json,
                model_cif=model_cif,
                target_chain=target_chain,
                binder_chain=binder_chain,
                pae_cutoff=pae_cutoff,
                dist_cutoff=dist_cutoff,
            )
        )
        result["ipsae_score_status"] = "success"
        return result
    except Exception as exc:  # noqa: BLE001 - per-job score failures should be captured
        result["ipsae_score_status"] = "error"
        result["ipsae_error"] = str(exc)
        return result


def write_ipsae_summary(summary_csv: Path, rows: list[dict[str, Any]]) -> None:
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


def score_ipsae_outputs(
    *,
    input_dir: Path,
    input_jsons: list[Path] | None = None,
    af_output_dir: Path,
    score_dir: Path,
    target_chain: str = "A",
    binder_chain: str = "B",
    pae_cutoff: float = 10.0,
    dist_cutoff: float = 15.0,
    use_ray: bool = True,
) -> list[dict[str, Any]]:
    af_output_dir = af_output_dir.resolve()
    score_dir = score_dir.resolve()
    jobs: list[dict[str, Any]] = []
    for input_json in (list(input_jsons) if input_jsons is not None else sorted(input_dir.glob("*.json"))):
        input_json = input_json.resolve()
        data = json.loads(input_json.read_text())
        job_name = str(data.get("name") or input_json.stem)
        job_dir = af_output_dir / job_name
        jobs.append(
            {
                "job_name": job_name,
                "confidences_json": job_dir / f"{job_name}_confidences.json",
                "model_cif": job_dir / f"{job_name}_model.cif",
            }
        )

    if use_ray and jobs:
        try:
            import ray
        except ImportError as exc:
            raise IPSAEError("Ray is required for score-ipsae unless --no-ray is set") from exc
        try:
            if not ray.is_initialized():
                ray.init()
        except Exception as exc:  # noqa: BLE001
            raise IPSAEError(f"Ray initialization failed for ipSAE scoring: {exc}") from exc

        @ray.remote
        def _score_job(job: dict[str, Any]) -> dict[str, Any]:
            return score_one_ipsae(
                job_name=job["job_name"],
                confidences_json=job["confidences_json"],
                model_cif=job["model_cif"],
                target_chain=target_chain,
                binder_chain=binder_chain,
                pae_cutoff=pae_cutoff,
                dist_cutoff=dist_cutoff,
            )

        rows = ray.get([_score_job.remote(job) for job in jobs])
    else:
        rows = [
            score_one_ipsae(
                job_name=job["job_name"],
                confidences_json=job["confidences_json"],
                model_cif=job["model_cif"],
                target_chain=target_chain,
                binder_chain=binder_chain,
                pae_cutoff=pae_cutoff,
                dist_cutoff=dist_cutoff,
            )
            for job in jobs
        ]

    write_ipsae_summary(score_dir / "ipsae_scores_summary.csv", rows)
    return rows
