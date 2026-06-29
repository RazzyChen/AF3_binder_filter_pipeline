"""Solvent accessible surface area metrics for AF3 complex models."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np


class SASAError(RuntimeError):
    """Raised when SASA calculation fails."""


def _load_atom_array(model_cif: Path):
    try:
        from biotite.structure.io import pdbx
    except ImportError as exc:
        raise SASAError("biotite is required for SASA calculation") from exc

    if not model_cif.exists():
        raise SASAError(f"model CIF does not exist: {model_cif}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cif = pdbx.CIFFile.read(str(model_cif))
        return pdbx.get_structure(cif, model=1, include_bonds=False)


def _sasa_values(atom_array, *, point_number: int) -> np.ndarray:
    try:
        import biotite.structure as struc
    except ImportError as exc:
        raise SASAError("biotite is required for SASA calculation") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return struc.sasa(atom_array, point_number=point_number)


def _sum(values: np.ndarray) -> float:
    return float(np.nansum(values))


def calculate_sasa_metrics(
    model_cif: Path,
    *,
    target_chain: str = "A",
    binder_chain: str = "B",
    point_number: int = 1000,
) -> dict[str, Any]:
    """Calculate BSA from a complex CIF.

    `sasa_target` is calculated by deleting the binder chain from the current
    complex conformation. `sasa_binder` is calculated by deleting the target
    chain from the current complex conformation. BSA is the buried surface area:
    sasa_target + sasa_binder - sasa_complex, where `sasa_complex` is calculated
    from the target and binder chains together.
    """

    if point_number <= 0:
        raise SASAError("point_number must be positive")

    atom_array = _load_atom_array(model_cif)
    chain_ids = set(str(chain) for chain in atom_array.chain_id.tolist())
    if target_chain not in chain_ids:
        raise SASAError(f"target chain {target_chain!r} not found in {model_cif}")
    if binder_chain not in chain_ids:
        raise SASAError(f"binder chain {binder_chain!r} not found in {model_cif}")

    target_mask = atom_array.chain_id == target_chain
    binder_mask = atom_array.chain_id == binder_chain
    complex_mask = target_mask | binder_mask

    complex_array = atom_array[complex_mask]
    target_array = atom_array[target_mask]
    binder_array = atom_array[binder_mask]

    sasa_complex = _sum(_sasa_values(complex_array, point_number=point_number))
    sasa_target = _sum(_sasa_values(target_array, point_number=point_number))
    sasa_binder = _sum(_sasa_values(binder_array, point_number=point_number))
    bsa = sasa_target + sasa_binder - sasa_complex

    return {
        "sasa_status": "success",
        "sasa_error": "",
        "sasa_point_number": point_number,
        "sasa_target": sasa_target,
        "sasa_binder": sasa_binder,
        "sasa_complex": sasa_complex,
        "bsa": bsa,
        "bsa_interface": bsa,
    }
