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
    """Calculate target/binder SASA and buried dSASA from a complex CIF.

    `sasa_target_chain` and `sasa_binder_chain` are SASA values for each chain in
    the predicted complex. `sasa_target_free` and `sasa_binder_free` are computed
    from isolated chain atom arrays. dSASA is the buried interface area:
    target_free + binder_free - target_complex - binder_complex.
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
    complex_sasa = _sasa_values(complex_array, point_number=point_number)
    complex_target_mask = complex_array.chain_id == target_chain
    complex_binder_mask = complex_array.chain_id == binder_chain
    target_complex = _sum(complex_sasa[complex_target_mask])
    binder_complex = _sum(complex_sasa[complex_binder_mask])

    target_free = _sum(_sasa_values(atom_array[target_mask], point_number=point_number))
    binder_free = _sum(_sasa_values(atom_array[binder_mask], point_number=point_number))

    dsasa_target = target_free - target_complex
    dsasa_binder = binder_free - binder_complex
    dsasa = dsasa_target + dsasa_binder

    return {
        "sasa_status": "success",
        "sasa_error": "",
        "sasa_point_number": point_number,
        "sasa_target_chain": target_complex,
        "sasa_binder_chain": binder_complex,
        "sasa_complex_total": target_complex + binder_complex,
        "sasa_target_free": target_free,
        "sasa_binder_free": binder_free,
        "sasa_free_total": target_free + binder_free,
        "dsasa_target": dsasa_target,
        "dsasa_binder": dsasa_binder,
        "dsasa": dsasa,
        "dsasa_interface": dsasa,
    }
