"""Sequence-derived design-chain metrics."""

from __future__ import annotations

from Bio.SeqUtils.ProtParam import ProteinAnalysis


class SequenceMetricError(RuntimeError):
    """Raised when sequence metrics cannot be calculated."""


def calculate_protein_pi(sequence: str) -> float:
    """Calculate protein isoelectric point from an amino-acid sequence."""

    normalized = "".join(str(sequence).split()).upper()
    if not normalized:
        raise SequenceMetricError("protein sequence must not be empty")
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    invalid = sorted(set(normalized) - allowed)
    if invalid:
        raise SequenceMetricError(
            "protein sequence contains unsupported amino-acid letters: " + "".join(invalid)
        )
    return float(ProteinAnalysis(normalized).isoelectric_point())
