import pytest

from af3_binder_filter.sequence_metrics import SequenceMetricError, calculate_protein_pi


def test_calculate_protein_pi_returns_float():
    value = calculate_protein_pi("ACDEFGHIKLMNPQRSTVWY")

    assert isinstance(value, float)
    assert 2.0 < value < 13.0


def test_calculate_protein_pi_rejects_invalid_sequence():
    with pytest.raises(SequenceMetricError, match="unsupported amino-acid"):
        calculate_protein_pi("ACDX")
