import numpy as np

from af3_binder_filter.sasa import calculate_sasa_metrics


class FakeAtomArray:
    def __init__(self, chain_ids):
        self.chain_id = np.asarray(chain_ids)

    def __getitem__(self, mask):
        return FakeAtomArray(self.chain_id[mask])


def test_bsa_uses_current_complex_conformation_chain_deletion(monkeypatch, tmp_path):
    def fake_load_atom_array(_model_cif):
        return FakeAtomArray(["A", "A", "B", "B"])

    def fake_sasa_values(atom_array, *, point_number):
        assert point_number == 100
        chains = tuple(atom_array.chain_id.tolist())
        if chains == ("A", "A", "B", "B"):
            return np.asarray([5.0, 7.0, 11.0, 13.0])
        if chains == ("A", "A"):
            return np.asarray([20.0, 22.0])
        if chains == ("B", "B"):
            return np.asarray([30.0, 32.0])
        raise AssertionError(f"unexpected chain selection: {chains}")

    monkeypatch.setattr("af3_binder_filter.sasa._load_atom_array", fake_load_atom_array)
    monkeypatch.setattr("af3_binder_filter.sasa._sasa_values", fake_sasa_values)

    metrics = calculate_sasa_metrics(tmp_path / "model.cif", point_number=100)

    assert metrics["sasa_status"] == "success"
    assert metrics["sasa_target"] == 42.0
    assert metrics["sasa_binder"] == 62.0
    assert metrics["sasa_complex"] == 36.0
    assert metrics["bsa"] == 68.0
    assert metrics["bsa_interface"] == metrics["bsa"]

    old_fields = (
        "sasa_target_chain",
        "sasa_binder_chain",
        "sasa_target_free",
        "sasa_binder_free",
        "sasa_complex_total",
        "sasa_free_total",
        "dsasa_target",
        "dsasa_binder",
        "dsasa",
        "dsasa_interface",
    )
    assert not any(field in metrics for field in old_fields)
