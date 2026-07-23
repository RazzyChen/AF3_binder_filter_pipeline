from af3_binder_filter.effective import apply_effective_backend


def _row(**values):
    row = {
        "job_name": "x",
        "primary_backend": "alphafold3",
        "primary_status": "success",
        "primary_interface_status": "success",
        "primary_best_model_path": "/tmp/primary.cif",
        "primary_final_pass": True,
        "primary_epitope_coverage": 0.5,
        "primary_interface_pae_mean": 4.0,
        "primary_rosetta_dG_separated_per_dSASA_x100": -1.0,
        "primary_rosetta_packstat": 0.6,
        "primary_iptm": 0.8,
        "secondary_backend": "opendde",
        "secondary_status": "success",
        "secondary_interface_status": "success",
        "secondary_best_model_path": "/tmp/secondary.cif",
        "secondary_final_pass": True,
        "secondary_epitope_coverage": 0.5,
        "secondary_interface_pae_mean": 4.0,
        "secondary_rosetta_dG_separated_per_dSASA_x100": -1.0,
        "secondary_rosetta_packstat": 0.6,
        "secondary_iptm": 0.8,
    }
    row.update(values)
    return row


def test_complete_quality_tie_prefers_secondary():
    selected = apply_effective_backend(
        _row(
            primary_normalized_binder_pdb_path="/tmp/primary-binder.pdb",
            secondary_normalized_binder_pdb_path="/tmp/secondary-binder.pdb",
        )
    )

    assert selected["effective_backend"] == "opendde"
    assert selected["effective_best_model_path"] == "/tmp/secondary.cif"
    assert selected["effective_normalized_binder_pdb_path"] == (
        "/tmp/secondary-binder.pdb"
    )
    assert selected["effective_selection_reason"] == "quality:secondary_tie_break"


def test_secondary_rescue_beats_primary_failure():
    selected = apply_effective_backend(
        _row(primary_final_pass=False, secondary_final_pass=True)
    )

    assert selected["effective_backend"] == "opendde"
    assert selected["effective_pass"] is True
    assert selected["effective_selection_reason"] == "quality:pass"


def test_missing_metric_loses_to_available_metric():
    selected = apply_effective_backend(
        _row(primary_interface_pae_mean=None, secondary_interface_pae_mean=12.0)
    )

    assert selected["effective_backend"] == "opendde"
    assert selected["effective_selection_reason"] == "quality:interface_pae_mean"


def test_only_geometry_success_backend_is_eligible():
    selected = apply_effective_backend(
        _row(secondary_interface_status="error", secondary_final_pass=False)
    )

    assert selected["effective_backend"] == "alphafold3"
    assert selected["effective_selection_reason"] == "only_primary_eligible"


def test_no_eligible_backend_has_missing_projection():
    selected = apply_effective_backend(
        _row(primary_interface_status="error", secondary_interface_status="error")
    )

    assert selected["effective_backend"] is None
    assert selected["effective_pass"] is None
    assert selected["effective_selection_reason"] == "no_eligible_backend"
