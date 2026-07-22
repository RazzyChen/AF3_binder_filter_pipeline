from __future__ import annotations

import json
import subprocess
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np

from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.config import RosettaSettings
from af3_binder_filter.interface import (
    analyze_interface_geometry,
    apply_balanced_shortlist,
)
from af3_binder_filter.jobs import JobSpec
from af3_binder_filter.rosetta import RosettaCliEngine, build_rosetta_command


def _structure(path: Path) -> Path:
    # A1/B1 are 4 Å apart; A2/B2 are separated and form no contact.
    array = struc.AtomArray(16)
    atom_offsets = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]], float)
    coordinates = []
    for origin in ([0, 0, 0], [20, 0, 0], [4, 0, 0], [40, 0, 0]):
        coordinates.extend(atom_offsets + np.asarray(origin))
    array.coord = np.asarray(coordinates)
    array.chain_id = np.array(["A"] * 8 + ["B"] * 8)
    array.res_id = np.array([1] * 4 + [2] * 4 + [1] * 4 + [2] * 4)
    array.res_name = np.array(["ALA"] * 16)
    array.atom_name = np.array(["N", "CA", "C", "O"] * 4)
    array.element = np.array(["N", "C", "C", "O"] * 4)
    array.hetero = np.array([False] * 16)
    strucio.save_structure(str(path), array)
    return path


def _job() -> JobSpec:
    return JobSpec(
        "job",
        "1",
        "run",
        "AA",
        "AA",
        "A",
        "B",
        2,
        42,
        "alphafold3",
        "alphafold3",
    )


def test_five_angstrom_contacts_and_epitope_scores(tmp_path: Path) -> None:
    model = _structure(tmp_path / "complex.pdb")
    prediction = UnifiedPrediction("job", "alphafold3", "success", best_model_path=model)

    result = analyze_interface_geometry(
        _job(),
        prediction,
        distance=5.0,
        epitope_residues="1-2",
        sasa_point_number=50,
        rosetta_input_dir=tmp_path / "rosetta_inputs",
    )

    assert result["interface_status"] == "success"
    assert result["interface_contact_pair_count"] == 1
    assert result["interface_minimum_distance"] == 4.0
    assert result["target_interface_residues"] == "A:1"
    assert result["binder_interface_residues"] == "B:1"
    assert result["interface_residue_pairs"] == "A:1-B:1"
    assert result["epitope_residues"] == "A:1;A:2"
    assert result["epitope_overlap_residues"] == "A:1"
    assert result["epitope_coverage"] == 0.5
    assert result["epitope_purity"] == 1.0
    assert result["epitope_jaccard"] == 0.5
    assert result["biotite_bsa_total"] > 0
    assert Path(result["rosetta_input_pdb"]).exists()
    residue_map = Path(result["residue_map_path"]).read_text()
    assert "sequence_position" in residue_map
    assert "\tA\t1\t" in residue_map


def test_balanced_shortlist_uses_epitope_coverage_without_purity_by_default() -> None:
    rows = [
        {
            "job_name": "one_of_three_hits",
            "interface_status": "success",
            "interface_contact_pair_count": 5,
            "epitope_coverage": 1 / 3,
            "epitope_purity": 0.01,
        },
        {
            "job_name": "insufficient_coverage",
            "interface_status": "success",
            "interface_contact_pair_count": 5,
            "epitope_coverage": 0.25,
            "epitope_purity": 1.0,
        },
    ]

    ranked = apply_balanced_shortlist(
        rows,
        epitope_configured=True,
        minimum_epitope_coverage=0.30,
    )
    by_job = {row["job_name"]: row for row in ranked}

    assert by_job["one_of_three_hits"]["epitope_pass"] is True
    assert by_job["one_of_three_hits"]["final_pass"] is True
    assert by_job["insufficient_coverage"]["epitope_pass"] is False


def test_legacy_epitope_purity_setting_is_ignored() -> None:
    ranked = apply_balanced_shortlist(
        [
            {
                "job_name": "legacy_config",
                "interface_status": "success",
                "interface_contact_pair_count": 5,
                "epitope_coverage": 1.0,
                "epitope_purity": 0.05,
            }
        ],
        epitope_configured=True,
        minimum_epitope_coverage=0.30,
        minimum_epitope_purity=0.30,
    )

    assert ranked[0]["epitope_pass"] is True
    assert ranked[0]["final_pass"] is True


def test_numeric_asym_ids_and_protenix_pae_are_mapped_to_input_chains(
    tmp_path: Path,
) -> None:
    model = _structure(tmp_path / "complex.pdb")
    confidence = tmp_path / "full_data.json"
    confidence.write_text(
        json.dumps(
            {
                "token_pair_pae": [
                    [0, 1, 4, 8],
                    [1, 0, 9, 9],
                    [6, 9, 0, 1],
                    [8, 9, 1, 0],
                ],
                "token_asym_id": [0, 0, 1, 1],
            }
        )
    )
    prediction = UnifiedPrediction(
        "job",
        "protenix",
        "success",
        best_model_path=model,
        confidence_path=confidence,
    )

    result = analyze_interface_geometry(
        _job(),
        prediction,
        distance=5.0,
        sasa_point_number=20,
    )

    assert result["interface_pae_target_to_binder_mean"] == 4.0
    assert result["interface_pae_binder_to_target_mean"] == 6.0
    assert result["interface_pae_mean"] == 5.0


def test_rosetta_parser_maps_interface_analyzer_fields(tmp_path: Path) -> None:
    pdb_path = _structure(tmp_path / "complex.pdb")

    def runner(command, **_kwargs):
        scorefile = Path(command[command.index("-out:file:score_only") + 1])
        scorefile.write_text(
            "SCORE: dSASA_int dG_separated dG_separated/dSASAx100 packstat description\n"
            "SCORE: 800 -12 -1.5 0.71 complex\n"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    result = RosettaCliEngine(RosettaSettings(), runner=runner).analyze(
        pdb_path,
        output_dir=tmp_path / "rosetta",
    )

    assert result["rosetta_status"] == "success"
    assert result["rosetta_dSASA_int"] == 800
    assert result["rosetta_dG_separated"] == -12
    assert result["rosetta_dG_separated_per_dSASA_x100"] == -1.5
    assert result["rosetta_packstat"] == 0.71


def test_rosetta_command_uses_a_fixed_seed_by_default(tmp_path: Path) -> None:
    settings = RosettaSettings(random_seed=424455)
    command = build_rosetta_command(
        settings,
        pdb_path=tmp_path / "complex.pdb",
        scorefile=tmp_path / "score.sc",
    )

    assert "-constant_seed" in command
    assert command[command.index("-jran") + 1] == "424455"

    unseeded = build_rosetta_command(
        RosettaSettings(constant_seed=False),
        pdb_path=tmp_path / "complex.pdb",
        scorefile=tmp_path / "score.sc",
    )
    assert "-constant_seed" not in unseeded
    assert "-jran" not in unseeded


def test_rosetta_failure_keeps_geometry_result(tmp_path: Path) -> None:
    model = _structure(tmp_path / "complex.pdb")
    geometry = analyze_interface_geometry(
        _job(),
        UnifiedPrediction("job", "alphafold3", "success", best_model_path=model),
        sasa_point_number=20,
    )

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, -9, "", "killed")

    rosetta = RosettaCliEngine(RosettaSettings(), runner=runner).analyze(
        model,
        output_dir=tmp_path / "rosetta",
    )

    assert geometry["interface_status"] == "success"
    assert rosetta["rosetta_status"] == "error"
    assert "returned -9" in rosetta["rosetta_error"]
