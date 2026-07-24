from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np
import pytest

from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.clustering import ClusteringError, prepare_foldseek_inputs
from af3_binder_filter.config import ConsensusSettings
from af3_binder_filter.consensus import (
    consensus_rows,
    structure_consensus_metrics_from_rows,
)
from af3_binder_filter.derived_structures import (
    DERIVED_STRUCTURE_SCHEMA,
    DerivedStructureValidationError,
    file_sha256,
    validate_derived_manifest,
    validated_artifacts_from_row,
)
from af3_binder_filter.effective import apply_effective_backend
from af3_binder_filter.esm_tools import (
    _chain_ca,
    _fold_comparison,
    add_esmfold_backend_comparison,
    collect_esm_rows,
    load_cached_esm_rows,
    write_esm_inputs,
)
from af3_binder_filter.interface import (
    InterfaceError,
    _chain_sequence_positions,
    analyze_interface_geometry,
)
from af3_binder_filter.jobs import JobSpec


def _structure(path: Path, *, binder_shift: float = 0.0) -> Path:
    array = struc.AtomArray(24)
    atom_offsets = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]],
        dtype=float,
    )
    coordinates = []
    for origin in (
        [0, 0, 0],
        [20, 0, 0],
        [0, 20, 0],
        [4, binder_shift, 0],
        [40, binder_shift, 0],
        [0, 40 + binder_shift, 0],
    ):
        coordinates.extend(atom_offsets + np.asarray(origin))
    array.coord = np.asarray(coordinates)
    array.chain_id = np.array(["A"] * 12 + ["B"] * 12)
    array.res_id = np.repeat([10, 20, 30, 40, 50, 60], 4)
    array.res_name = np.array(["ALA"] * 24)
    array.atom_name = np.array(["N", "CA", "C", "O"] * 6)
    array.element = np.array(["N", "C", "C", "O"] * 6)
    array.hetero = np.array([False] * 24)
    strucio.save_structure(str(path), array)
    return path


def _ca_trace(
    path: Path,
    positions: list[int],
    coordinates: list[list[float]],
    *,
    chain_id: str = "B",
) -> Path:
    array = struc.AtomArray(len(positions))
    array.coord = np.asarray(coordinates, dtype=float)
    array.chain_id = np.asarray([chain_id] * len(positions))
    array.res_id = np.asarray(positions)
    array.res_name = np.asarray(["ALA"] * len(positions))
    array.atom_name = np.asarray(["CA"] * len(positions))
    array.element = np.asarray(["C"] * len(positions))
    array.hetero = np.asarray([False] * len(positions))
    strucio.save_structure(str(path), array)
    return path


def _job() -> JobSpec:
    return JobSpec(
        "job",
        "1",
        "run",
        "AAA",
        "AAA",
        "A",
        "B",
        2,
        42,
        "alphafold3",
        "alphafold3",
    )


def _derive(
    tmp_path: Path,
    *,
    backend: str = "alphafold3",
    distance: float = 5.0,
    model_name: str = "source.pdb",
    binder_shift: float = 0.0,
) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model = _structure(tmp_path / model_name, binder_shift=binder_shift)
    row = analyze_interface_geometry(
        _job(),
        UnifiedPrediction(
            "job",
            backend,
            "success",
            best_model_path=model,
        ),
        distance=distance,
        sasa_point_number=20,
        rosetta_input_dir=(tmp_path / f"{backend}_stage" / "artifacts" / "rosetta_inputs"),
    )
    assert row["interface_status"] == "success"
    return model, row


def _effective_row(
    row: dict[str, object],
    model: Path,
    *,
    backend: str = "alphafold3",
    prefix: str = "effective",
) -> dict[str, object]:
    result = {
        "job_name": "job",
        "target_chain": "A",
        "binder_chain": "B",
        "target_sequence": "AAA",
        "binder_sequence": "AAA",
        f"{prefix}_backend": backend,
        f"{prefix}_best_model_path": str(model),
    }
    result.update({f"{prefix}_{key}": value for key, value in row.items()})
    return result


def test_interface_materializes_and_atomically_validates_run_local_derivatives(
    tmp_path: Path,
) -> None:
    model, first = _derive(tmp_path)

    assert first["derived_structure_cache_hit"] is False
    manifest_path = Path(str(first["derived_structure_manifest_path"]))
    payload = json.loads(manifest_path.read_text())
    assert payload["schema"] == DERIVED_STRUCTURE_SCHEMA
    assert payload["source_model_sha256"] == file_sha256(model)
    assert payload["target_chain"] == "A"
    assert payload["binder_chain"] == "B"
    assert payload["content_id"] in manifest_path.parts

    artifact_fields = (
        "normalized_complex_pdb_path",
        "normalized_target_pdb_path",
        "normalized_binder_pdb_path",
        "derived_residue_map_path",
        "derived_coordinates_path",
    )
    assert all(Path(str(first[field])).is_file() for field in artifact_fields)
    with np.load(Path(str(first["derived_coordinates_path"]))) as coordinates:
        assert coordinates["ca_coord"].shape == (6, 3)
        assert coordinates["sequence_position"].tolist() == [1, 2, 3, 1, 2, 3]
        assert coordinates["is_interface"].tolist() == [
            True,
            False,
            False,
            True,
            False,
            False,
        ]

    _model, second = _derive(tmp_path)
    assert second["derived_structure_cache_hit"] is True
    assert second["derived_structure_id"] == first["derived_structure_id"]

    binder = Path(str(first["normalized_binder_pdb_path"]))
    binder.write_text("corrupt\n")
    assert validate_derived_manifest(manifest_path) is None
    _model, repaired = _derive(tmp_path)
    assert repaired["derived_structure_cache_hit"] is False
    assert validate_derived_manifest(manifest_path) is not None


def test_foldseek_staging_reuses_validated_normalized_structures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model, interface_row = _derive(tmp_path)

    def fail_raw_parse(*_args, **_kwargs):
        raise AssertionError("raw model must not be parsed when derivatives are valid")

    monkeypatch.setattr(
        "af3_binder_filter.clustering.load_protein_complex",
        fail_raw_parse,
    )
    binder_dir, complex_dir = prepare_foldseek_inputs(
        (_job(),),
        {"job": model},
        work_dir=tmp_path / "foldseek",
        rows=(_effective_row(interface_row, model),),
    )

    staged_binder = binder_dir / "job.pdb"
    staged_complex = complex_dir / "job.pdb"
    assert file_sha256(staged_binder) == file_sha256(
        Path(str(interface_row["normalized_binder_pdb_path"]))
    )
    assert file_sha256(staged_complex) == file_sha256(
        Path(str(interface_row["normalized_complex_pdb_path"]))
    )


def test_foldseek_rejects_a_declared_derivative_that_no_longer_validates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model, interface_row = _derive(tmp_path)
    effective = _effective_row(interface_row, model)
    Path(str(interface_row["normalized_binder_pdb_path"])).write_text("corrupt\n")

    def fail_raw_parse(*_args, **_kwargs):
        raise AssertionError("a declared derivative must not raw-fallback")

    monkeypatch.setattr(
        "af3_binder_filter.clustering.load_protein_complex",
        fail_raw_parse,
    )
    with pytest.raises(ClusteringError, match="no longer validates"):
        prepare_foldseek_inputs(
            (_job(),),
            {"job": model},
            work_dir=tmp_path / "foldseek_corrupt",
            rows=(effective,),
        )


def test_esm_uses_staged_derived_binder_for_if_and_fold_comparison(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model, interface_row = _derive(tmp_path)
    job = _job()
    prediction = UnifiedPrediction(
        "job",
        "alphafold3",
        "success",
        best_model_path=model,
    )
    effective = _effective_row(interface_row, model)
    esm_inputs = tmp_path / "esm_inputs"

    _fasta, manifest = write_esm_inputs(
        (job,),
        (prediction,),
        esm_inputs,
        structure_rows=(effective,),
    )

    jobs = json.loads(manifest.read_text())
    assert jobs[0]["structure_path"] == "/inputs/derived_structures/job.pdb"
    staged = esm_inputs / "derived_structures" / "job.pdb"
    assert file_sha256(staged) == file_sha256(
        Path(str(interface_row["normalized_binder_pdb_path"]))
    )

    output = tmp_path / "esm_output"
    esmfold = output / "esmfold"
    esmfold.mkdir(parents=True)
    esmfold_model = esmfold / "job_chain_B.pdb"
    esmfold_model.write_bytes(staged.read_bytes())
    seen_paths: list[Path] = []
    original_chain_ca = _chain_ca

    def recording_chain_ca(
        path: Path,
        chain_id: str | None = None,
    ) -> dict[int, np.ndarray]:
        seen_paths.append(Path(path))
        return original_chain_ca(path, chain_id)

    monkeypatch.setattr(
        "af3_binder_filter.esm_tools._chain_ca",
        recording_chain_ca,
    )
    rows = collect_esm_rows(
        (job,),
        (prediction,),
        output,
        structure_rows=(effective,),
    )

    assert rows[0]["esmfold_effective_binder_tm"] == 1.0
    assert rows[0]["esmfold_effective_binder_rmsd"] == pytest.approx(0.0, abs=1e-12)
    assert seen_paths[0] == Path(str(interface_row["normalized_binder_pdb_path"]))


def test_derivative_identity_covers_mapping_sequences_contacts_and_cutoff(
    tmp_path: Path,
) -> None:
    _model, first = _derive(tmp_path, distance=5.0)
    _model, changed_cutoff = _derive(tmp_path, distance=3.0)

    assert first["derived_structure_id"] != changed_cutoff["derived_structure_id"]
    first_payload = json.loads(Path(str(first["derived_structure_manifest_path"])).read_text())
    changed_payload = json.loads(
        Path(str(changed_cutoff["derived_structure_manifest_path"])).read_text()
    )
    identity = first_payload["identity"]
    assert identity["target_sequence_sha256"]
    assert identity["binder_sequence_sha256"]
    assert identity["residue_mapping"][0]["mapping_mode"] == ("complete_sequence_order")
    assert identity["target_interface_positions"] == [1]
    assert changed_payload["identity"]["target_interface_positions"] == []
    assert changed_payload["identity"]["interface_distance_cutoff"] == 3.0


@pytest.mark.parametrize(
    ("observed", "expected", "author_ids", "positions", "mode"),
    [
        ("ACA", "ACA", [10, 20, 30], [1, 2, 3], "complete_sequence_order"),
        ("AD", "ACD", [10, 30], [1, 3], "unique_exact_subsequence"),
        ("AD", "ACD", [1, 3], [1, 3], "author_residue_ids"),
    ],
)
def test_chain_mapping_is_whole_chain_and_sequence_validated(
    observed: str,
    expected: str,
    author_ids: list[int],
    positions: list[int],
    mode: str,
) -> None:
    assert _chain_sequence_positions(
        observed_sequence=observed,
        expected_sequence=expected,
        author_residue_ids=author_ids,
    ) == (positions, mode)


@pytest.mark.parametrize(
    ("observed", "expected", "author_ids", "message"),
    [
        ("AA", "AAA", [10, 20], "ambiguous"),
        ("AAD", "ACD", [10, 20, 30], "does not match"),
        ("AE", "ACD", [10, 30], "not an exact subsequence"),
    ],
)
def test_chain_mapping_rejects_ambiguous_or_mismatched_structures(
    observed: str,
    expected: str,
    author_ids: list[int],
    message: str,
) -> None:
    with pytest.raises(InterfaceError, match=message):
        _chain_sequence_positions(
            observed_sequence=observed,
            expected_sequence=expected,
            author_residue_ids=author_ids,
        )


def test_derivative_failure_preserves_biotite_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _structure(tmp_path / "source.pdb")

    def fail_derivation(*_args, **_kwargs):
        raise RuntimeError("simulated bundle failure")

    monkeypatch.setattr(
        "af3_binder_filter.interface.materialize_derived_structures",
        fail_derivation,
    )
    row = analyze_interface_geometry(
        _job(),
        UnifiedPrediction(
            "job",
            "alphafold3",
            "success",
            best_model_path=model,
        ),
        distance=5.0,
        sasa_point_number=20,
        derived_structure_dir=tmp_path / "derived",
    )

    assert row["interface_status"] == "success"
    assert row["interface_contact_pair_count"] == 1
    assert row["derived_structure_status"] == "error"
    assert row["derived_structure_error"] == "simulated bundle failure"
    assert "rosetta_input_pdb" not in row


def test_row_manifest_binding_rejects_every_identity_mismatch(
    tmp_path: Path,
) -> None:
    model, interface_row = _derive(tmp_path)
    effective = _effective_row(interface_row, model)
    assert validated_artifacts_from_row(effective) is not None

    changes = {
        "job_name": "other-job",
        "effective_backend": "opendde",
        "target_chain": "C",
        "target_sequence": "AAC",
        "effective_derived_source_model_sha256": "0" * 64,
        "effective_derived_structure_id": "1" * 64,
        "effective_derived_interface_distance_cutoff": 7.0,
        # Numeric position still matches the manifest; only the chain is wrong.
        "effective_target_interface_residues": "B:1",
        "effective_normalized_binder_pdb_path": str(tmp_path / "wrong.pdb"),
    }
    for key, value in changes.items():
        tampered = dict(effective)
        tampered[key] = value
        assert validated_artifacts_from_row(tampered) is None, key

    model.write_text(model.read_text() + "REMARK changed after derivation\n")
    assert validated_artifacts_from_row(effective) is None


def test_empty_job_name_falls_back_to_scalar_job_id_for_all_consumers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model, interface_row = _derive(tmp_path)
    row = _effective_row(interface_row, model)
    row["job_name"] = None
    row["job_id"] = "job"

    assert validated_artifacts_from_row(row) is not None

    def fail_raw_parse(*_args, **_kwargs):
        raise AssertionError("validated fallback job_id should select derivatives")

    monkeypatch.setattr(
        "af3_binder_filter.clustering.load_protein_complex",
        fail_raw_parse,
    )
    binder_dir, _complex_dir = prepare_foldseek_inputs(
        (_job(),),
        {"job": model},
        work_dir=tmp_path / "fallback_foldseek",
        rows=(row,),
    )
    assert (binder_dir / "job.pdb").is_file()

    _fasta, manifest = write_esm_inputs(
        (_job(),),
        (
            UnifiedPrediction(
                "job",
                "alphafold3",
                "success",
                best_model_path=model,
            ),
        ),
        tmp_path / "fallback_esm",
        structure_rows=(row,),
    )
    assert json.loads(manifest.read_text())[0]["structure_path"] == (
        "/inputs/derived_structures/job.pdb"
    )

    invalid = dict(row)
    invalid["job_name"] = ["job"]
    invalid["job_id"] = None
    assert validated_artifacts_from_row(invalid) is None


@pytest.mark.parametrize(
    "corruption",
    [
        "non_finite_coordinate",
        "coordinate_shape",
        "wrong_chain",
        "duplicate_position",
        "non_boolean_interface",
    ],
)
def test_npz_semantic_corruption_is_rejected(
    tmp_path: Path,
    corruption: str,
) -> None:
    _model, interface_row = _derive(tmp_path / corruption)
    manifest_path = Path(str(interface_row["derived_structure_manifest_path"]))
    coordinate_path = Path(str(interface_row["derived_coordinates_path"]))
    with np.load(coordinate_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if corruption == "non_finite_coordinate":
        arrays["ca_coord"][0, 0] = np.nan
    elif corruption == "coordinate_shape":
        arrays["ca_coord"] = arrays["ca_coord"][:, :2]
    elif corruption == "wrong_chain":
        arrays["chain_id"][0] = "Z"
    elif corruption == "duplicate_position":
        arrays["sequence_position"][1] = arrays["sequence_position"][0]
    else:
        arrays["is_interface"] = arrays["is_interface"].astype(np.int8)
    with coordinate_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)

    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["coordinates"].update(
        {
            "sha256": file_sha256(coordinate_path),
            "size": coordinate_path.stat().st_size,
        }
    )
    manifest_path.write_text(json.dumps(manifest))
    assert validate_derived_manifest(manifest_path) is None


def test_same_content_writers_publish_one_complete_bundle(tmp_path: Path) -> None:
    model = _structure(tmp_path / "source.pdb")

    def analyze() -> dict[str, object]:
        return analyze_interface_geometry(
            _job(),
            UnifiedPrediction(
                "job",
                "alphafold3",
                "success",
                best_model_path=model,
            ),
            distance=5.0,
            sasa_point_number=20,
            derived_structure_dir=tmp_path / "derived",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(lambda _index: analyze(), range(2)))

    assert {row["interface_status"] for row in rows} == {"success"}
    assert {row["derived_structure_status"] for row in rows} == {"success"}
    assert len({row["derived_structure_id"] for row in rows}) == 1
    assert sorted(row["derived_structure_cache_hit"] for row in rows) == [False, True]
    assert (
        validate_derived_manifest(Path(str(rows[0]["derived_structure_manifest_path"]))) is not None
    )


def test_source_mutation_during_parse_is_rejected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _structure(tmp_path / "source.pdb")
    from af3_binder_filter import interface as interface_module

    original_loader = interface_module.load_protein_complex

    def mutating_loader(path: Path, **kwargs):
        array = original_loader(path, **kwargs)
        path.write_text(path.read_text() + "REMARK changed during parse\n")
        return array

    monkeypatch.setattr(interface_module, "load_protein_complex", mutating_loader)
    row = analyze_interface_geometry(
        _job(),
        UnifiedPrediction(
            "job",
            "alphafold3",
            "success",
            best_model_path=model,
        ),
        sasa_point_number=20,
        derived_structure_dir=tmp_path / "derived",
    )
    assert row["interface_status"] == "error"
    assert "changed while it was being parsed" in str(row["interface_error"])
    assert row["source_model_provenance_status"] == "changed"
    assert row["source_model_sha256_preparse"]
    assert row["source_model_sha256_observed"]


def test_source_mutation_during_bundle_build_preserves_geometry(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model = _structure(tmp_path / "source.pdb")
    from af3_binder_filter import derived_structures as derived_module

    original_writer = derived_module._write_bundle

    def mutating_writer(*args, **kwargs):
        original_writer(*args, **kwargs)
        source = Path(kwargs["source_model_path"])
        source.write_text(source.read_text() + "REMARK changed during build\n")

    monkeypatch.setattr(derived_module, "_write_bundle", mutating_writer)
    row = analyze_interface_geometry(
        _job(),
        UnifiedPrediction(
            "job",
            "alphafold3",
            "success",
            best_model_path=model,
        ),
        sasa_point_number=20,
        derived_structure_dir=tmp_path / "derived",
    )
    assert row["interface_status"] == "error"
    assert row["interface_contact_pair_count"] == 1
    assert row["derived_structure_status"] == "error"
    assert row["source_model_provenance_status"] == "changed"
    assert row["source_model_sha256_preparse"] != row["source_model_sha256_observed"]
    assert "source model changed while building derivatives" in str(row["derived_structure_error"])
    selection_input = {
        "job_name": "job",
        "target_chain": "A",
        "binder_chain": "B",
        "target_sequence": "AAA",
        "binder_sequence": "AAA",
        "primary_status": "success",
        "primary_best_model_path": str(model),
        "primary_final_pass": True,
        **{f"primary_{key}": value for key, value in row.items()},
    }
    selected = apply_effective_backend(selection_input)
    assert selected["effective_backend"] is None
    assert selected["effective_selection_reason"] == "no_eligible_backend"

    binder_dir, complex_dir = prepare_foldseek_inputs(
        (_job(),),
        {"job": model},
        work_dir=tmp_path / "blocked_foldseek",
        rows=(selected,),
    )
    assert not (binder_dir / "job.pdb").exists()
    assert not (complex_dir / "job.pdb").exists()
    _fasta, manifest = write_esm_inputs(
        (_job(),),
        (
            UnifiedPrediction(
                "job",
                "alphafold3",
                "success",
                best_model_path=model,
            ),
        ),
        tmp_path / "blocked_esm",
        structure_rows=(selected,),
    )
    assert json.loads(manifest.read_text()) == []


def test_esm_cache_is_bound_to_validated_effective_structure(
    tmp_path: Path,
) -> None:
    model, interface_row = _derive(tmp_path)
    job = _job()
    prediction = UnifiedPrediction(
        "job",
        "alphafold3",
        "success",
        best_model_path=model,
    )
    effective = _effective_row(interface_row, model)
    row = {
        "job_name": "job",
        "esmfold_status": "missing",
        "esm_if_status": "not_available",
        "esm_effective_backend": "alphafold3",
        "esm_effective_derived_structure_id": interface_row["derived_structure_id"],
        "esm_effective_source_model_sha256": interface_row["derived_source_model_sha256"],
    }
    cache = tmp_path / "esm_rows.csv"

    def write_cache(value: dict[str, object]) -> None:
        with cache.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(value))
            writer.writeheader()
            writer.writerow(value)

    write_cache(row)
    assert (
        load_cached_esm_rows(
            cache,
            (job,),
            (prediction,),
            require_esmfold=False,
            require_inverse_folding=False,
            structure_rows=(effective,),
        )
        is not None
    )

    for field in (
        "esm_effective_backend",
        "esm_effective_derived_structure_id",
        "esm_effective_source_model_sha256",
    ):
        tampered = dict(row)
        tampered[field] = "wrong"
        write_cache(tampered)
        assert (
            load_cached_esm_rows(
                cache,
                (job,),
                (prediction,),
                require_esmfold=False,
                require_inverse_folding=False,
                structure_rows=(effective,),
            )
            is None
        )


def test_chain_ca_errors_when_requested_chain_is_absent(tmp_path: Path) -> None:
    model = _structure(tmp_path / "source.pdb")
    with pytest.raises(ValueError, match="chain 'Z' is absent"):
        _chain_ca(model, "Z")


def test_fold_comparison_pairs_missing_residues_by_sequence_position(
    tmp_path: Path,
) -> None:
    coordinates = {
        1: [0.0, 0.0, 0.0],
        2: [50.0, 2.0, 1.0],
        3: [2.0, 8.0, 0.5],
        4: [5.0, 1.0, 6.0],
    }
    complete = _ca_trace(
        tmp_path / "complete.pdb",
        [1, 2, 3, 4],
        [coordinates[position] for position in (1, 2, 3, 4)],
    )
    missing = _ca_trace(
        tmp_path / "missing.pdb",
        [1, 3, 4],
        [coordinates[position] for position in (1, 3, 4)],
    )

    rmsd, tm = _fold_comparison(_chain_ca(missing), _chain_ca(complete))
    assert rmsd == pytest.approx(0.0, abs=1e-12)
    assert tm == pytest.approx(1.0, abs=1e-12)

    non_unique = _ca_trace(
        tmp_path / "non_unique.pdb",
        [1, 2, 1],
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="unique positive"):
        _chain_ca(non_unique)


def test_secondary_structure_change_invalidates_cached_comparison(
    tmp_path: Path,
) -> None:
    effective_model, interface_row = _derive(tmp_path / "effective")
    secondary_model = _structure(tmp_path / "secondary.pdb", binder_shift=2.0)
    job = _job()
    effective_prediction = UnifiedPrediction(
        "job",
        "alphafold3",
        "success",
        best_model_path=effective_model,
    )
    secondary_prediction = UnifiedPrediction(
        "job",
        "opendde",
        "success",
        best_model_path=secondary_model,
    )
    effective_row = _effective_row(interface_row, effective_model)
    output = tmp_path / "esm_output"
    esmfold = output / "esmfold"
    esmfold.mkdir(parents=True)
    esmfold_model = esmfold / "job_chain_B.pdb"
    esmfold_model.write_bytes(Path(str(interface_row["normalized_binder_pdb_path"])).read_bytes())
    rows = collect_esm_rows(
        (job,),
        (effective_prediction,),
        output,
        structure_rows=(effective_row,),
    )
    rows = add_esmfold_backend_comparison(
        rows,
        (job,),
        (effective_prediction,),
        output,
        label="primary",
    )
    rows = add_esmfold_backend_comparison(
        rows,
        (job,),
        (secondary_prediction,),
        output,
        label="secondary",
    )
    cache = tmp_path / "esm_comparisons.csv"
    with cache.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert (
        load_cached_esm_rows(
            cache,
            (job,),
            (effective_prediction,),
            require_esmfold=False,
            require_inverse_folding=False,
            structure_rows=(effective_row,),
            primary_predictions=(effective_prediction,),
            secondary_predictions=(secondary_prediction,),
        )
        is not None
    )

    secondary_model.write_text(
        secondary_model.read_text() + "REMARK non-effective backend changed\n"
    )
    assert (
        load_cached_esm_rows(
            cache,
            (job,),
            (effective_prediction,),
            require_esmfold=False,
            require_inverse_folding=False,
            structure_rows=(effective_row,),
            primary_predictions=(effective_prediction,),
            secondary_predictions=(secondary_prediction,),
        )
        is None
    )


def test_consensus_uses_validated_coordinate_npz_and_falls_back_safely(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary_model, primary_interface = _derive(
        tmp_path / "primary",
        backend="alphafold3",
    )
    secondary_model, secondary_interface = _derive(
        tmp_path / "secondary",
        backend="opendde",
        binder_shift=1.0,
    )
    primary = _effective_row(primary_interface, primary_model)
    secondary = _effective_row(
        secondary_interface,
        secondary_model,
        backend="opendde",
    )
    calls: list[Path] = []
    from af3_binder_filter import consensus as consensus_module

    original_loader = consensus_module.load_protein_complex

    def recording_loader(path: Path, **kwargs):
        calls.append(Path(path))
        return original_loader(path, **kwargs)

    monkeypatch.setattr(consensus_module, "load_protein_complex", recording_loader)
    contacts = frozenset({1})
    settings = ConsensusSettings(
        target_alignment_min_residues=3,
        target_alignment_min_fraction=1.0,
    )
    metrics = structure_consensus_metrics_from_rows(
        primary,
        secondary,
        target_chain="A",
        binder_chain="B",
        primary_target_contacts=contacts,
        secondary_target_contacts=contacts,
        primary_binder_contacts=contacts,
        secondary_binder_contacts=contacts,
        settings=settings,
        primary_prefix="effective",
        secondary_prefix="effective",
    )
    assert metrics["consensus_coordinate_source"] == "derived_cache"
    assert calls == []

    with pytest.raises(ValueError, match="only 3 target residues"):
        structure_consensus_metrics_from_rows(
            primary,
            secondary,
            target_chain="A",
            binder_chain="B",
            primary_target_contacts=contacts,
            secondary_target_contacts=contacts,
            primary_binder_contacts=contacts,
            secondary_binder_contacts=contacts,
            settings=ConsensusSettings(target_alignment_min_residues=4),
            primary_prefix="effective",
            secondary_prefix="effective",
        )
    assert calls == []

    secondary["effective_derived_structure_id"] = "invalid"
    with pytest.raises(
        DerivedStructureValidationError,
        match="effective derived structure no longer validates",
    ):
        structure_consensus_metrics_from_rows(
            primary,
            secondary,
            target_chain="A",
            binder_chain="B",
            primary_target_contacts=contacts,
            secondary_target_contacts=contacts,
            primary_binder_contacts=contacts,
            secondary_binder_contacts=contacts,
            settings=settings,
            primary_prefix="effective",
            secondary_prefix="effective",
        )
    assert calls == []

    # A derivative that was never successfully published remains eligible for
    # the explicit raw-model compatibility path.
    secondary["effective_derived_structure_status"] = "error"
    fallback = structure_consensus_metrics_from_rows(
        primary,
        secondary,
        target_chain="A",
        binder_chain="B",
        primary_target_contacts=contacts,
        secondary_target_contacts=contacts,
        primary_binder_contacts=contacts,
        secondary_binder_contacts=contacts,
        settings=settings,
        primary_prefix="effective",
        secondary_prefix="effective",
    )
    assert fallback["consensus_coordinate_source"] == "raw_structure_fallback"
    assert calls == [primary_model, secondary_model]


def test_consensus_rows_consumes_validated_derivatives_without_raw_reparse(
    tmp_path: Path,
    monkeypatch,
) -> None:
    primary_model, primary_interface = _derive(
        tmp_path / "primary_rows",
        backend="alphafold3",
    )
    secondary_model, secondary_interface = _derive(
        tmp_path / "secondary_rows",
        backend="opendde",
        binder_shift=1.0,
    )
    common = {
        "job_name": "job",
        "job_status": "success",
        "target_chain": "A",
        "binder_chain": "B",
        "target_sequence": "AAA",
        "binder_sequence": "AAA",
    }
    primary = {
        **common,
        "backend": "alphafold3",
        "best_model_path": str(primary_model),
        **primary_interface,
    }
    secondary = {
        **common,
        "backend": "opendde",
        "best_model_path": str(secondary_model),
        **secondary_interface,
    }

    def fail_raw_parse(*_args, **_kwargs):
        raise AssertionError("consensus_rows must consume coordinate NPZ artifacts")

    monkeypatch.setattr(
        "af3_binder_filter.consensus.load_protein_complex",
        fail_raw_parse,
    )
    result = consensus_rows(
        (primary,),
        (secondary,),
        ConsensusSettings(
            target_alignment_min_residues=3,
            target_alignment_min_fraction=1.0,
        ),
    )

    assert result[0]["consensus_status"] == "success", result[0].get("consensus_error")
    assert result[0]["consensus_coordinate_source"] == "derived_cache"
    assert result[0]["primary_derived_structure_status"] == "success"
    assert result[0]["secondary_derived_structure_status"] == "success"
