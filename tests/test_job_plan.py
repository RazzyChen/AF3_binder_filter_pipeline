import json
import os
from pathlib import Path

import pytest

from af3_binder_filter.config import AerithConfig
from af3_binder_filter.csv_input import CsvInputError, read_target_sequence
from af3_binder_filter.jobs import (
    build_job_plan,
    file_asset_identity,
    job_fingerprint,
    parse_epitope_residues,
    run_fingerprint,
    run_provenance,
    scientific_config_identity,
)


def _write(path: Path, rows: str) -> Path:
    path.write_text("sample_no,run_name,binder_sequence,target_seq\n" + rows)
    return path


def _lightweight_scientific_config(tmp_path: Path, csv_path: Path) -> AerithConfig:
    config = AerithConfig()
    config.project.csv_path = str(csv_path)
    config.backend.model_dir = str(tmp_path / "missing-models")
    config.features.mmseqs_dir = str(tmp_path / "missing-mmseqs")
    config.features.pdb_seqres_fasta = str(tmp_path / "missing-pdb-seqres")
    config.features.mmcif_dir = str(tmp_path / "missing-mmcif")
    config.scoring.esm.enabled = False
    config.interface.rosetta.binary = str(tmp_path / "missing-rosetta")
    config.interface.rosetta.database = str(tmp_path / "missing-rosetta-db")
    return config


def test_plan_validates_target_before_applying_limit(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "input.csv",
        "1,one,ACDE,LMNP\n2,two,FGHI,QRST\n",
    )
    config = _lightweight_scientific_config(tmp_path, csv_path)
    config.project.limit = 1

    with pytest.raises(CsvInputError, match="share one target"):
        build_job_plan(config)
    with pytest.raises(CsvInputError, match="same target"):
        read_target_sequence(csv_path)


def test_limit_is_applied_to_single_immutable_plan(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "input.csv",
        "\n1,one,ACDE,LMNP\n\n2,two,FGHI,LMNP\n",
    )
    config = _lightweight_scientific_config(tmp_path, csv_path)
    config.project.limit = 1

    plan = build_job_plan(config)

    assert len(plan.jobs) == 1
    assert plan.total_csv_jobs == 2
    assert plan.jobs[0].source_row_number == 3


def test_duplicate_sanitized_names_are_rejected(tmp_path: Path) -> None:
    csv_path = _write(
        tmp_path / "input.csv",
        "x,a,ACDE,LMNP\nx,b,FGHI,LMNP\n",
    )
    config = AerithConfig()
    config.project.csv_path = str(csv_path)

    with pytest.raises(CsvInputError, match="duplicate sanitized job name"):
        build_job_plan(config)


def test_fingerprint_changes_with_backend_or_sequence(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    config = AerithConfig()
    config.project.csv_path = str(csv_path)
    job = build_job_plan(config).jobs[0]
    original = job_fingerprint(job, config, feature_fingerprint="features")
    config.backend.name = "opendde"
    changed_backend = job_fingerprint(job, config, feature_fingerprint="features")

    assert original != changed_backend
    assert original != job_fingerprint(job, config, feature_fingerprint="new-features")


def test_fingerprint_changes_with_secondary_checkpoint(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    general = tmp_path / "opendde.pt"
    abag = tmp_path / "opendde_abag.pt"
    general.write_bytes(b"general")
    abag.write_bytes(b"abag")
    config = AerithConfig()
    config.project.csv_path = str(csv_path)
    config.secondary_backend.name = "opendde"
    config.secondary_backend.checkpoint_path = str(general)
    job = build_job_plan(config).jobs[0]
    original = job_fingerprint(job, config, feature_fingerprint="features")

    config.secondary_backend.checkpoint_path = str(abag)

    assert original != job_fingerprint(job, config, feature_fingerprint="features")


def test_epitope_parser_checks_one_based_range() -> None:
    assert parse_epitope_residues("1-3,5", target_length=5) == {1, 2, 3, 5}
    with pytest.raises(ValueError, match="outside target sequence"):
        parse_epitope_residues("0,6", target_length=5)


def test_run_fingerprint_covers_downstream_scientific_settings(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    config = _lightweight_scientific_config(tmp_path, csv_path)
    plan = build_job_plan(config)
    original = run_fingerprint(plan, config)

    config.interface.distance = 4.5
    interface_changed = run_fingerprint(plan, config)
    config.interface.distance = 5.0
    config.interface.rosetta.score_function = "beta_nov16"
    rosetta_changed = run_fingerprint(plan, config)
    config.interface.rosetta.score_function = "ref2015"
    config.clustering.epitope_jaccard_threshold = 0.75
    clustering_changed = run_fingerprint(plan, config)

    assert len({original, interface_changed, rosetta_changed, clustering_changed}) == 4
    provenance = run_provenance(plan, config)
    assert provenance["output_schema"]["decision_columns_sha256"]
    assert provenance["output_schema"]["backend_review_columns_sha256"]
    assert provenance["aerith"]["package_version"]
    assert provenance["aerith"]["runtime_source_sha256"]


def test_run_fingerprint_excludes_paths_gpu_and_worker_counts(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    config = _lightweight_scientific_config(tmp_path, csv_path)
    plan = build_job_plan(config)
    original = run_fingerprint(plan, config)

    config.project.work_dir = str(tmp_path / "different-work")
    config.project.output_dir = str(tmp_path / "different-output")
    config.project.results_dir = str(tmp_path / "different-results")
    config.project.prune = True
    config.project.allow_partial = True
    config.runtime.force = True
    config.runtime.dry_run = True
    config.runtime.gpu_ids = [3, 1]
    config.runtime.gpu_busy_threshold_mib = 999
    config.runtime.geometry_max_workers = 12
    config.features.threads = 64
    config.interface.rosetta.max_workers = 32
    config.clustering.max_workers = 2
    config.project.adopt_legacy = True

    assert run_fingerprint(plan, config) == original


def test_run_fingerprint_uses_csv_content_not_csv_absolute_path(tmp_path: Path) -> None:
    first_path = _write(tmp_path / "first.csv", "1,a,ACDE,LMNP\n")
    (tmp_path / "nested").mkdir()
    second_path = _write(tmp_path / "nested" / "second.csv", "1,a,ACDE,LMNP\n")
    first = _lightweight_scientific_config(tmp_path, first_path)
    second = _lightweight_scientific_config(tmp_path, second_path)

    assert run_fingerprint(build_job_plan(first), first) == run_fingerprint(
        build_job_plan(second), second
    )

    second_path.write_text(second_path.read_text() + "\n")
    assert run_fingerprint(build_job_plan(first), first) != run_fingerprint(
        build_job_plan(second), second
    )


def test_file_asset_identity_is_path_free_and_detects_content_mutation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one" / "checkpoint.pt"
    second = tmp_path / "two" / "relocated.pt"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"same-content")
    second.write_bytes(first.read_bytes())
    first_stat = first.stat()
    os.utime(second, ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns))

    assert file_asset_identity(first) == file_asset_identity(second)
    original_stat = second.stat()
    second.write_bytes(b"evil-content")
    os.utime(second, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    assert first.stat().st_size == second.stat().st_size
    assert first.stat().st_mtime_ns == second.stat().st_mtime_ns
    assert file_asset_identity(first) != file_asset_identity(second)


def test_scientific_identity_contains_no_raw_host_paths(tmp_path: Path) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    checkpoint = tmp_path / "assets" / "af3.bin"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"weights")
    config = _lightweight_scientific_config(tmp_path, csv_path)
    config.backend.model_dir = str(checkpoint.parent)

    identity_text = str(scientific_config_identity(config))

    assert str(tmp_path) not in identity_text


def test_af3_target_data_identity_includes_referenced_msa_content(
    tmp_path: Path,
) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    msa_path = tmp_path / "target.a3m"
    msa_path.write_text(">query\nLMNP\n")
    target_data = tmp_path / "target_data.json"
    target_data.write_text(
        json.dumps(
            {
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "LMNP",
                            "unpairedMsaPath": "target.a3m",
                        }
                    }
                ]
            }
        )
    )
    config = _lightweight_scientific_config(tmp_path, csv_path)
    config.backend.target_data_json = str(target_data)
    plan = build_job_plan(config)
    original = run_fingerprint(plan, config)

    msa_path.write_text(">query\nMNLP\n")

    assert run_fingerprint(plan, config) != original


def test_scientific_identity_uses_inspected_image_id_and_relocatable_commands(
    tmp_path: Path,
) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    config = _lightweight_scientific_config(tmp_path, csv_path)
    config.backend.image_id = "sha256:immutable-runtime"
    config.features.image_id = "sha256:immutable-runtime"
    config.backend.command = ["python", "/first/runtime/request.json"]
    plan = build_job_plan(config)
    original = run_fingerprint(plan, config)

    config.backend.image = "registry.example/renamed-runtime:alias"
    config.features.image = "registry.example/renamed-runtime:alias"
    config.backend.command = ["python", "/relocated/runtime/request.json"]

    assert run_fingerprint(plan, config) == original


def test_large_database_identity_is_bounded_labelled_and_content_sensitive(
    tmp_path: Path,
) -> None:
    csv_path = _write(tmp_path / "input.csv", "1,a,ACDE,LMNP\n")
    config = _lightweight_scientific_config(tmp_path, csv_path)
    mmseqs_dir = tmp_path / "mmseqs"
    mmseqs_dir.mkdir()
    database = mmseqs_dir / "uniref90_padded"
    database.write_bytes(b"A" * (4 * 1024 * 1024 + 17))
    config.features.mmseqs_dir = str(mmseqs_dir)
    config.features.use_environment_database = False
    config.features.template_database = "uniref90_padded"
    original_stat = database.stat()

    original = scientific_config_identity(config)
    member = original["features"]["database_release"]["mmseqs_prefixes"][
        "entries"
    ][0]["content_identity"]
    assert member["identity_mode"] == "bounded-content-sample-v1"
    assert member["sample_sha256"]

    with database.open("r+b") as handle:
        handle.write(b"B")
    os.utime(database, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    changed = scientific_config_identity(config)
    assert changed != original
