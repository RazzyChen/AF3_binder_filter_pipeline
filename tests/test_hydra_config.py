from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from af3_binder_filter.cli import app
from af3_binder_filter.config import (
    AerithConfig,
    ConfigError,
    compose_hydra_config,
    validate_hydra_config,
)
from af3_binder_filter.config_tools import (
    EnvironmentDetection,
    write_initial_config,
    write_minimal_production_config,
)
from af3_binder_filter.workflow import (
    PipelineExecutionError,
    create_run_context,
    run_pipeline,
)


def _database(root: Path) -> Path:
    mmseqs = root / "mmseqs"
    mmseqs.mkdir(parents=True)
    for name in (
        "uniref90_padded",
        "mgnify_padded",
        "small_bfd_padded",
        "pdb_seqres_padded",
    ):
        (mmseqs / name).write_text("db")
    (root / "pdb_seqres_2022_09_28.fasta").write_text(">x\nA\n")
    (root / "mmcif_files").mkdir()
    return root


def _csv(path: Path) -> Path:
    path.write_text(
        "sample_no,run_name,binder_sequence,target_seq\n"
        "1,run,ACDE,LMNP\n"
    )
    return path


def test_minimal_production_config_composes_with_structured_defaults(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "production"
    csv_path = _csv(tmp_path / "input.csv")
    config_path = write_minimal_production_config(
        tmp_path / "minimal.yaml",
        project_root=project_root,
        csv_path=csv_path,
        gpu_ids=(0, 2),
        epitope_residues="1-2",
    )

    payload = yaml.safe_load(config_path.read_text())
    assert set(payload) == {"defaults", "project", "runtime", "interface", "hydra"}
    assert set(payload["project"]) == {
        "csv_path",
        "work_dir",
        "output_dir",
        "results_dir",
    }

    config, _resolved = compose_hydra_config(config_path)
    assert config.backend.name == "alphafold3"
    assert not hasattr(config.backend, "enabled")
    assert config.secondary_backend.name == "opendde"
    assert config.secondary_backend.enabled is True
    assert not hasattr(config.features, "enabled")
    assert not hasattr(config.scoring.esm, "hard_filter")
    assert not hasattr(config.consensus, "hard_filter")
    assert config.consensus.explicit_different_interface_pair_jaccard == 0.30
    assert config.secondary_backend.checkpoint_path.endswith("/opendde.pt")
    assert config.project.csv_path == str(csv_path.resolve())
    assert config.project.work_dir == str(project_root.resolve() / "work")
    assert config.project.output_dir == str(project_root.resolve() / "outputs")
    assert config.project.results_dir == str(project_root.resolve() / "results")
    assert config.project.target_chain == "A"
    assert config.project.binder_chain == "B"
    assert config.project.allow_partial is False
    assert config.runtime.gpu_ids == [0, 2]
    assert config.interface.epitope_residues == "1-2"
    assert config.interface.minimum_epitope_coverage == 0.30
    assert config.interface.minimum_epitope_purity is None
    assert validate_hydra_config(config, check_paths=False).ok


def test_removed_legacy_commands_are_not_registered() -> None:
    registered = {command.name for command in app.registered_commands}

    assert registered == {
        "analyze-interface",
        "build-runtime-image",
        "cluster",
        "pipeline",
        "prepare-features",
    }


def test_geometry_and_deprecated_purity_validation() -> None:
    config = AerithConfig()
    config.interface.geometry_engine = "unsupported"
    report = validate_hydra_config(config, check_paths=False)
    assert "interface.geometry_engine must be biotite" in report.errors
    assert any("minimum_epitope_purity is deprecated" in item for item in report.warnings)

    config.interface.geometry_engine = "biotite"
    config.interface.minimum_epitope_purity = 0.30
    report = validate_hydra_config(config, check_paths=False)
    assert any("minimum_epitope_purity is no longer supported" in item for item in report.errors)


def test_database_validation_uses_configured_names_and_environment_switch(
    tmp_path: Path,
) -> None:
    config = AerithConfig()
    config.project.csv_path = str(_csv(tmp_path / "input.csv"))
    config.backend.model_dir = str(tmp_path / "models")
    Path(config.backend.model_dir).mkdir()
    config.scoring.esm.enabled = False
    config.interface.energy_engine = "none"

    database = tmp_path / "database"
    mmseqs = database / "custom_mmseqs"
    mmseqs.mkdir(parents=True)
    config.features.database_dir = str(database)
    config.features.mmseqs_dir = str(mmseqs)
    config.features.primary_database = "custom_primary"
    config.features.environment_database = "custom_environment"
    config.features.template_database = "custom_template"
    config.features.use_environment_database = False
    config.features.pdb_seqres_fasta = str(database / "custom_templates.fasta")
    config.features.mmcif_dir = str(database / "custom_mmcif")
    (mmseqs / "custom_primary").write_text("db")
    (mmseqs / "custom_template").write_text("db")
    Path(config.features.pdb_seqres_fasta).write_text(">x\nA\n")
    Path(config.features.mmcif_dir).mkdir()

    report = validate_hydra_config(config)
    assert report.ok, report.errors
    assert not any("custom_environment" in item for item in report.errors)

    config.features.use_environment_database = True
    report = validate_hydra_config(config)
    assert any("custom_environment" in item for item in report.errors)


def test_missing_configured_rosetta_is_an_error(tmp_path: Path) -> None:
    config = AerithConfig()
    config.project.csv_path = str(_csv(tmp_path / "input.csv"))
    config.backend.model_dir = str(tmp_path / "models")
    Path(config.backend.model_dir).mkdir()
    config.scoring.esm.enabled = False
    database = _database(tmp_path / "database")
    config.features.database_dir = str(database)
    config.features.mmseqs_dir = str(database / "mmseqs")
    config.features.pdb_seqres_fasta = str(
        database / "pdb_seqres_2022_09_28.fasta"
    )
    config.features.mmcif_dir = str(database / "mmcif_files")
    config.interface.rosetta.binary = str(tmp_path / "missing_rosetta")
    config.interface.rosetta.database = str(tmp_path / "missing_rosetta_database")

    report = validate_hydra_config(config)

    assert any("Rosetta binary does not exist" in item for item in report.errors)
    assert any("Rosetta database does not exist" in item for item in report.errors)
    assert not any("Rosetta" in item for item in report.warnings)


def test_minimal_production_config_accepts_cli_only_field_overrides(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "production"
    csv_path = _csv(tmp_path / "input.csv")
    config_path = write_minimal_production_config(
        tmp_path / "minimal.yaml",
        project_root=project_root,
        csv_path=csv_path,
        gpu_ids=(0, 2),
    )

    context = create_run_context(
        config_path,
        dry_run=True,
        limit=1,
        overrides=["interface.distance=4.5"],
    )

    assert context.config.runtime.dry_run is True
    assert context.config.project.limit == 1
    assert context.config.interface.distance == 4.5


def test_config_create_cli_writes_minimal_yaml_and_protects_existing_file(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    output = tmp_path / "screen.yaml"
    project_root = tmp_path / "screen"
    csv_path = _csv(tmp_path / "input.csv")
    arguments = [
        "config",
        "create",
        "--output",
        str(output),
        "--project-root",
        str(project_root),
        "--csv",
        str(csv_path),
        "--secondary-backend",
        "protenix",
        "--gpu-ids",
        "0,2",
    ]

    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    assert "Wrote minimal production configuration" in result.output
    config, _resolved = compose_hydra_config(output)
    assert config.secondary_backend.name == "protenix"
    assert config.runtime.gpu_ids == [0, 2]

    original = output.read_text()
    result = runner.invoke(app, arguments)
    assert result.exit_code == 1
    assert "use --force" in result.output
    assert output.read_text() == original

    result = runner.invoke(app, [*arguments, "--force"])
    assert result.exit_code == 0, result.output


def test_generated_config_composes_primary_and_secondary_groups(tmp_path: Path) -> None:
    database = _database(tmp_path / "db")
    csv_path = _csv(tmp_path / "input.csv")
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(database_dir=str(database), gpu_indexes=(2,)),
        csv_path=str(csv_path),
    )

    config, _resolved = compose_hydra_config(
        config_path,
        secondary_backend="protenix",
        overrides=["interface.distance=4.5"],
    )

    assert config.backend.name == "alphafold3"
    assert config.secondary_backend.name == "protenix"
    assert config.secondary_backend.model == "protenix-v2"
    assert config.interface.distance == 4.5
    assert config.runtime.gpu_ids == [2]
    assert validate_hydra_config(config, check_paths=False).ok


@pytest.mark.parametrize(
    ("backend", "runtime_entry"),
    (("alphafold3", "af3"), ("protenix", "protenix"), ("opendde", "opendde")),
)
def test_all_backend_groups_use_unified_runtime_image(
    tmp_path: Path, backend: str, runtime_entry: str
) -> None:
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(),
        backend=backend,
    )

    config, _resolved = compose_hydra_config(config_path)

    assert config.backend.image == "aerith/fold-runtime:local"
    assert config.backend.runtime_entry == runtime_entry


def test_structured_config_rejects_wrong_type(tmp_path: Path) -> None:
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(),
    )
    text = config_path.read_text().replace("epitope_residues: null", "epitope_residues: null\n  distance: wrong")
    config_path.write_text(text)

    with pytest.raises(ConfigError, match="invalid Hydra configuration"):
        compose_hydra_config(config_path)


def test_dry_run_resolved_config_matches_manifest_backend(tmp_path: Path) -> None:
    database = _database(tmp_path / "db")
    csv_path = _csv(tmp_path / "input.csv")
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(database_dir=str(database), gpu_indexes=(1,)),
        csv_path=str(csv_path),
    )
    overrides = [
        f"project.work_dir={tmp_path / 'work'}",
        f"project.output_dir={tmp_path / 'output'}",
        f"project.results_dir={tmp_path / 'results'}",
    ]
    context = create_run_context(
        config_path,
        secondary_backend="opendde",
        overrides=overrides,
        dry_run=True,
    )

    assert run_pipeline(context) == []
    manifest = json.loads(context.manifest_path.read_text())
    resolved = (context.results_dir / "resolved_config.yaml").read_text()
    command = (
        context.results_dir
        / "stages"
        / "02_features"
        / "logs"
        / "prepare_features.command.txt"
    ).read_text()
    assert manifest["backend"] == "alphafold3"
    assert manifest["fingerprint"] == context.fingerprint
    assert manifest["output_schema_version"] == 3
    assert "secondary_backend:" in resolved
    assert "name: opendde" in resolved
    assert "--network none" in command
    assert f"{database}:/db:ro" in command


def test_preflight_failure_is_recorded_in_run_manifest(tmp_path: Path) -> None:
    database = _database(tmp_path / "db")
    csv_path = _csv(tmp_path / "input.csv")
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(database_dir=str(database)),
        csv_path=str(csv_path),
    )
    context = create_run_context(
        config_path,
        overrides=[
            f"project.results_dir={tmp_path / 'results'}",
            f"backend.model_dir={tmp_path / 'missing-models'}",
        ],
    )

    with pytest.raises(PipelineExecutionError, match="preflight failed"):
        run_pipeline(context)

    manifest = json.loads(context.manifest_path.read_text())
    assert manifest["status"] == "error"
    assert manifest["stage_status"]["preflight"] == "error"
    assert "preflight failed" in manifest["errors"][0]


def test_af3_external_target_data_is_externalized_into_complex_inputs(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "db")
    csv_path = _csv(tmp_path / "input.csv")
    target_data = tmp_path / "target_data.json"
    target_data.write_text(
        json.dumps(
            {
                "name": "target_A",
                "sequences": [
                    {
                        "protein": {
                            "id": "A",
                            "sequence": "LMNP",
                            "unpairedMsa": ">query\nLMNP\n",
                            "pairedMsa": ">query\nLMNP\n",
                            "templates": [
                                {
                                    "mmcif": "data_template\n",
                                    "queryIndices": [0],
                                    "templateIndices": [0],
                                }
                            ],
                        }
                    }
                ],
            }
        )
    )
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(database_dir=str(database)),
        csv_path=str(csv_path),
    )
    work_dir = tmp_path / "work"
    context = create_run_context(
        config_path,
        dry_run=True,
        overrides=[
            f"backend.target_data_json={target_data}",
            f"project.work_dir={work_dir}",
            f"project.output_dir={tmp_path / 'output'}",
            f"project.results_dir={tmp_path / 'results'}",
        ],
    )

    assert run_pipeline(context) == []

    input_path = work_dir / context.run_id / "inputs" / "alphafold3" / (
        "sample_1_binder_candiate_complex_pred.json"
    )
    payload = json.loads(input_path.read_text())
    target = payload["sequences"][0]["protein"]
    binder = payload["sequences"][1]["protein"]
    assert Path(target["unpairedMsaPath"]).is_file()
    assert Path(target["templates"][0]["mmcifPath"]).is_file()
    assert binder["unpairedMsa"] == ">query\nACDE\n"
    assert binder["pairedMsa"] == ""


def test_af3_dry_run_plans_gpu_mmseqs_preprocessing_first(tmp_path: Path) -> None:
    database = _database(tmp_path / "db")
    csv_path = _csv(tmp_path / "input.csv")
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(database_dir=str(database)),
        csv_path=str(csv_path),
    )
    context = create_run_context(
        config_path,
        dry_run=True,
        overrides=[
            f"project.work_dir={tmp_path / 'work'}",
            f"project.output_dir={tmp_path / 'output'}",
            f"project.results_dir={tmp_path / 'results'}",
        ],
    )

    assert run_pipeline(context) == []

    feature_command = (
        context.results_dir
        / "stages"
        / "02_features"
        / "logs"
        / "prepare_features.command.txt"
    ).read_text()
    prediction_command = (
        context.results_dir
        / "stages"
        / "03_primary_prediction"
        / "logs"
        / "prediction.command.txt"
    ).read_text()
    assert " aerith/fold-runtime:local prepare-features " in feature_command
    assert "--use-gpu 1" in feature_command
    assert "GPU MMseqs2 preprocessing" in prediction_command
    assert {path.name for path in context.results_dir.glob("*.csv")} == {
        "all_results.csv",
        "backend_review.csv",
        "candidates.csv",
        "final_shortlist.csv",
    }
