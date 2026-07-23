import csv
import json
from pathlib import Path
from types import SimpleNamespace

import af3_binder_filter.workflow as workflow
import pytest
from typer.testing import CliRunner
from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.cli import app
from af3_binder_filter.config import AerithConfig, BackendSettings
from af3_binder_filter.config_tools import EnvironmentDetection, write_initial_config
from af3_binder_filter.features import FeatureBundle
from af3_binder_filter.io_utils import atomic_write_csv, atomic_write_json
from af3_binder_filter.jobs import JobSpec, file_sha256
from af3_binder_filter.manifest import write_job_manifest
from af3_binder_filter.workflow import (
    _backend_job_fingerprint,
    PipelineExecutionError,
    create_run_context,
    load_predictions_for_context,
    run_clustering_only,
    secondary_gate_job_ids,
)


def test_backend_job_fingerprint_uses_actual_backend_feature_bundle() -> None:
    context = SimpleNamespace(config=AerithConfig())
    backend = BackendSettings(
        name="opendde",
        model="opendde_v1",
        image="aerith/fold-runtime:local",
        image_id="sha256:image",
    )
    job = JobSpec(
        "job",
        "1",
        "run",
        "LMNP",
        "ACDE",
        "A",
        "B",
        2,
        42,
        "opendde",
        "opendde_v1",
    )

    first = _backend_job_fingerprint(context, job, "secondary-features-a", backend)
    second = _backend_job_fingerprint(context, job, "secondary-features-b", backend)

    assert first != second
    assert first == _backend_job_fingerprint(
        context, job, "secondary-features-a", backend
    )


def test_backend_job_fingerprint_uses_actual_checkpoint(tmp_path) -> None:
    context = SimpleNamespace(config=AerithConfig())
    general = tmp_path / "opendde.pt"
    abag = tmp_path / "opendde_abag.pt"
    general.write_bytes(b"general")
    abag.write_bytes(b"abag")
    backend = BackendSettings(
        name="opendde",
        model="opendde_v1",
        image="aerith/fold-runtime:local",
        image_id="sha256:image",
        checkpoint_path=str(general),
    )
    job = JobSpec(
        "job",
        "1",
        "run",
        "LMNP",
        "ACDE",
        "A",
        "B",
        2,
        42,
        "opendde",
        "opendde_v1",
    )
    original = _backend_job_fingerprint(
        context, job, "secondary-features", backend
    )

    backend.checkpoint_path = str(abag)

    assert original != _backend_job_fingerprint(
        context, job, "secondary-features", backend
    )


def test_matching_secondary_job_manifest_has_no_pending_job(
    tmp_path, monkeypatch
) -> None:
    context = SimpleNamespace(config=AerithConfig())
    backend = BackendSettings(
        name="opendde",
        model="opendde_v1",
        image="aerith/fold-runtime:local",
        image_id="sha256:image",
    )
    job = JobSpec(
        "job",
        "1",
        "run",
        "LMNP",
        "ACDE",
        "A",
        "B",
        2,
        42,
        "opendde",
        "opendde_v1",
    )
    model = tmp_path / "model.cif"
    model.write_text("data_model\n")
    prediction = UnifiedPrediction(
        "job",
        "opendde",
        "success",
        best_model_path=model,
        iptm=0.9,
    )

    class Adapter:
        def parse(self, _job, _output_root):
            return prediction

    monkeypatch.setattr(workflow, "output_adapter", lambda _name: Adapter())
    monkeypatch.setattr(workflow, "structure_has_chains", lambda *_args: True)
    output_root = tmp_path / "outputs"
    fingerprint = _backend_job_fingerprint(
        context, job, "secondary-features", backend
    )
    write_job_manifest(
        output_root / job.job_id,
        job=job,
        fingerprint=fingerprint,
        backend=backend.name,
        artifacts={"best_model_path": str(model)},
    )

    reusable, pending = workflow._reusable_predictions(
        context,
        [tmp_path / "input.json"],
        "secondary-features",
        jobs=[job],
        backend=backend,
        output_root=output_root,
    )

    assert pending == []
    assert reusable["job"].fingerprint_valid is True


def test_changed_feature_bytes_invalidate_prediction_with_same_generation_key(
    tmp_path: Path, monkeypatch
) -> None:
    context = SimpleNamespace(config=AerithConfig())
    backend = BackendSettings(
        name="opendde",
        model="opendde_v1",
        image="sha256:image",
        image_id="sha256:image",
    )
    job = JobSpec(
        "job",
        "1",
        "run",
        "LMNP",
        "ACDE",
        "A",
        "B",
        2,
        42,
        "opendde",
        "opendde_v1",
    )
    root = tmp_path / "features"
    template_dir = root / "templates"
    template_dir.mkdir(parents=True)
    pairing = root / "pairing.a3m"
    non_pairing = root / "non_pairing.a3m"
    hmmsearch = root / "hmmsearch.a3m"
    template_json = root / "af3_templates.json"
    for path in (pairing, non_pairing, hmmsearch):
        path.write_text(">query\nLMNP\n")
    template_json.write_text('{"templates": []}\n')
    bundle = FeatureBundle(
        sequence_sha256="sequence",
        cache_dir=root,
        pairing_a3m=pairing,
        non_pairing_a3m=non_pairing,
        hmmsearch_a3m=hmmsearch,
        fingerprint="same-generation-fingerprint",
        af3_templates_json=template_json,
        template_mmcif_dir=template_dir,
        source_mmcif_dir=tmp_path / "source-mmcif",
    )
    original_feature_identity = workflow._prediction_feature_identity(bundle)
    model = tmp_path / "model.cif"
    model.write_text("data_model\n")
    prediction = UnifiedPrediction(
        "job",
        "opendde",
        "success",
        best_model_path=model,
        iptm=0.9,
    )

    class Adapter:
        def parse(self, _job, _output_root):
            return prediction

    monkeypatch.setattr(workflow, "output_adapter", lambda _name: Adapter())
    monkeypatch.setattr(workflow, "structure_has_chains", lambda *_args: True)
    output_root = tmp_path / "outputs"
    write_job_manifest(
        output_root / job.job_id,
        job=job,
        fingerprint=_backend_job_fingerprint(
            context,
            job,
            original_feature_identity,
            backend,
        ),
        backend=backend.name,
        artifacts={"best_model_path": str(model)},
    )

    non_pairing.write_text(">query\nLMNP\n>changed\nLMNP\n")
    changed_feature_identity = workflow._prediction_feature_identity(bundle)
    reusable, pending = workflow._reusable_predictions(
        context,
        [tmp_path / "input.json"],
        changed_feature_identity,
        jobs=[job],
        backend=backend,
        output_root=output_root,
    )

    assert bundle.fingerprint == "same-generation-fingerprint"
    assert changed_feature_identity != original_feature_identity
    assert reusable == {}
    assert pending == [job]


def test_standalone_loader_accepts_prediction_stage_fingerprint(
    tmp_path, monkeypatch
) -> None:
    config = AerithConfig()
    config.backend.image_id = "sha256:image"
    config.project.output_dir = str(tmp_path / "outputs")
    config.project.work_dir = str(tmp_path / "work")
    job = JobSpec(
        "job",
        "1",
        "run",
        "LMNP",
        "ACDE",
        "A",
        "B",
        2,
        42,
        "alphafold3",
        "alphafold3",
    )
    context = SimpleNamespace(
        config=config,
        plan=SimpleNamespace(jobs=(job,), target_sequence="LMNP"),
        run_id="run",
    )
    model = tmp_path / "model.cif"
    model.write_text("data_model\n")
    prediction = UnifiedPrediction(
        "job",
        "alphafold3",
        "success",
        best_model_path=model,
    )

    class Adapter:
        def parse(self, _job, _output_root):
            return prediction

    monkeypatch.setattr(workflow, "output_adapter", lambda _name: Adapter())
    monkeypatch.setattr(workflow, "structure_has_chains", lambda *_args: True)
    monkeypatch.setattr(
        workflow,
        "_expected_feature_fingerprint",
        lambda *_args: "primary-features",
    )
    output_root = tmp_path / "outputs" / "run" / "alphafold3"
    write_job_manifest(
        output_root / job.job_id,
        job=job,
        fingerprint=_backend_job_fingerprint(
            context,
            job,
            "primary-features",
            config.backend,
        ),
        backend=config.backend.name,
        artifacts={"best_model_path": str(model)},
    )

    loaded, _rows = load_predictions_for_context(context)

    assert loaded[0].status == "success"
    assert loaded[0].best_model_path == model


def test_secondary_gate_allows_af3_structure_rescue_but_not_stale_metrics() -> None:
    predictions = [
        UnifiedPrediction(
            "rescue",
            "alphafold3",
            "error",
            iptm=0.70,
            error="missing required chains",
            fingerprint_valid=True,
        ),
        UnifiedPrediction(
            "stale",
            "alphafold3",
            "error",
            iptm=0.99,
            fingerprint_valid=False,
        ),
        UnifiedPrediction(
            "low",
            "alphafold3",
            "success",
            iptm=0.699,
            fingerprint_valid=True,
        ),
    ]

    assert secondary_gate_job_ids(predictions, 0.70) == {"rescue"}


def _run_config(tmp_path: Path) -> Path:
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "sample_no,run_name,binder_sequence,target_seq\n"
        "1,run,ACDE,LMNP\n"
    )
    return write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(),
        csv_path=str(csv_path),
    )


def _stable_image_id(_docker_bin: str, image: str) -> str:
    return "sha256:" + ("primary" if "runtime" in image else "image")


def _run_overrides(tmp_path: Path, *extra: str) -> list[str]:
    return [
        f"project.results_dir={tmp_path / 'results'}",
        f"backend.model_dir={tmp_path / 'missing-models'}",
        f"features.mmseqs_dir={tmp_path / 'missing-mmseqs'}",
        f"features.pdb_seqres_fasta={tmp_path / 'missing-pdb-seqres'}",
        f"features.mmcif_dir={tmp_path / 'missing-mmcif'}",
        "scoring.esm.enabled=false",
        "interface.energy_engine=none",
        *extra,
    ]


def test_primary_only_context_records_all_resolved_image_and_source_ids(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )

    resolved = (context.results_dir / "resolved_config.yaml").read_text()
    manifest = json.loads(context.manifest_path.read_text())
    assert "image_id: sha256:primary" in resolved
    assert context.config.backend.image == "sha256:primary"
    assert context.config.features.image == "sha256:primary"
    assert "image: sha256:primary" in resolved
    assert manifest["primary_image_id"] == "sha256:primary"
    assert manifest["resolved_config_sha256"] == file_sha256(
        context.results_dir / "resolved_config.yaml"
    )
    assert manifest["source_csv_sha256"]
    assert manifest["provenance"]["output_schema"]["version"] == 3


def test_pipeline_keyboard_interrupt_is_persisted_and_re_raised(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )

    def interrupt(_context):
        raise KeyboardInterrupt

    monkeypatch.setattr(workflow, "prepare_features_stage", interrupt)
    with pytest.raises(KeyboardInterrupt):
        workflow.run_pipeline(context)

    manifest = json.loads(context.manifest_path.read_text())
    assert manifest["status"] == "interrupted"
    assert manifest["stage_status"]["features"] == "interrupted"
    assert "pipeline interrupted by user" in manifest["errors"]


def test_explicit_run_id_rejects_fingerprint_change_before_any_write(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    config_path = _run_config(tmp_path)
    overrides = _run_overrides(tmp_path, "project.run_id=fixed-run")
    context = create_run_context(config_path, overrides=overrides, dry_run=True)
    resolved_path = context.results_dir / "resolved_config.yaml"
    manifest_path = context.manifest_path
    original_resolved = resolved_path.read_bytes()
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(PipelineExecutionError, match="different fingerprint"):
        create_run_context(
            config_path,
            overrides=[*overrides, "interface.distance=4.5"],
            dry_run=True,
        )

    assert resolved_path.read_bytes() == original_resolved
    assert manifest_path.read_bytes() == original_manifest


def test_explicit_run_id_rejects_nonempty_directory_without_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    config_path = _run_config(tmp_path)
    run_dir = tmp_path / "results" / "orphan-run"
    run_dir.mkdir(parents=True)
    sentinel = run_dir / "keep.txt"
    sentinel.write_text("do not overwrite")

    with pytest.raises(PipelineExecutionError, match="no valid manifest"):
        create_run_context(
            config_path,
            overrides=[
                *_run_overrides(tmp_path),
                "project.run_id=orphan-run",
            ],
            dry_run=True,
        )

    assert sentinel.read_text() == "do not overwrite"
    assert not (run_dir / "resolved_config.yaml").exists()


def test_standalone_cluster_rejects_tampered_candidate_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )
    model = tmp_path / "secondary-model.cif"
    model.write_text("data_secondary\n")
    manifest = workflow._existing_or_new_manifest(
        context,
        workflow._expected_feature_fingerprint(
            context.config, context.plan.target_sequence
        ),
    )
    workflow._persist_clustering_inputs(
        context,
        [
            {
                "job_name": context.plan.jobs[0].job_id,
                "candidate_pool": True,
                "effective_backend": "opendde",
                "effective_status": "success",
                "effective_best_model_path": str(model),
                "esmfold_status": "success",
                "esm_if_status": "success",
                "esm_if_perplexity": 2.5,
            }
        ],
        manifest,
    )
    _all_path, candidate_path = workflow._clustering_input_paths(context)
    candidate_path.write_text(candidate_path.read_text() + "tampered\n")
    original_manifest = context.manifest_path.read_bytes()
    original_resolved = (context.results_dir / "resolved_config.yaml").read_bytes()

    read_only_context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
        initialize_run=False,
    )

    assert context.manifest_path.read_bytes() == original_manifest
    assert (context.results_dir / "resolved_config.yaml").read_bytes() == original_resolved

    with pytest.raises(PipelineExecutionError, match="clustering_candidates"):
        run_clustering_only(read_only_context)
    assert context.manifest_path.read_bytes() == original_manifest


def test_stale_configured_image_id_is_rejected_before_run_directory_write(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    results = tmp_path / "results"

    with pytest.raises(PipelineExecutionError, match="does not match actual image ID"):
        create_run_context(
            _run_config(tmp_path),
            overrides=[
                *_run_overrides(tmp_path),
                "backend.image_id=sha256:stale",
            ],
            dry_run=True,
        )

    assert not results.exists()


def test_malformed_keyed_manifest_is_rejected_without_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    config_path = _run_config(tmp_path)
    overrides = _run_overrides(tmp_path, "project.run_id=malformed-run")
    context = create_run_context(config_path, overrides=overrides, dry_run=True)
    payload = json.loads(context.manifest_path.read_text())
    payload["job_fingerprints"] = ["keyed-but-not-a-mapping"]
    atomic_write_json(context.manifest_path, payload)
    original_manifest = context.manifest_path.read_bytes()
    original_resolved = (context.results_dir / "resolved_config.yaml").read_bytes()

    with pytest.raises(PipelineExecutionError, match="manifest is invalid"):
        create_run_context(config_path, overrides=overrides, dry_run=True)

    assert context.manifest_path.read_bytes() == original_manifest
    assert (context.results_dir / "resolved_config.yaml").read_bytes() == original_resolved


def test_stale_job_fingerprint_value_is_rejected_without_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    config_path = _run_config(tmp_path)
    overrides = _run_overrides(tmp_path, "project.run_id=stale-job-run")
    context = create_run_context(config_path, overrides=overrides, dry_run=True)
    payload = json.loads(context.manifest_path.read_text())
    job_id = context.plan.jobs[0].job_id
    payload["job_fingerprints"][job_id] = "stale-but-well-typed"
    atomic_write_json(context.manifest_path, payload)
    original_manifest = context.manifest_path.read_bytes()
    original_resolved = (context.results_dir / "resolved_config.yaml").read_bytes()

    with pytest.raises(PipelineExecutionError, match="manifest is invalid"):
        create_run_context(config_path, overrides=overrides, dry_run=True)

    assert context.manifest_path.read_bytes() == original_manifest
    assert (context.results_dir / "resolved_config.yaml").read_bytes() == original_resolved


def test_resolved_config_sha_is_verified_before_resume_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    config_path = _run_config(tmp_path)
    overrides = _run_overrides(tmp_path, "project.run_id=config-tamper")
    context = create_run_context(config_path, overrides=overrides, dry_run=True)
    resolved_path = context.results_dir / "resolved_config.yaml"
    resolved_path.write_text(resolved_path.read_text() + "# tampered\n")
    tampered = resolved_path.read_bytes()
    original_manifest = context.manifest_path.read_bytes()

    with pytest.raises(PipelineExecutionError, match="manifest is invalid"):
        create_run_context(config_path, overrides=overrides, dry_run=True)

    assert resolved_path.read_bytes() == tampered
    assert context.manifest_path.read_bytes() == original_manifest


def test_clustering_input_preserves_post_esm_fields_and_binds_effective_model(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )
    model = tmp_path / "effective-secondary.cif"
    model.write_text("data_effective\n")
    manifest = workflow._existing_or_new_manifest(
        context,
        workflow._expected_feature_fingerprint(
            context.config, context.plan.target_sequence
        ),
    )
    workflow._persist_clustering_inputs(
        context,
        [
            {
                "job_name": context.plan.jobs[0].job_id,
                "candidate_pool": True,
                "effective_backend": "opendde",
                "effective_status": "success",
                "effective_best_model_path": str(model),
                "esmfold_status": "success",
                "esm_if_status": "success",
                "esm_if_perplexity": 3.25,
            }
        ],
        manifest,
    )

    all_rows, candidate_rows = workflow._validated_clustering_inputs(
        context, manifest
    )

    assert all_rows[0]["esm_if_perplexity"] == "3.25"
    assert candidate_rows[0]["effective_backend"] == "opendde"
    assert manifest.effective_model_sha256[context.plan.jobs[0].job_id] == file_sha256(
        model
    )


def test_standalone_cluster_rejects_changed_effective_model_without_manifest_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )
    model = tmp_path / "effective.cif"
    model.write_text("data_before\n")
    manifest = workflow._existing_or_new_manifest(
        context,
        workflow._expected_feature_fingerprint(
            context.config, context.plan.target_sequence
        ),
    )
    workflow._persist_clustering_inputs(
        context,
        [
            {
                "job_name": context.plan.jobs[0].job_id,
                "candidate_pool": True,
                "effective_backend": "opendde",
                "effective_status": "success",
                "effective_best_model_path": str(model),
            }
        ],
        manifest,
    )
    model.write_text("data_after!\n")
    original_manifest = context.manifest_path.read_bytes()

    with pytest.raises(PipelineExecutionError, match="effective model"):
        run_clustering_only(context)

    assert context.manifest_path.read_bytes() == original_manifest


def test_pipeline_resume_preserves_existing_manifest_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )
    payload = json.loads(context.manifest_path.read_text())
    payload["artifact_sha256"]["resume_sentinel"] = "abc123"
    atomic_write_json(context.manifest_path, payload)

    workflow.run_pipeline(context)

    resumed = json.loads(context.manifest_path.read_text())
    assert resumed["artifact_sha256"]["resume_sentinel"] == "abc123"


def test_prepared_feature_content_is_bound_to_run_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )
    root = tmp_path / "prepared-features"
    templates = root / "templates"
    templates.mkdir(parents=True)
    pairing = root / "pairing.a3m"
    non_pairing = root / "non_pairing.a3m"
    hmmsearch = root / "hmmsearch.a3m"
    template_json = root / "af3_templates.json"
    for path, content in (
        (pairing, ">query\nLMNP\n"),
        (non_pairing, ">query\nLMNP\n"),
        (hmmsearch, ">query\nLMNP\n"),
        (template_json, '{"templates": []}\n'),
        (templates / "1abc.cif", "data_template\n"),
    ):
        path.write_text(content)
    bundle = FeatureBundle(
        sequence_sha256="sequence",
        cache_dir=root,
        pairing_a3m=pairing,
        non_pairing_a3m=non_pairing,
        hmmsearch_a3m=hmmsearch,
        fingerprint="feature-fingerprint",
        af3_templates_json=template_json,
        template_mmcif_dir=templates,
        source_mmcif_dir=tmp_path / "source-mmcif",
    )
    manifest = workflow._existing_or_new_manifest(
        context,
        workflow._expected_feature_fingerprint(
            context.config, context.plan.target_sequence
        ),
    )

    first = workflow._bind_feature_content(manifest, bundle)
    non_pairing.write_text(">query\nLNMP\n")
    second = workflow._bind_feature_content(manifest, bundle)

    assert first != second
    assert manifest.feature_content_sha256 == second
    assert manifest.artifact_sha256["target_features"] == second


def test_standalone_cluster_rejects_wrong_job_membership_without_mutation(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(workflow, "resolve_docker_image_id", _stable_image_id)
    context = create_run_context(
        _run_config(tmp_path),
        overrides=_run_overrides(tmp_path),
        dry_run=True,
    )
    model = tmp_path / "effective.cif"
    model.write_text("data_effective\n")
    manifest = workflow._existing_or_new_manifest(
        context,
        workflow._expected_feature_fingerprint(
            context.config, context.plan.target_sequence
        ),
    )
    workflow._persist_clustering_inputs(
        context,
        [
            {
                "job_name": context.plan.jobs[0].job_id,
                "candidate_pool": True,
                "effective_backend": "opendde",
                "effective_status": "success",
                "effective_best_model_path": str(model),
            }
        ],
        manifest,
    )
    all_path, _candidate_path = workflow._clustering_input_paths(context)
    with all_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["job_name"] = "not-in-the-immutable-plan"
    atomic_write_csv(all_path, rows, fieldnames=list(rows[0]))
    manifest.artifact_sha256["clustering_input"] = file_sha256(all_path) or ""
    manifest.write(context.manifest_path)
    original_manifest = context.manifest_path.read_bytes()

    with pytest.raises(PipelineExecutionError, match="job membership"):
        run_clustering_only(context)

    assert context.manifest_path.read_bytes() == original_manifest


def test_cluster_cli_requests_read_only_existing_run_context(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    config = AerithConfig()
    context = SimpleNamespace(
        config=config,
        results_dir=tmp_path / "results" / "run",
    )

    def fake_create_run_context(*_args, **kwargs):
        captured.update(kwargs)
        return context

    monkeypatch.setattr(workflow, "create_run_context", fake_create_run_context)
    monkeypatch.setattr(workflow, "run_clustering_only", lambda _context: False)

    result = CliRunner().invoke(app, ["cluster"])

    assert result.exit_code == 0, result.output
    assert captured["initialize_run"] is False


def test_cluster_cli_failure_does_not_claim_fake_singletons(
    tmp_path: Path, monkeypatch
) -> None:
    context = SimpleNamespace(
        config=AerithConfig(),
        results_dir=tmp_path / "results" / "run",
    )
    monkeypatch.setattr(
        workflow,
        "create_run_context",
        lambda *_args, **_kwargs: context,
    )
    monkeypatch.setattr(workflow, "run_clustering_only", lambda _context: True)

    result = CliRunner().invoke(app, ["cluster"])

    assert result.exit_code == 1
    assert "partial clustering audit outputs" in result.output
    assert "singleton" not in result.output
