from types import SimpleNamespace

import af3_binder_filter.workflow as workflow
from af3_binder_filter.af3_runner import combined_return_code
from af3_binder_filter.backends import UnifiedPrediction
from af3_binder_filter.config import AerithConfig, BackendSettings
from af3_binder_filter.jobs import JobSpec
from af3_binder_filter.manifest import write_job_manifest
from af3_binder_filter.workflow import (
    _backend_job_fingerprint,
    load_predictions_for_context,
    secondary_gate_job_ids,
)


def test_negative_subprocess_return_code_is_failure() -> None:
    assert combined_return_code([0, 0]) == 0
    assert combined_return_code([-9, 0]) == -9
    assert combined_return_code([0, 2]) == 2


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
