from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path

import pytest

from af3_binder_filter.backends import (
    AlphaFold3OutputAdapter,
    RankedJsonOutputAdapter,
    build_backend_command,
    make_protenix_style_input,
    prepare_runtime_build_contexts,
    write_backend_inputs,
)
from af3_binder_filter.config import AerithConfig, FeatureSettings
from af3_binder_filter.features import (
    FeatureError,
    FeatureBundle,
    build_feature_builder_command,
    prepare_target_features,
    write_query_only_msa,
)
from af3_binder_filter.jobs import JobSpec
from af3_binder_filter.target_data import TargetDataError, extract_target_features


def _job() -> JobSpec:
    return JobSpec(
        "job",
        "1",
        "run",
        "LMNP",
        "ACDE",
        "A",
        "B",
        2,
        42,
        "protenix",
        "protenix-v2",
    )


def _features(tmp_path: Path) -> FeatureBundle:
    root = tmp_path / "features"
    root.mkdir()
    paths = [root / name for name in ("pairing.a3m", "non_pairing.a3m", "hmmsearch.a3m")]
    for path in paths:
        path.write_text(">query\nLMNP\n")
    templates = root / "templates"
    templates.mkdir()
    template_json = root / "af3_templates.json"
    template_json.write_text(json.dumps({"version": 1, "templates": []}))
    return FeatureBundle(
        "digest",
        root,
        *paths,
        "fingerprint",
        template_json,
        templates,
    )


def test_protenix_contract_uses_local_target_and_query_only_binder(tmp_path: Path) -> None:
    features = _features(tmp_path)
    binder_msa = write_query_only_msa("ACDE", tmp_path / "binder" / "non_pairing.a3m")
    binder_templates = write_query_only_msa(
        "ACDE", tmp_path / "binder" / "hmmsearch.a3m"
    )

    payload = make_protenix_style_input(
        _job(),
        target_features=features,
        binder_msa_path=binder_msa,
        binder_templates_path=binder_templates,
    )

    target = payload["sequences"][0]["proteinChain"]
    binder = payload["sequences"][1]["proteinChain"]
    assert Path(target["pairedMsaPath"]).exists()
    assert Path(target["unpairedMsaPath"]).exists()
    assert Path(target["templatesPath"]).exists()
    assert Path(binder["unpairedMsaPath"]).read_text() == ">query\nACDE\n"
    assert "pairedMsaPath" not in binder
    assert Path(binder["templatesPath"]).read_text() == ">query\nACDE\n"


def test_af3_contract_uses_converted_gpu_features_without_data_pipeline(
    tmp_path: Path,
) -> None:
    features = _features(tmp_path)
    template = features.template_mmcif_dir / "template.cif"
    template.write_text("data_template\n")
    features.af3_templates_json.write_text(
        json.dumps(
            {
                "version": 1,
                "templates": [
                    {
                        "mmcifFile": template.name,
                        "queryIndices": [0, 1],
                        "templateIndices": [2, 3],
                    }
                ],
            }
        )
    )
    config = AerithConfig()
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    paths = write_backend_inputs(
        [_job()],
        config,
        input_dir=input_dir,
        target_features=features,
        force=True,
    )
    payload = json.loads(paths[0].read_text())
    target = payload["sequences"][0]["protein"]
    binder = payload["sequences"][1]["protein"]

    assert target["unpairedMsaPath"] == str(features.non_pairing_a3m.resolve())
    assert target["pairedMsaPath"] == str(features.pairing_a3m.resolve())
    assert target["templates"][0]["mmcifPath"] == str(template.resolve())
    assert binder["unpairedMsa"] == ">query\nACDE\n"
    assert binder["pairedMsa"] == ""
    assert binder["templates"] == []

    command = build_backend_command(
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        gpu_index=0,
        feature_dir=features.cache_dir,
    )
    assert "--norun_data_pipeline" in command


def test_feature_builder_uses_in_image_gpu_mmseqs_and_exposes_search_limits(
    tmp_path: Path,
) -> None:
    database = tmp_path / "db"
    database.mkdir()
    settings = FeatureSettings(
        database_dir=str(database),
        threads=4,
        split_memory_limit="8G",
        iterations=2,
    )

    command, query = build_feature_builder_command(
        settings,
        target_sequence="ACDE",
        output_dir=tmp_path / "features",
    )

    joined = " ".join(command)
    assert settings.image == "aerith/fold-runtime:local"
    assert "aerith/fold-runtime:local prepare-features" in joined
    assert "--network none" in joined
    assert "--gpus device=0" in joined
    assert f"{database.resolve()}:/db:ro" in joined
    assert ":/opt/aerith/mmseqs:ro" not in joined
    assert "--mmseqs-binary mmseqs" in joined
    assert "--threads 4" in joined
    assert "--split-memory-limit 8G" in joined
    assert "--iterations 2" in joined
    assert "--use-gpu 1" in joined
    assert query.read_text() == ">query\nACDE\n"


def test_feature_builder_rejects_host_mmseqs_override(tmp_path: Path) -> None:
    database = tmp_path / "db"
    database.mkdir()
    mmseqs = tmp_path / "mmseqs"
    mmseqs.write_text("host binary")

    with pytest.raises(FeatureError, match="host overrides are disabled"):
        build_feature_builder_command(
            FeatureSettings(
                database_dir=str(database),
                mmseqs_binary=str(mmseqs),
            ),
            target_sequence="ACDE",
            output_dir=tmp_path / "features",
        )


def test_local_feature_search_enables_mmseqs_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = runpy.run_path(
        str(
            Path(__file__).parents[1]
            / "docker"
            / "feature-builder"
            / "build_local_features.py"
        )
    )
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module["subprocess"], "run", runner)
    module["_run_mmseqs_search"](
        wrapper="/opt/aerith/mmseqs_wrapper.py",
        query_db=tmp_path / "query",
        database=tmp_path / "database",
        output_root=tmp_path,
        label="primary",
        threads=4,
        iterations=1,
        split_memory_limit="8G",
        use_gpu=True,
        environment={},
    )

    assert calls[0][-2:] == ["--gpu", "1"]
    assert "--gpu" not in calls[1]


def test_feature_preparation_publishes_complete_cache_atomically(
    tmp_path: Path,
) -> None:
    settings = FeatureSettings(
        database_dir=str(tmp_path / "db"),
        cache_dir=str(tmp_path / "cache"),
    )
    (tmp_path / "db").mkdir()

    def runner(command, **_kwargs):
        output_mount = next(
            value for value in command if value.endswith(":/output")
        )
        output = Path(output_mount.removesuffix(":/output"))
        for name in ("pairing.a3m", "non_pairing.a3m", "hmmsearch.a3m"):
            (output / name).write_text(">query\nACDE\n")
        (output / "templates").mkdir()
        (output / "af3_templates.json").write_text(
            json.dumps({"version": 1, "templates": []})
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    preparation = prepare_target_features(settings, "ACDE", runner=runner)

    assert preparation.bundle is not None
    assert (preparation.bundle.cache_dir / "manifest.json").is_file()
    assert not list(preparation.bundle.cache_dir.glob(".feature-build-*"))


def test_feature_preparation_failure_does_not_publish_manifest(
    tmp_path: Path,
) -> None:
    settings = FeatureSettings(
        database_dir=str(tmp_path / "db"),
        cache_dir=str(tmp_path / "cache"),
    )
    (tmp_path / "db").mkdir()

    def runner(command, **_kwargs):
        return subprocess.CompletedProcess(command, -9, "", "killed")

    with pytest.raises(FeatureError, match="return code -9"):
        prepare_target_features(settings, "ACDE", runner=runner)

    cache_roots = list((tmp_path / "cache").glob("*"))
    assert len(cache_roots) == 1
    assert not (cache_roots[0] / "manifest.json").exists()
    assert not list(cache_roots[0].glob(".feature-build-*"))


@pytest.mark.parametrize("backend", ["protenix", "opendde"])
def test_ranked_backend_fixture_selects_highest_score(tmp_path: Path, backend: str) -> None:
    job = _job()
    root = tmp_path / "output" / job.job_id / "seed_42" / "predictions"
    root.mkdir(parents=True)
    for sample, score in ((0, 0.2), (1, 0.9)):
        (root / f"job_summary_confidence_sample_{sample}.json").write_text(
            json.dumps({"ranking_score": score, "iptm": score, "ptm": 0.7, "plddt": 80})
        )
        (root / f"job_sample_{sample}.cif").write_text("data_job\n")

    prediction = RankedJsonOutputAdapter(backend).parse(job, tmp_path / "output")

    assert prediction.status == "success"
    assert prediction.ranking_score == 0.9
    assert prediction.best_model_path.name == "job_sample_1.cif"
    assert prediction.iptm == 0.9


def test_ranked_backend_adapter_does_not_cross_assign_batched_jobs(
    tmp_path: Path,
) -> None:
    job = _job()
    root = tmp_path / "output"
    for job_name, score in (("job", 0.3), ("other_job", 0.99)):
        job_root = root / "batch" / job_name
        job_root.mkdir(parents=True)
        (job_root / f"{job_name}_summary_confidence_sample_0.json").write_text(
            json.dumps({"ranking_score": score, "iptm": score})
        )
        (job_root / f"{job_name}_sample_0.cif").write_text("data_job\n")

    prediction = RankedJsonOutputAdapter("protenix").parse(job, root)

    assert prediction.ranking_score == 0.3
    assert prediction.best_model_path is not None
    assert "other_job" not in prediction.best_model_path.as_posix()


def test_ranked_backend_adapter_reports_missing_instead_of_borrowing_other_job(
    tmp_path: Path,
) -> None:
    job = _job()
    root = tmp_path / "output"
    other_root = root / "batch" / "other_job"
    other_root.mkdir(parents=True)
    (other_root / "other_job_summary_confidence_sample_0.json").write_text(
        json.dumps({"ranking_score": 0.99, "iptm": 0.99})
    )
    (other_root / "other_job_sample_0.cif").write_text("data_other_job\n")

    prediction = RankedJsonOutputAdapter("opendde").parse(job, root)

    assert prediction.status == "missing"
    assert prediction.best_model_path is None
    assert prediction.error is not None
    assert "job 'job'" in prediction.error


def test_af3_fixture_uses_ranking_csv_best_sample(tmp_path: Path) -> None:
    job = _job()
    output = tmp_path / "output"
    job_dir = output / job.job_id
    sample_dir = job_dir / "seed-7_sample-1"
    sample_dir.mkdir(parents=True)
    (job_dir / "job_ranking_scores.csv").write_text(
        "seed,sample,ranking_score\n42,0,0.2\n7,1,0.9\n"
    )
    basename = "job_seed-7_sample-1"
    (sample_dir / f"{basename}_summary_confidences.json").write_text(
        json.dumps({"ranking_score": 0.9, "iptm": 0.8, "ptm": 0.7})
    )
    (sample_dir / f"{basename}_confidences.json").write_text(
        json.dumps({"plddt": [80, 100], "pae": [[0, 1], [1, 0]]})
    )
    (sample_dir / f"{basename}_model.cif").write_text("data_job\n")

    prediction = AlphaFold3OutputAdapter().parse(job, output)

    assert prediction.status == "success"
    assert prediction.best_model_path.name == f"{basename}_model.cif"
    assert prediction.plddt == 90


def test_secondary_dry_run_contract_is_offline_and_uses_minimal_mounts(
    tmp_path: Path,
) -> None:
    config = AerithConfig()
    config.backend.name = "opendde"
    config.backend.image = "opendde:test"
    config.features.database_dir = str(tmp_path / "db")
    (tmp_path / "db").mkdir()
    command = build_backend_command(
        config,
        input_dir=tmp_path / "inputs",
        output_dir=tmp_path / "outputs",
        gpu_index=3,
        feature_dir=tmp_path / "features",
    )
    joined = " ".join(command)
    assert "--network none" in joined
    assert "--gpus device=3" in joined
    assert f"{tmp_path / 'db'}:{tmp_path / 'db'}:ro" not in joined
    assert "--use_msa true" in joined
    assert "--use_template false" in joined
    assert "--dtype bf16" in joined
    assert "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in joined
    assert "/opendde_data/checkpoint/opendde.pt" in joined
    assert "opendde_abag.pt" not in joined


def test_protenix_common_assets_use_non_overlapping_file_mounts(
    tmp_path: Path,
) -> None:
    config = AerithConfig()
    config.backend.name = "protenix"
    config.backend.image = "protenix:test"
    checkpoint_dir = tmp_path / "checkpoint"
    common_dir = tmp_path / "common"
    metadata_dir = tmp_path / "metadata"
    checkpoint_dir.mkdir()
    common_dir.mkdir()
    metadata_dir.mkdir()
    for filename in (
        "components.cif",
        "components.cif.rdkit_mol.pkl",
        "clusters-by-entity-40.txt",
        "obsolete_release_date.csv",
    ):
        (common_dir / filename).write_text("fixture\n")
    for filename in ("release_date_cache.json", "obsolete_to_successor.json"):
        (metadata_dir / filename).write_text("{}\n")
    config.backend.checkpoint_dir = str(checkpoint_dir)
    config.backend.common_dir = str(common_dir)
    config.backend.metadata_dir = str(metadata_dir)

    command = build_backend_command(
        config,
        input_dir=tmp_path / "inputs",
        output_dir=tmp_path / "outputs",
        gpu_index=1,
    )
    volumes = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--volume"
    ]

    assert f"{common_dir.resolve()}:/protenix_data/common:ro" not in volumes
    assert "--load_checkpoint_dir" not in command
    for filename in (
        "components.cif",
        "components.cif.rdkit_mol.pkl",
        "clusters-by-entity-40.txt",
        "obsolete_release_date.csv",
    ):
        assert (
            f"{(common_dir / filename).resolve()}:"
            f"/protenix_data/common/{filename}:ro"
        ) in volumes
    for filename in ("release_date_cache.json", "obsolete_to_successor.json"):
        assert (
            f"{(metadata_dir / filename).resolve()}:"
            f"/protenix_data/common/{filename}:ro"
        ) in volumes


def test_target_feature_sequence_mismatch_is_rejected(tmp_path: Path) -> None:
    data = {
        "name": "target",
        "sequences": [{"protein": {"id": "A", "sequence": "ACDE", "unpairedMsa": ">q\nACDE"}}],
    }
    path = tmp_path / "data.json"
    path.write_text(json.dumps(data))

    with pytest.raises(TargetDataError, match="sequence mismatch"):
        extract_target_features(
            path,
            tmp_path / "out",
            expected_sequence="LMNP",
        )


def test_runtime_staging_keeps_af3_common_but_excludes_opendde_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = {}
    for name in ("af3", "protenix", "opendde", "esm"):
        source = tmp_path / name
        source.mkdir()
        (source / "package.py").write_text("VALUE = 1\n")
        sources[name] = source
    af3_common = sources["af3"] / "src" / "alphafold3" / "common"
    af3_common.mkdir(parents=True)
    (af3_common / "resources.py").write_text("RESOURCE = True\n")
    opendde_common = sources["opendde"] / "common"
    opendde_common.mkdir()
    (opendde_common / "components.cif").write_text("mounted at runtime\n")
    config = AerithConfig()
    config.project.work_dir = str(tmp_path / "work")
    config.runtime.af3_source_dir = str(sources["af3"])
    config.runtime.protenix_source_dir = str(sources["protenix"])
    config.runtime.opendde_source_dir = str(sources["opendde"])
    config.runtime.esm_source_dir = str(sources["esm"])
    config.runtime.opendde_source_commit = "deadbeef"
    config.runtime.esm_source_commit = "deadbeef"

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="deadbeef\n", stderr=""
        ),
    )

    staged = prepare_runtime_build_contexts(config)

    assert (
        staged / "af3-src" / "src" / "alphafold3" / "common" / "resources.py"
    ).is_file()
    assert not (staged / "opendde-src" / "common").exists()
    assert (staged / "opendde-src" / "package.py").is_file()
    assert not (staged / "mmseqs-src").exists()
