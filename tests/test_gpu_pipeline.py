from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from af3_binder_filter.backends import UnifiedPrediction, build_backend_command
from af3_binder_filter.config import AerithConfig
from af3_binder_filter.esm_tools import load_cached_esm_rows
from af3_binder_filter.derived_structures import file_sha256
from af3_binder_filter.gpu import GPUInfo
from af3_binder_filter.jobs import JobPlan, JobSpec
from af3_binder_filter.workflow import (
    GpuJobShard,
    RunContext,
    _runtime_gpus,
    _run_sharded_commands,
    plan_gpu_job_shards,
)


def _job(index: int) -> JobSpec:
    return JobSpec(
        job_id=f"job_{index}",
        sample_no=str(index),
        run_name=f"run_{index}",
        target_sequence="LMNP",
        binder_sequence="ACDE",
        target_chain="A",
        binder_chain="B",
        source_row_number=index + 2,
        seed=42,
        backend="alphafold3",
        model="alphafold3",
    )


def _gpu(index: int) -> GPUInfo:
    return GPUInfo(index, "RTX 3090", 0, 24576)


def _context(tmp_path: Path, jobs: tuple[JobSpec, ...]) -> RunContext:
    config = AerithConfig()
    config.project.results_dir = str(tmp_path)
    plan = JobPlan(jobs, "LMNP", tmp_path / "input.csv", len(jobs))
    return RunContext(
        config=config,
        resolved_config=None,
        plan=plan,
        fingerprint="fingerprint",
        run_id="run-test",
        results_dir=tmp_path,
        manifest_path=tmp_path / "manifest.json",
    )


def test_gpu_job_shards_use_deterministic_weighted_lpt() -> None:
    jobs = tuple(
        replace(_job(index), binder_sequence="A" * length)
        for index, length in enumerate((40, 30, 20, 10, 8, 6, 4))
    )

    shards = plan_gpu_job_shards(jobs, [_gpu(0), _gpu(2), _gpu(3)])

    assignments = [[job.job_id for job in shard.jobs] for shard in shards]
    repeated = plan_gpu_job_shards(jobs, [_gpu(0), _gpu(2), _gpu(3)])
    assert assignments == [[job.job_id for job in shard.jobs] for shard in repeated]
    assert [shard.gpu.index for shard in shards] == [0, 2, 3]
    lpt_max = max(shard.estimated_cost for shard in shards)
    round_robin = [jobs[index::3] for index in range(3)]
    round_robin_max = max(
        sum((len(job.target_sequence) + len(job.binder_sequence)) ** 2 for job in bucket)
        for bucket in round_robin
    )
    assert lpt_max <= round_robin_max


def test_runtime_gpu_selection_filters_busy_and_disallowed_devices(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = _context(tmp_path, (_job(0), _job(1)))
    context.config.runtime.gpu_ids = [0, 2, 3]
    context.config.runtime.gpu_busy_threshold_mib = 100
    monkeypatch.setattr(
        "af3_binder_filter.workflow.query_gpus",
        lambda: [
            GPUInfo(0, "allowed", 2, 24576),
            GPUInfo(1, "not-allowed", 2, 24576),
            GPUInfo(2, "threshold", 100, 24576),
            GPUInfo(3, "busy", 101, 24576),
        ],
    )

    selected = _runtime_gpus(
        context,
        job_count=4,
        stage_name="primary_prediction",
    )

    assert [gpu.index for gpu in selected] == [0, 2]


def test_sharded_runner_preserves_negative_return_codes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jobs = (_job(0), _job(1))
    context = _context(tmp_path, jobs)
    shards = [
        GpuJobShard(_gpu(0), (jobs[0],)),
        GpuJobShard(_gpu(2), (jobs[1],)),
    ]

    def fake_run(_context, command, *, name):
        assert name in {"prediction.gpu_0", "prediction.gpu_2"}
        return int(command[-1])

    monkeypatch.setattr(
        "af3_binder_filter.workflow._run_prediction_command",
        fake_run,
    )

    return_codes, errors = _run_sharded_commands(
        context,
        "prediction",
        [
            (shards[0], ["fake", "0"]),
            (shards[1], ["fake", "-9"]),
        ],
    )

    assert return_codes == {0: 0, 2: -9}
    assert errors == []
    command_log = (
        tmp_path
        / "stages"
        / "03_primary_prediction"
        / "logs"
        / "prediction.command.txt"
    ).read_text()
    assert "# gpu=0 jobs=job_0" in command_log
    assert "# gpu=2 jobs=job_1" in command_log


def test_backend_container_has_unique_name_and_one_visible_gpu(
    tmp_path: Path,
) -> None:
    config = AerithConfig()
    config.features.database_dir = str(tmp_path / "database")
    config.backend.model_dir = str(tmp_path / "models")
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    output_dir.mkdir()

    command = build_backend_command(
        config,
        input_dir=input_dir,
        output_dir=output_dir,
        gpu_index=2,
        container_name="aerith-run-primary-gpu2",
    )

    assert command[command.index("--name") + 1] == "aerith-run-primary-gpu2"
    assert command[command.index("--gpus") + 1] == "device=2"
    assert "--gpu_device=0" in command


def test_esm_cache_requires_complete_parseable_outputs(tmp_path: Path) -> None:
    job = _job(0)
    prediction_path = tmp_path / "complex.cif"
    prediction_path.write_text("cif")
    model_path = tmp_path / "binder.pdb"
    model_path.write_text("pdb")
    prediction = UnifiedPrediction(
        job.job_id,
        "alphafold3",
        "success",
        best_model_path=prediction_path,
    )
    score_path = tmp_path / "esm_scores.csv"
    header = (
        "job_name,esmfold_status,esmfold_model_path,esmfold_plddt,"
        "esm_if_status,esm_if_log_likelihood,esm_if_log_likelihood_with_coord,esm_if_perplexity,"
        "esm_effective_backend,esm_effective_derived_structure_id,"
        "esm_effective_source_model_sha256\n"
    )
    row = (
        f"{job.job_id},success,{model_path},80.0,success,"
        f"-1.0,-1.0,3.0,alphafold3,,{file_sha256(prediction_path)}\n"
    )
    score_path.write_text(header + row)

    def load():
        return load_cached_esm_rows(
            score_path,
            (job,),
            (prediction,),
            require_esmfold=True,
            require_inverse_folding=True,
        )

    assert load() is not None

    score_path.write_text(header + row + row)
    assert load() is None

    score_path.write_text(
        header + row.replace(",-1.0,-1.0,3.0,", ",-1.0,-1.0,nan,")
    )
    assert load() is None

    score_path.write_text(header + row)
    model_path.unlink()
    assert load() is None
