from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from af3_binder_filter.backends import UnifiedPrediction, build_backend_command
from af3_binder_filter.config import AerithConfig
from af3_binder_filter.derived_structures import file_sha256
from af3_binder_filter.esm_tools import load_cached_esm_rows
from af3_binder_filter.execution import CommandOutcome, LocalCommandExecutor
from af3_binder_filter.gpu import GPUInfo
from af3_binder_filter.jobs import JobPlan, JobSpec
from af3_binder_filter.manifest import RunManifest
from af3_binder_filter.workflow import (
    GpuJobShard,
    RunContext,
    _runtime_gpus,
    _run_sharded_commands,
    clustering_stage,
    esm_stage,
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
    config.project.work_dir = str(tmp_path / "work")
    config.project.output_dir = str(tmp_path / "outputs")
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


def _manifest(context: RunContext) -> RunManifest:
    return RunManifest(
        run_id=context.run_id,
        fingerprint=context.fingerprint,
        backend=context.config.backend.name,
        model=context.config.backend.model,
        source_csv=str(context.plan.source_csv),
        target_sequence_sha256="target",
        job_fingerprints={job.job_id: "fingerprint" for job in context.plan.jobs},
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
        "af3_binder_filter.orchestration.context.query_gpus",
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
) -> None:
    jobs = (_job(0), _job(1))
    context = _context(tmp_path, jobs)
    shards = [
        GpuJobShard(_gpu(0), (jobs[0],)),
        GpuJobShard(_gpu(2), (jobs[1],)),
    ]

    class FakeExecutor:
        def run(self, command, *, dry_run=False):
            assert command.name in {"prediction.gpu_0", "prediction.gpu_2"}
            return CommandOutcome(
                command=command,
                returncode=int(command.argv[-1]),
                duration_seconds=0.0,
                dry_run=dry_run,
            )

    return_codes, errors = _run_sharded_commands(
        context,
        "prediction",
        [
            (shards[0], ["fake", "0"]),
            (shards[1], ["fake", "-9"]),
        ],
        executor=FakeExecutor(),  # type: ignore[arg-type]
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


def test_sharded_dry_run_writes_aggregate_and_per_gpu_command_records(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, (_job(0),))
    context.config.runtime.dry_run = True
    shard = GpuJobShard(_gpu(0), context.plan.jobs)

    return_codes, errors = _run_sharded_commands(
        context,
        "prediction",
        [(shard, ["docker", "run", "image name", "argument with spaces"])],
    )

    assert return_codes == {0: 0}
    assert errors == []
    log_dir = context.layout.stage("primary_prediction").logs
    aggregate = (log_dir / "prediction.command.txt").read_text()
    per_gpu = (log_dir / "prediction.gpu_0.command.txt").read_text()
    assert "'image name'" in aggregate
    assert "'argument with spaces'" in per_gpu
    assert not (log_dir / "prediction.gpu_0.stdout.log").exists()
    assert not (log_dir / "prediction.gpu_0.stderr.log").exists()


def test_sharded_timeout_preserves_real_signal_and_reaps_process(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path, (_job(0),))
    shard = GpuJobShard(_gpu(0), context.plan.jobs)
    executor = LocalCommandExecutor(termination_grace_seconds=0.05)

    return_codes, errors = _run_sharded_commands(
        context,
        "prediction",
        [(shard, [sys.executable, "-c", "import time; time.sleep(30)"])],
        timeout_seconds=0.05,
        executor=executor,
    )

    assert return_codes[0] is not None and return_codes[0] < 0
    assert len(errors) == 1 and "exceeded timeout" in errors[0]
    assert executor.active_process_count == 0


def test_sharded_keyboard_interrupt_cancels_every_active_process(
    tmp_path: Path,
    monkeypatch,
) -> None:
    jobs = (_job(0), _job(1))
    context = _context(tmp_path, jobs)
    shards = (
        GpuJobShard(_gpu(0), (jobs[0],)),
        GpuJobShard(_gpu(1), (jobs[1],)),
    )
    executor = LocalCommandExecutor(termination_grace_seconds=0.1)

    def interrupt_after_start(*_args, **_kwargs):
        deadline = time.monotonic() + 5.0
        while executor.active_process_count < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert executor.active_process_count >= 1
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "af3_binder_filter.orchestration.command_runtime.wait",
        interrupt_after_start,
    )
    with pytest.raises(KeyboardInterrupt):
        _run_sharded_commands(
            context,
            "prediction",
            [
                (
                    shard,
                    [sys.executable, "-c", "import time; time.sleep(30)"],
                )
                for shard in shards
            ],
            executor=executor,
        )

    assert executor.active_process_count == 0
    assert executor.cancellation_requested
    log_dir = context.layout.stage("primary_prediction").logs
    for gpu_index in (0, 1):
        stdout = log_dir / f"prediction.gpu_{gpu_index}.stdout.log"
        stderr = log_dir / f"prediction.gpu_{gpu_index}.stderr.log"
        if stdout.exists():
            stdout.rename(log_dir / f"closed.{stdout.name}")
        if stderr.exists():
            stderr.rename(log_dir / f"closed.{stderr.name}")


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


def test_clustering_stage_forwards_effective_rows_to_foldseek_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = _job(0)
    context = _context(tmp_path, (job,))
    model = tmp_path / "effective.cif"
    model.write_text("model")
    prediction = UnifiedPrediction(
        job.job_id,
        "opendde",
        "success",
        best_model_path=model,
    )
    effective_rows = ({"job_name": job.job_id, "effective_backend": "opendde"},)

    class WiringObserved(RuntimeError):
        pass

    def capture(_jobs, _paths, *, work_dir, rows):
        assert work_dir.is_dir()
        assert rows is effective_rows
        raise WiringObserved

    monkeypatch.setattr(
        "af3_binder_filter.orchestration.clustering_stage.prepare_foldseek_inputs",
        capture,
    )
    with pytest.raises(WiringObserved):
        clustering_stage(
            context,
            (prediction,),
            effective_rows,
        )


def test_esm_stage_forwards_effective_rows_to_all_cache_and_input_helpers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = _job(0)
    context = _context(tmp_path, (job,))
    context.config.scoring.esm.enabled = True
    context.config.scoring.esm.esmfold = False
    context.config.scoring.esm.inverse_folding = False
    model = tmp_path / "effective.cif"
    model.write_text("model")
    prediction = UnifiedPrediction(
        job.job_id,
        "opendde",
        "success",
        best_model_path=model,
    )
    effective_rows = ({"job_name": job.job_id, "effective_backend": "opendde"},)
    seen: dict[str, object] = {}

    def cache_loader(*_args, **kwargs):
        seen["cache_rows"] = kwargs["structure_rows"]
        seen["cache_primary"] = kwargs["primary_predictions"]
        seen["cache_secondary"] = kwargs["secondary_predictions"]
        return None

    def input_writer(_jobs, _predictions, _directory, **kwargs):
        seen.setdefault("input_rows", []).append(kwargs["structure_rows"])
        return tmp_path / "binders.fasta", tmp_path / "esm_if_jobs.json"

    def collector(_jobs, _predictions, _directory, **kwargs):
        seen["collect_rows"] = kwargs["structure_rows"]
        return [{"job_name": job.job_id, "esm_status": "disabled"}]

    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage.load_cached_esm_rows",
        cache_loader,
    )
    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage.write_esm_inputs",
        input_writer,
    )
    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage.collect_esm_rows",
        collector,
    )

    rows, failed = esm_stage(
        context,
        (prediction,),
        _manifest(context),
        primary_predictions=(prediction,),
        secondary_predictions=(prediction,),
        structure_rows=effective_rows,
    )

    assert failed is False
    assert rows[0]["job_name"] == job.job_id
    assert seen["cache_rows"] is effective_rows
    assert seen["cache_primary"] == (prediction,)
    assert seen["cache_secondary"] == (prediction,)
    assert seen["input_rows"] == [effective_rows]
    assert seen["collect_rows"] is effective_rows


def test_esm_stage_forwards_configured_timeout_to_shard_executor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    job = _job(0)
    context = _context(tmp_path, (job,))
    context.config.scoring.esm.enabled = True
    context.config.scoring.esm.esmfold = True
    context.config.scoring.esm.inverse_folding = False
    context.config.scoring.esm.timeout_seconds = 321
    model = tmp_path / "effective-timeout.cif"
    model.write_text("model")
    prediction = UnifiedPrediction(
        job.job_id,
        "alphafold3",
        "success",
        best_model_path=model,
    )
    seen_timeouts: list[float | None] = []

    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage.load_cached_esm_rows",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage.write_esm_inputs",
        lambda *_args, **_kwargs: (
            tmp_path / "binders.fasta",
            tmp_path / "esm_if_jobs.json",
        ),
    )
    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage._runtime_gpus",
        lambda *_args, **_kwargs: [_gpu(0)],
    )

    def run_shards(_context, _name, _commands, **kwargs):
        seen_timeouts.append(kwargs.get("timeout_seconds"))
        return {0: 0}, []

    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage._run_sharded_commands",
        run_shards,
    )
    monkeypatch.setattr(
        "af3_binder_filter.orchestration.esm_stage.collect_esm_rows",
        lambda *_args, **_kwargs: [
            {
                "job_name": job.job_id,
                "esmfold_status": "success",
                "esm_if_status": "not_available",
            }
        ],
    )

    _rows, failed = esm_stage(
        context,
        (prediction,),
        _manifest(context),
    )

    assert failed is False
    assert seen_timeouts == [321]


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
