from __future__ import annotations

from io import StringIO
from pathlib import Path

from rich.console import Console

from af3_binder_filter.config_tools import EnvironmentDetection, write_initial_config
from af3_binder_filter.orchestration.command_runtime import stable_completion_probe
from af3_binder_filter.orchestration.context import (
    create_run_context,
    pipeline_stage_specs,
)
from af3_binder_filter.orchestration.pipeline import run_pipeline
from af3_binder_filter.progress import (
    NullProgressReporter,
    PipelineRunInfo,
    RichProgressReporter,
    StageSpec,
)


class RecordingReporter(NullProgressReporter):
    def __init__(self) -> None:
        self.stage_starts: list[str] = []
        self.stage_finishes: list[tuple[str, str]] = []
        self.cache_events: list[tuple[str, int, int, int, bool]] = []
        self.pipeline_status: str | None = None

    def stage_started(self, stage: str, *, log_dir: Path) -> None:
        self.stage_starts.append(stage)

    def stage_finished(
        self,
        stage: str,
        *,
        status: str,
        detail: str = "",
    ) -> None:
        self.stage_finishes.append((stage, status))

    def cache_status(
        self,
        stage: str,
        *,
        hits: int,
        misses: int,
        total: int,
        force: bool = False,
    ) -> None:
        self.cache_events.append((stage, hits, misses, total, force))

    def pipeline_finished(
        self,
        *,
        status: str,
        detail: str = "",
    ) -> None:
        self.pipeline_status = status


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
        "sample_no,run_name,binder_sequence,target_seq\n1,run_1,ACDE,LMNP\n2,run_2,FGHI,LMNP\n"
    )
    return path


def test_stage_plan_is_dynamic() -> None:
    from af3_binder_filter.config import AerithConfig

    config = AerithConfig()
    config.secondary_backend.enabled = False
    config.scoring.esm.enabled = False
    assert [stage.key for stage in pipeline_stage_specs(config)] == [
        "features",
        "primary_prediction",
        "primary_interface",
        "consensus",
        "clustering",
    ]

    config.secondary_backend.enabled = True
    config.secondary_backend.name = "opendde"
    config.scoring.esm.enabled = True
    assert [stage.key for stage in pipeline_stage_specs(config)] == [
        "features",
        "primary_prediction",
        "primary_interface",
        "secondary_features",
        "secondary_prediction",
        "secondary_interface",
        "consensus",
        "esm",
        "clustering",
    ]


def test_stable_completion_probe_rejects_baseline_and_partial_changes(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "job.done"

    def signature(_key: str) -> tuple[tuple[str, int, int], ...]:
        if not artifact.is_file():
            return ()
        stat = artifact.stat()
        return ((str(artifact), stat.st_size, stat.st_mtime_ns),)

    probe = stable_completion_probe(("job",), signature)
    assert probe() == 0
    artifact.write_text("partial")
    assert probe() == 0
    assert probe() == 1

    baseline_probe = stable_completion_probe(("job",), signature)
    assert baseline_probe() == 0
    assert baseline_probe() == 0
    artifact.write_text("refreshed output")
    assert baseline_probe() == 0
    assert baseline_probe() == 1


def test_plain_renderer_uses_cache_hit_and_cache_missing_terms(
    tmp_path: Path,
) -> None:
    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None)
    reporter = RichProgressReporter(console)
    reporter.pipeline_started(
        PipelineRunInfo(
            run_id="run-test",
            job_count=3,
            primary_backend="alphafold3",
            secondary_backend="opendde",
            gpu_ids=(0, 1),
            results_dir=tmp_path / "results",
            output_dir=tmp_path / "outputs",
            logs_dir=tmp_path / "results" / "stages",
        ),
        (StageSpec("features", "MSA/template searching"),),
    )
    reporter.stage_started("features", log_dir=tmp_path / "logs")
    reporter.cache_status("features", hits=1, misses=2, total=3)
    reporter.task_started("features", "Target MSA/templates", total=3, completed=1)
    reporter.task_finished(
        "features",
        "Target MSA/templates",
        completed=3,
        total=3,
        success=2,
        failed=1,
    )
    reporter.stage_finished("features", status="partial")
    reporter.pipeline_finished(status="partial")
    reporter.close()

    output = stream.getvalue()
    assert "cache hit: 1/3" in output
    assert "cache missing: 2/3" in output
    assert "reused" not in output
    assert "3/3" in output
    assert "[Stage 1/1] MSA/template searching" in output


def test_dry_run_reports_every_enabled_stage_and_cache_missing(
    tmp_path: Path,
) -> None:
    database = _database(tmp_path / "db")
    csv_path = _csv(tmp_path / "input.csv")
    config_path = write_initial_config(
        tmp_path / "config.yaml",
        EnvironmentDetection(database_dir=str(database), gpu_indexes=(0, 1)),
        csv_path=str(csv_path),
        secondary_backend="opendde",
    )
    context = create_run_context(
        config_path,
        dry_run=True,
        overrides=[
            f"project.work_dir={tmp_path / 'work'}",
            f"project.output_dir={tmp_path / 'outputs'}",
            f"project.results_dir={tmp_path / 'results'}",
        ],
    )
    reporter = RecordingReporter()

    assert run_pipeline(context, reporter=reporter) == []

    expected = [stage.key for stage in pipeline_stage_specs(context.config)]
    assert reporter.stage_starts == expected
    assert reporter.stage_finishes == [(stage, "dry_run") for stage in expected]
    assert reporter.cache_events == [("features", 0, 1, 1, False)]
    assert reporter.pipeline_status == "dry_run"
