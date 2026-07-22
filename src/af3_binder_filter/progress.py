"""Structured pipeline progress events and terminal rendering.

The workflow emits progress through the small :class:`PipelineProgressReporter`
protocol instead of writing directly to stdout.  This keeps programmatic
workflow use quiet while allowing the Typer CLI to provide a Rich live view.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.rule import Rule


@dataclass(frozen=True, slots=True)
class StageSpec:
    """One enabled stage in display order."""

    key: str
    label: str


@dataclass(frozen=True, slots=True)
class PipelineRunInfo:
    """Small, presentation-safe summary shown before execution."""

    run_id: str
    job_count: int
    primary_backend: str
    secondary_backend: str
    gpu_ids: tuple[int, ...]
    results_dir: Path
    output_dir: Path
    logs_dir: Path


class PipelineProgressReporter(Protocol):
    """Events emitted by the workflow.

    All progress values are absolute, not deltas.  Implementations can
    therefore safely ignore duplicate updates from polling loops.
    """

    def pipeline_started(
        self,
        info: PipelineRunInfo,
        stages: Sequence[StageSpec],
    ) -> None: ...

    def stage_started(self, stage: str, *, log_dir: Path) -> None: ...

    def cache_status(
        self,
        stage: str,
        *,
        hits: int,
        misses: int,
        total: int,
        force: bool = False,
    ) -> None: ...

    def task_started(
        self,
        stage: str,
        task: str,
        *,
        total: int | None,
        completed: int = 0,
        detail: str = "",
    ) -> None: ...

    def task_progress(
        self,
        stage: str,
        task: str,
        *,
        completed: int,
        total: int | None = None,
        success: int | None = None,
        failed: int | None = None,
        skipped: int | None = None,
        detail: str = "",
    ) -> None: ...

    def task_finished(
        self,
        stage: str,
        task: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        success: int | None = None,
        failed: int | None = None,
        skipped: int | None = None,
        detail: str = "",
    ) -> None: ...

    def stage_finished(
        self,
        stage: str,
        *,
        status: str,
        detail: str = "",
    ) -> None: ...

    def message(self, text: str, *, level: str = "info") -> None: ...

    def pipeline_finished(
        self,
        *,
        status: str,
        detail: str = "",
    ) -> None: ...

    def close(self) -> None: ...


class NullProgressReporter:
    """No-op reporter used by the Python workflow API."""

    def pipeline_started(
        self,
        info: PipelineRunInfo,
        stages: Sequence[StageSpec],
    ) -> None:
        return None

    def stage_started(self, stage: str, *, log_dir: Path) -> None:
        return None

    def cache_status(
        self,
        stage: str,
        *,
        hits: int,
        misses: int,
        total: int,
        force: bool = False,
    ) -> None:
        return None

    def task_started(
        self,
        stage: str,
        task: str,
        *,
        total: int | None,
        completed: int = 0,
        detail: str = "",
    ) -> None:
        return None

    def task_progress(
        self,
        stage: str,
        task: str,
        *,
        completed: int,
        total: int | None = None,
        success: int | None = None,
        failed: int | None = None,
        skipped: int | None = None,
        detail: str = "",
    ) -> None:
        return None

    def task_finished(
        self,
        stage: str,
        task: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        success: int | None = None,
        failed: int | None = None,
        skipped: int | None = None,
        detail: str = "",
    ) -> None:
        return None

    def stage_finished(
        self,
        stage: str,
        *,
        status: str,
        detail: str = "",
    ) -> None:
        return None

    def message(self, text: str, *, level: str = "info") -> None:
        return None

    def pipeline_finished(
        self,
        *,
        status: str,
        detail: str = "",
    ) -> None:
        return None

    def close(self) -> None:
        return None


@dataclass(slots=True)
class _TaskState:
    rich_id: int | None
    total: int | None
    completed: int
    last_plain_bucket: int = -1
    last_plain_at: float = 0.0


class RichProgressReporter:
    """Rich live renderer with a plain-text fallback for redirected output."""

    def __init__(self, console: Console):
        self.console = console
        self._interactive = bool(console.is_terminal)
        self._stages: tuple[StageSpec, ...] = ()
        self._stage_index: dict[str, int] = {}
        self._stage_started_at: dict[str, float] = {}
        self._stage_logs: dict[str, Path] = {}
        self._active_stage: str | None = None
        self._progress: Progress | None = None
        self._tasks: dict[tuple[str, str], _TaskState] = {}
        self._pipeline_started_at: float | None = None
        self._closed = False

    def pipeline_started(
        self,
        info: PipelineRunInfo,
        stages: Sequence[StageSpec],
    ) -> None:
        self._pipeline_started_at = time.monotonic()
        self._stages = tuple(stages)
        self._stage_index = {
            stage.key: index for index, stage in enumerate(self._stages, start=1)
        }
        gpu_text = ",".join(str(index) for index in info.gpu_ids) or "auto"
        secondary = (
            info.secondary_backend
            if info.secondary_backend and info.secondary_backend != "none"
            else "disabled"
        )
        self.console.print(Rule("[bold cyan]Aerith pipeline[/bold cyan]"))
        self.console.print(f"Run ID: [bold]{info.run_id}[/bold]")
        self.console.print(
            f"Jobs: [bold]{info.job_count}[/bold]  "
            f"Backends: [bold]{info.primary_backend}[/bold] → "
            f"[bold]{secondary}[/bold]  GPUs: [bold]{gpu_text}[/bold]"
        )
        self.console.print(f"Results: {info.results_dir}", markup=False)
        self.console.print(f"Outputs: {info.output_dir}", markup=False)
        self.console.print(f"Stage logs: {info.logs_dir}", markup=False)

    def _stage_spec(self, key: str) -> StageSpec:
        for stage in self._stages:
            if stage.key == key:
                return stage
        return StageSpec(key, key.replace("_", " ").title())

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def stage_started(self, stage: str, *, log_dir: Path) -> None:
        self._stop_progress()
        self._active_stage = stage
        self._stage_started_at[stage] = time.monotonic()
        self._stage_logs[stage] = log_dir
        spec = self._stage_spec(stage)
        index = self._stage_index.get(stage, 0)
        total = len(self._stages)
        prefix = f"[Stage {index}/{total}]" if index else "[Stage]"
        self.console.print(f"\n[bold cyan]{prefix} {spec.label}[/bold cyan]")
        if self._interactive:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TextColumn("ETA"),
                TimeRemainingColumn(),
                TextColumn("{task.fields[detail]}"),
                console=self.console,
                refresh_per_second=4,
                transient=False,
            )
            self._progress.start()

    def cache_status(
        self,
        stage: str,
        *,
        hits: int,
        misses: int,
        total: int,
        force: bool = False,
    ) -> None:
        if force:
            self.console.print(
                f"  [yellow]cache bypassed by --force[/yellow]: {total}/{total}"
            )
            return
        self.console.print(
            f"  [green]cache hit[/green]: {hits}/{total}  |  "
            f"[yellow]cache missing[/yellow]: {misses}/{total}"
        )

    @staticmethod
    def _counter_detail(
        *,
        success: int | None,
        failed: int | None,
        skipped: int | None,
        detail: str,
    ) -> str:
        parts: list[str] = []
        if success is not None:
            parts.append(f"success={success}")
        if failed is not None:
            parts.append(f"failed={failed}")
        if skipped is not None:
            parts.append(f"skipped={skipped}")
        if detail:
            parts.append(detail)
        return "  " + " ".join(parts) if parts else ""

    def task_started(
        self,
        stage: str,
        task: str,
        *,
        total: int | None,
        completed: int = 0,
        detail: str = "",
    ) -> None:
        key = (stage, task)
        rich_id = None
        if self._progress is not None:
            rich_id = self._progress.add_task(
                task,
                total=total,
                completed=completed,
                detail=("  " + detail if detail else ""),
            )
        state = _TaskState(
            rich_id=rich_id,
            total=total,
            completed=completed,
            last_plain_at=time.monotonic(),
        )
        self._tasks[key] = state
        if not self._interactive:
            total_text = str(total) if total is not None else "?"
            suffix = f"  {detail}" if detail else ""
            self.console.print(f"  {task}: {completed}/{total_text}{suffix}")

    def _plain_progress_due(
        self,
        state: _TaskState,
        *,
        completed: int,
        total: int | None,
    ) -> bool:
        now = time.monotonic()
        if total is not None and completed >= total:
            state.last_plain_at = now
            return True
        if now - state.last_plain_at >= 30:
            state.last_plain_at = now
            return True
        if total:
            bucket = int((completed * 20) / total)
            if bucket > state.last_plain_bucket:
                state.last_plain_bucket = bucket
                state.last_plain_at = now
                return True
        return False

    def task_progress(
        self,
        stage: str,
        task: str,
        *,
        completed: int,
        total: int | None = None,
        success: int | None = None,
        failed: int | None = None,
        skipped: int | None = None,
        detail: str = "",
    ) -> None:
        key = (stage, task)
        state = self._tasks.get(key)
        if state is None:
            self.task_started(
                stage,
                task,
                total=total,
                completed=completed,
                detail=detail,
            )
            state = self._tasks[key]
        effective_total = state.total if total is None else total
        completed = max(state.completed, completed)
        counter_detail = self._counter_detail(
            success=success,
            failed=failed,
            skipped=skipped,
            detail=detail,
        )
        if self._progress is not None and state.rich_id is not None:
            self._progress.update(
                state.rich_id,
                completed=completed,
                total=effective_total,
                detail=counter_detail,
                refresh=True,
            )
        elif self._plain_progress_due(
            state,
            completed=completed,
            total=effective_total,
        ):
            total_text = str(effective_total) if effective_total is not None else "?"
            self.console.print(
                f"  {task}: {completed}/{total_text}{counter_detail}"
            )
        state.completed = completed
        state.total = effective_total

    def task_finished(
        self,
        stage: str,
        task: str,
        *,
        completed: int | None = None,
        total: int | None = None,
        success: int | None = None,
        failed: int | None = None,
        skipped: int | None = None,
        detail: str = "",
    ) -> None:
        state = self._tasks.get((stage, task))
        effective_total = total if total is not None else (state.total if state else None)
        effective_completed = (
            completed
            if completed is not None
            else (
                effective_total
                if effective_total is not None
                else (state.completed if state else 0)
            )
        )
        self.task_progress(
            stage,
            task,
            completed=effective_completed,
            total=effective_total,
            success=success,
            failed=failed,
            skipped=skipped,
            detail=detail,
        )

    def stage_finished(
        self,
        stage: str,
        *,
        status: str,
        detail: str = "",
    ) -> None:
        self._stop_progress()
        elapsed = time.monotonic() - self._stage_started_at.get(stage, time.monotonic())
        normalized = status.upper().replace("_", "-")
        color = {
            "SUCCESS": "green",
            "DRY-RUN": "cyan",
            "SKIPPED": "yellow",
            "DISABLED": "yellow",
            "PARTIAL": "yellow",
            "FAILED": "red",
            "ERROR": "red",
        }.get(normalized, "white")
        suffix = f" — {detail}" if detail else ""
        self.console.print(
            f"  [{color}]{normalized}[/{color}] in {elapsed:.1f}s{suffix}"
        )
        if normalized in {"FAILED", "ERROR", "PARTIAL"}:
            log_dir = self._stage_logs.get(stage)
            if log_dir is not None:
                self.console.print(f"  Logs: {log_dir}", markup=False)
        if self._active_stage == stage:
            self._active_stage = None

    def message(self, text: str, *, level: str = "info") -> None:
        style = {
            "warning": "yellow",
            "error": "red",
            "success": "green",
        }.get(level)
        if style:
            self.console.print(f"[{style}]{text}[/{style}]")
        else:
            self.console.print(text)

    def pipeline_finished(
        self,
        *,
        status: str,
        detail: str = "",
    ) -> None:
        self._stop_progress()
        elapsed = (
            time.monotonic() - self._pipeline_started_at
            if self._pipeline_started_at is not None
            else 0.0
        )
        normalized = status.upper().replace("_", "-")
        color = {
            "SUCCESS": "green",
            "DRY-RUN": "cyan",
            "PARTIAL": "yellow",
            "FAILED": "red",
            "ERROR": "red",
        }.get(normalized, "white")
        self.console.print(Rule())
        suffix = f" — {detail}" if detail else ""
        self.console.print(
            f"[bold {color}]Pipeline {normalized}[/bold {color}] "
            f"in {elapsed:.1f}s{suffix}"
        )

    def close(self) -> None:
        if self._closed:
            return
        self._stop_progress()
        self._closed = True
