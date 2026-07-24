"""Cohesive command runtime orchestration boundary."""

from __future__ import annotations

import json
import shlex
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    wait,
)
from pathlib import Path
from typing import (
    Callable,
    Sequence,
)

from af3_binder_filter.execution import (
    CommandSpec,
    LocalCommandExecutor,
    LocalDockerExecutor,
)
from af3_binder_filter.io_utils import atomic_write_text
from af3_binder_filter.orchestration.context import (
    GpuJobShard,
    PipelineExecutionError,
    RunContext,
)


def _command_stage_name(name: str) -> str:
    if name.startswith("secondary_prediction"):
        return "secondary_prediction"
    if name.startswith(("esmfold", "esm_if")):
        return "esm"
    if name.startswith("target_features"):
        return "features"
    if name.startswith(("primary_prediction", "prediction")):
        return "primary_prediction"
    return name


def run_prediction_command(
    context: RunContext,
    command: Sequence[str],
    *,
    name: str = "prediction",
    timeout_seconds: float | None = None,
    executor: LocalCommandExecutor | None = None,
) -> int | None:
    log_dir = context.layout.stage(_command_stage_name(name)).logs
    if not command:
        raise PipelineExecutionError(f"{name} command is empty")
    runner = executor or LocalDockerExecutor(
        docker_executable=str(command[0]),
    )
    outcome = runner.run(
        CommandSpec.logged(
            command,
            log_dir=log_dir,
            name=name,
            timeout_seconds=timeout_seconds,
            stage=_command_stage_name(name),
        ),
        dry_run=context.config.runtime.dry_run,
    )
    if outcome.error is not None:
        raise PipelineExecutionError(f"{name}: {outcome.error}")
    return outcome.returncode


def run_sharded_commands(
    context: RunContext,
    stage_name: str,
    commands: Sequence[tuple[GpuJobShard, Sequence[str]]],
    *,
    progress_probe: Callable[[], int] | None = None,
    progress_callback: Callable[[int], None] | None = None,
    timeout_seconds: float | None = None,
    executor: LocalCommandExecutor | None = None,
) -> tuple[dict[int, int | None], list[str]]:
    """Run one Docker command per GPU concurrently and preserve every log."""

    log_dir = context.layout.stage(_command_stage_name(stage_name)).logs
    atomic_write_text(
        log_dir / f"{stage_name}.command.txt",
        "".join(
            "# gpu={} jobs={}\n".format(
                shard.gpu.index,
                ",".join(job.job_id for job in shard.jobs),
            )
            + shlex.join(command)
            + "\n"
            for shard, command in commands
        ),
    )
    if not commands:
        return {}, []
    if executor is None:
        executables = {str(command[0]) for _shard, command in commands if command}
        if len(executables) != 1 or any(not command for _shard, command in commands):
            raise PipelineExecutionError(
                f"{stage_name} shards must share one non-empty Docker executable"
            )
        executor = LocalDockerExecutor(docker_executable=executables.pop())

    return_codes: dict[int, int | None] = {}
    errors_by_gpu: dict[int, str] = {}
    pool = ThreadPoolExecutor(max_workers=max(1, len(commands)))
    futures = {}
    try:
        for shard, command in commands:
            spec = CommandSpec.logged(
                command,
                log_dir=log_dir,
                name=f"{stage_name}.gpu_{shard.gpu.index}",
                timeout_seconds=timeout_seconds,
                stage=_command_stage_name(stage_name),
                shard_id=shard.gpu.index,
            )
            future = pool.submit(
                executor.run,
                spec,
                dry_run=context.config.runtime.dry_run,
            )
            futures[future] = shard
        pending_futures = set(futures)
        while pending_futures:
            done, pending_futures = wait(
                pending_futures,
                timeout=1.0,
                return_when=FIRST_COMPLETED,
            )
            if progress_probe is not None and progress_callback is not None:
                try:
                    progress_callback(progress_probe())
                except Exception:
                    # Progress is observational and must never change execution
                    # or failure semantics.
                    pass
            for future in done:
                shard = futures[future]
                try:
                    outcome = future.result()
                except Exception as exc:
                    return_codes[shard.gpu.index] = None
                    errors_by_gpu[shard.gpu.index] = (
                        f"{stage_name} GPU {shard.gpu.index} raised {type(exc).__name__}: {exc}"
                    )
                else:
                    return_codes[shard.gpu.index] = outcome.returncode
                    if outcome.error is not None:
                        errors_by_gpu[shard.gpu.index] = (
                            f"{stage_name} GPU {shard.gpu.index}: {outcome.error}"
                        )
        if progress_probe is not None and progress_callback is not None:
            try:
                progress_callback(progress_probe())
            except Exception:
                pass
    except BaseException:
        try:
            executor.cancel_all()
        except Exception:
            # Preserve the original interruption/exception. Executor cleanup
            # is best effort after all registered process groups were signalled.
            pass
        for future in futures:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return return_codes, [errors_by_gpu[index] for index in sorted(errors_by_gpu)]


def return_code_failure_message(
    stage_name: str,
    gpu_index: int,
    return_code: int | None,
) -> str:
    if return_code is None:
        return f"{stage_name} GPU {gpu_index} did not produce an exit code"
    return f"{stage_name} GPU {gpu_index} command returned {return_code}"


def file_signature(paths: Sequence[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in sorted(set(paths)):
        try:
            stat = path.stat()
        except OSError:
            continue
        if stat.st_size > 0:
            signature.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns))
    return tuple(signature)


def small_json_is_complete(path: Path) -> bool:
    try:
        return isinstance(json.loads(path.read_text(encoding="utf-8")), dict)
    except (OSError, json.JSONDecodeError):
        return False


def path_belongs_to_job(path: Path, job_id: str) -> bool:
    return job_id in path.parts or path.stem == job_id or path.stem.startswith(f"{job_id}_")


def stable_completion_probe(
    keys: Sequence[str],
    signature: Callable[[str], tuple[tuple[str, int, int], ...]],
) -> Callable[[], int]:
    """Count changed completion signatures after two stable observations."""

    baseline = {key: signature(key) for key in keys}
    observed: dict[str, tuple[tuple[str, int, int], ...]] = {}
    completed: set[str] = set()

    def probe() -> int:
        for key in keys:
            if key in completed:
                continue
            current = signature(key)
            if not current or current == baseline[key]:
                observed.pop(key, None)
                continue
            if observed.get(key) == current:
                completed.add(key)
            else:
                observed[key] = current
        return len(completed)

    return probe
