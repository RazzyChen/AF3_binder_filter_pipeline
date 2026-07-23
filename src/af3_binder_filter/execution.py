"""Process-safe local execution primitives for pipeline stages.

The production workflow currently builds complete command lines before it
launches them.  This module deliberately keeps that boundary: it owns process
lifecycle, logs, timeouts, and structured outcomes, but it does not know how an
AF3, Protenix, OpenDDE, ESM, or Foldseek command is assembled.  That makes it a
small foundation for gradually moving stage execution out of ``workflow.py``.
"""

from __future__ import annotations

import os
import re
import shlex
import signal as signal_module
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping, Protocol, Sequence, TextIO

from af3_binder_filter.io_utils import atomic_write_text


EventKind = Literal[
    "stage_started",
    "command_started",
    "command_finished",
    "command_timed_out",
    "command_interrupted",
    "stage_finished",
]


@dataclass(frozen=True, slots=True)
class CommandSpec:
    """Immutable description of one already-assembled external command.

    ``environment`` contains overrides rather than a mutable environment
    dictionary.  It is normalized to an immutable, sorted tuple so a caller
    cannot change command semantics while the command is running.
    """

    argv: tuple[str, ...]
    name: str
    stdout_path: Path
    stderr_path: Path
    command_path: Path | None = None
    cwd: Path | None = None
    environment: tuple[tuple[str, str], ...] = ()
    inherit_environment: bool = True
    timeout_seconds: float | None = None
    stage: str = ""
    shard_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.argv, str):
            raise TypeError("argv must be a sequence of arguments, not a string")
        argv = tuple(str(argument) for argument in self.argv)
        if not argv or not argv[0]:
            raise ValueError("argv must contain an executable")
        if any("\x00" in argument for argument in argv):
            raise ValueError("argv must not contain NUL bytes")
        if not self.name.strip():
            raise ValueError("command name must not be empty")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")

        raw_environment: Sequence[tuple[str, str]]
        if isinstance(self.environment, Mapping):
            raw_environment = tuple(self.environment.items())
        else:
            raw_environment = tuple(self.environment)
        normalized_environment = tuple(
            sorted((str(key), str(value)) for key, value in raw_environment)
        )
        keys = [key for key, _value in normalized_environment]
        if any(not key or "=" in key or "\x00" in key for key in keys):
            raise ValueError("environment contains an invalid variable name")
        if len(keys) != len(set(keys)):
            raise ValueError("environment contains duplicate variable names")
        if any("\x00" in value for _key, value in normalized_environment):
            raise ValueError("environment values must not contain NUL bytes")

        object.__setattr__(self, "argv", argv)
        object.__setattr__(self, "stdout_path", Path(self.stdout_path))
        object.__setattr__(self, "stderr_path", Path(self.stderr_path))
        if self.command_path is not None:
            object.__setattr__(self, "command_path", Path(self.command_path))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", Path(self.cwd))
        object.__setattr__(self, "environment", normalized_environment)

    @classmethod
    def logged(
        cls,
        argv: Sequence[str],
        *,
        log_dir: Path,
        name: str,
        cwd: Path | None = None,
        environment: Mapping[str, str] | Sequence[tuple[str, str]] = (),
        inherit_environment: bool = True,
        timeout_seconds: float | None = None,
        stage: str = "",
        shard_id: str | int | None = None,
    ) -> CommandSpec:
        """Build a spec using Aerith's existing per-command log convention."""

        root = Path(log_dir)
        return cls(
            argv=tuple(argv),
            name=name,
            stdout_path=root / f"{name}.stdout.log",
            stderr_path=root / f"{name}.stderr.log",
            command_path=root / f"{name}.command.txt",
            cwd=cwd,
            environment=environment,  # type: ignore[arg-type]
            inherit_environment=inherit_environment,
            timeout_seconds=timeout_seconds,
            stage=stage,
            shard_id=None if shard_id is None else str(shard_id),
        )

    def resolved_environment(self) -> dict[str, str] | None:
        """Return the environment passed to ``Popen``.

        ``None`` preserves Python's native inheritance fast path when there are
        no overrides.  An explicitly isolated environment is represented by an
        empty dictionary.
        """

        if self.inherit_environment and not self.environment:
            return None
        environment = dict(os.environ) if self.inherit_environment else {}
        environment.update(self.environment)
        return environment


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """Complete, non-lossy result of one external command."""

    command: CommandSpec
    returncode: int | None
    duration_seconds: float
    timed_out: bool = False
    signal: int | None = None
    dry_run: bool = False
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        """Only a real or simulated zero exit is successful."""

        return (
            self.returncode == 0
            and not self.timed_out
            and self.error is None
        )

    @property
    def failed(self) -> bool:
        return not self.succeeded

    @property
    def status(self) -> str:
        if self.dry_run:
            return "dry_run"
        if self.timed_out:
            return "timeout"
        return "success" if self.succeeded else "error"


@dataclass(frozen=True, slots=True)
class CancellationReport:
    """Summary of one collective cancellation request."""

    requested: int
    term_signalled: int
    kill_signalled: int
    reaped: int
    still_running: int
    cleaned_containers: tuple[str, ...] = ()
    cleanup_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Presentation-neutral event emitted at stage and shard boundaries."""

    kind: EventKind
    stage: str
    command_name: str = ""
    shard_id: str | None = None
    status: str = ""
    returncode: int | None = None
    duration_seconds: float | None = None
    timestamp_monotonic: float = 0.0
    detail: str = ""


class EventSink(Protocol):
    def __call__(self, event: ExecutionEvent) -> None: ...


class Executor(Protocol):
    def run(self, command: CommandSpec, *, dry_run: bool = False) -> CommandOutcome: ...


PopenFactory = Callable[..., subprocess.Popen[str]]
CleanupRunner = Callable[..., subprocess.CompletedProcess[str]]
Clock = Callable[[], float]


@dataclass(slots=True)
class _ActiveProcess:
    process: subprocess.Popen[str]
    command: CommandSpec
    cleanup_claimed: bool = False


def _signal_from_returncode(returncode: int | None) -> int | None:
    return -returncode if returncode is not None and returncode < 0 else None


def _emit_safely(sink: EventSink | None, event: ExecutionEvent) -> None:
    if sink is None:
        return
    try:
        sink(event)
    except Exception:
        # Events are observational and must never change command semantics.
        return


class LocalCommandExecutor:
    """Run one local command with durable logs and process-group cleanup."""

    def __init__(
        self,
        *,
        termination_grace_seconds: float = 10.0,
        popen_factory: PopenFactory = subprocess.Popen,
        clock: Clock = time.monotonic,
        event_sink: EventSink | None = None,
    ) -> None:
        if termination_grace_seconds <= 0:
            raise ValueError("termination_grace_seconds must be greater than zero")
        self.termination_grace_seconds = termination_grace_seconds
        self._popen = popen_factory
        self._clock = clock
        self._event_sink = event_sink
        self._registry_lock = threading.RLock()
        self._cancel_lock = threading.Lock()
        self._active_processes: dict[subprocess.Popen[str], _ActiveProcess] = {}
        self._cancellation_requested = False

    @property
    def active_process_count(self) -> int:
        """Number of commands currently owned by this executor."""

        with self._registry_lock:
            return len(self._active_processes)

    @property
    def cancellation_requested(self) -> bool:
        """Whether this stage-scoped executor rejects new process starts."""

        with self._cancel_lock:
            return self._cancellation_requested

    def reset_cancellation(self) -> None:
        """Allow new starts after a completed cancellation barrier.

        Resetting while a process is registered could let new work overlap a
        stage that is still unwinding, so it is rejected deterministically.
        """

        with self._cancel_lock:
            if self.active_process_count:
                raise RuntimeError("cannot reset cancellation while processes are active")
            self._cancellation_requested = False

    def _register_process(
        self,
        process: subprocess.Popen[str],
        command: CommandSpec,
    ) -> _ActiveProcess:
        record = _ActiveProcess(process=process, command=command)
        with self._registry_lock:
            self._active_processes[process] = record
        return record

    def _unregister_process(self, process: subprocess.Popen[str]) -> None:
        with self._registry_lock:
            self._active_processes.pop(process, None)

    def _active_snapshot(self) -> tuple[_ActiveProcess, ...]:
        with self._registry_lock:
            return tuple(self._active_processes.values())

    def _record_for_process(
        self,
        process: subprocess.Popen[str],
    ) -> _ActiveProcess | None:
        with self._registry_lock:
            return self._active_processes.get(process)

    def _emit(
        self,
        kind: EventKind,
        command: CommandSpec,
        *,
        status: str = "",
        returncode: int | None = None,
        duration_seconds: float | None = None,
        detail: str = "",
    ) -> None:
        _emit_safely(
            self._event_sink,
            ExecutionEvent(
                kind=kind,
                stage=command.stage,
                command_name=command.name,
                shard_id=command.shard_id,
                status=status,
                returncode=returncode,
                duration_seconds=duration_seconds,
                timestamp_monotonic=self._clock(),
                detail=detail,
            ),
        )

    @staticmethod
    def _write_command_record(command: CommandSpec) -> None:
        if command.command_path is None:
            return
        atomic_write_text(command.command_path, shlex.join(command.argv) + "\n")

    @staticmethod
    def _send_signal(process: subprocess.Popen[str], signal_number: int) -> bool:
        if os.name == "posix":
            try:
                # start_new_session=True makes the child PID its process-group
                # ID, so descendants launched by Docker wrappers are included.
                os.killpg(process.pid, signal_number)
                return True
            except ProcessLookupError:
                return False
            except (OSError, PermissionError):
                # Fall back to the direct child if process-group signalling is
                # unavailable (for example in a restricted test environment).
                pass
        try:
            process.send_signal(signal_number)
            return True
        except ProcessLookupError:
            return False

    @staticmethod
    def _process_running(record: _ActiveProcess) -> bool:
        try:
            return record.process.poll() is None
        except (ChildProcessError, ProcessLookupError):
            return False

    @classmethod
    def _wait_for_records(
        cls,
        records: Sequence[_ActiveProcess],
        *,
        timeout_seconds: float,
    ) -> tuple[_ActiveProcess, ...]:
        """Wait for a group against one shared deadline, not N timeouts."""

        pending = [record for record in records if cls._process_running(record)]
        deadline = time.monotonic() + timeout_seconds
        while pending:
            pending = [record for record in pending if cls._process_running(record)]
            if not pending:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                pending[0].process.wait(timeout=min(0.05, remaining))
            except subprocess.TimeoutExpired:
                continue
            except (ChildProcessError, ProcessLookupError):
                continue
        return tuple(record for record in pending if cls._process_running(record))

    def _claim_cleanup(
        self,
        records: Sequence[_ActiveProcess],
    ) -> tuple[_ActiveProcess, ...]:
        claimed: list[_ActiveProcess] = []
        with self._registry_lock:
            for record in records:
                if record.cleanup_claimed:
                    continue
                record.cleanup_claimed = True
                claimed.append(record)
        return tuple(claimed)

    def _cleanup_cancelled_processes(
        self,
        records: Sequence[_ActiveProcess],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Backend-specific cleanup hook called at most once per process."""

        return (), ()

    def _cancel_records_locked(
        self,
        records: Sequence[_ActiveProcess],
    ) -> CancellationReport:
        unique_records = tuple(dict.fromkeys(record.process for record in records))
        by_process = {record.process: record for record in records}
        ordered = tuple(by_process[process] for process in unique_records)
        targets = tuple(record for record in ordered if self._process_running(record))

        term_signalled = sum(
            self._send_signal(record.process, signal_module.SIGTERM) for record in targets
        )
        remaining = self._wait_for_records(
            targets,
            timeout_seconds=self.termination_grace_seconds,
        )
        kill_signalled = sum(
            self._send_signal(record.process, signal_module.SIGKILL) for record in remaining
        )
        remaining = self._wait_for_records(
            remaining,
            timeout_seconds=self.termination_grace_seconds,
        )
        remaining_processes = {record.process for record in remaining}

        for record in ordered:
            if record.process not in remaining_processes:
                self._unregister_process(record.process)

        cleanup_records = self._claim_cleanup(targets)
        cleaned_containers, cleanup_errors = self._cleanup_cancelled_processes(cleanup_records)
        return CancellationReport(
            requested=len(ordered),
            term_signalled=term_signalled,
            kill_signalled=kill_signalled,
            reaped=len(ordered) - len(remaining),
            still_running=len(remaining),
            cleaned_containers=cleaned_containers,
            cleanup_errors=cleanup_errors,
        )

    def cancel_all(self) -> CancellationReport:
        """Cancel, reap, and unregister every command active at call time.

        Cancellation calls are serialized, while command registration and
        completion remain concurrent.  TERM and KILL are broadcast before each
        bounded wait so cancellation latency is independent of shard count.
        """

        with self._cancel_lock:
            self._cancellation_requested = True
            return self._cancel_records_locked(self._active_snapshot())

    def _terminate_process_group(self, process: subprocess.Popen[str]) -> int | None:
        with self._cancel_lock:
            record = self._record_for_process(process)
            if record is None:
                return process.returncode
            self._cancel_records_locked((record,))
            return process.returncode

    @staticmethod
    def _append_lifecycle_error(handle: TextIO, message: str) -> None:
        handle.write(f"\n[aerith executor] {message}\n")
        handle.flush()

    def run(self, command: CommandSpec, *, dry_run: bool = False) -> CommandOutcome:
        command.stdout_path.parent.mkdir(parents=True, exist_ok=True)
        command.stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_command_record(command)

        started = self._clock()
        self._emit("command_started", command, status="dry_run" if dry_run else "running")
        if dry_run:
            outcome = CommandOutcome(
                command=command,
                returncode=0,
                duration_seconds=max(0.0, self._clock() - started),
                dry_run=True,
            )
            self._emit(
                "command_finished",
                command,
                status=outcome.status,
                returncode=outcome.returncode,
                duration_seconds=outcome.duration_seconds,
            )
            return outcome

        process: subprocess.Popen[str] | None = None
        with command.stdout_path.open("w", encoding="utf-8") as stdout, command.stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr:
            popen_kwargs: dict[str, object] = {
                "shell": False,
                "stdout": stdout,
                "stderr": stderr,
                "text": True,
            }
            if command.cwd is not None:
                popen_kwargs["cwd"] = command.cwd
            environment = command.resolved_environment()
            if environment is not None:
                popen_kwargs["env"] = environment
            if os.name == "posix":
                popen_kwargs["start_new_session"] = True

            try:
                # Starting and registering share the cancellation lock.  A
                # concurrent cancel_all() therefore sees either no process yet
                # or the fully registered process, never an untracked child.
                cancelled = False
                with self._cancel_lock:
                    if self._cancellation_requested:
                        cancelled = True
                    else:
                        process = self._popen(list(command.argv), **popen_kwargs)
                        self._register_process(process, command)
                if cancelled:
                    error = "executor cancellation is active; command was not started"
                    self._append_lifecycle_error(stderr, error)
                    outcome = CommandOutcome(
                        command=command,
                        returncode=None,
                        duration_seconds=max(0.0, self._clock() - started),
                        error=error,
                    )
                    self._emit(
                        "command_finished",
                        command,
                        status=outcome.status,
                        duration_seconds=outcome.duration_seconds,
                        detail=error,
                    )
                    return outcome
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                error = f"failed to start command: {type(exc).__name__}: {exc}"
                self._append_lifecycle_error(stderr, error)
                outcome = CommandOutcome(
                    command=command,
                    returncode=None,
                    duration_seconds=max(0.0, self._clock() - started),
                    error=error,
                )
                self._emit(
                    "command_finished",
                    command,
                    status=outcome.status,
                    duration_seconds=outcome.duration_seconds,
                    detail=error,
                )
                return outcome

            try:
                try:
                    if command.timeout_seconds is None:
                        returncode = process.wait()
                    else:
                        returncode = process.wait(timeout=command.timeout_seconds)
                except subprocess.TimeoutExpired:
                    returncode = self._terminate_process_group(process)
                    error = f"exceeded timeout of {command.timeout_seconds:g} seconds"
                    self._append_lifecycle_error(stderr, error)
                    outcome = CommandOutcome(
                        command=command,
                        returncode=returncode,
                        duration_seconds=max(0.0, self._clock() - started),
                        timed_out=True,
                        signal=_signal_from_returncode(returncode),
                        error=error,
                    )
                    self._emit(
                        "command_timed_out",
                        command,
                        status=outcome.status,
                        returncode=returncode,
                        duration_seconds=outcome.duration_seconds,
                        detail=error,
                    )
                    return outcome
                except KeyboardInterrupt:
                    returncode = self._terminate_process_group(process)
                    duration = max(0.0, self._clock() - started)
                    self._append_lifecycle_error(
                        stderr,
                        "interrupted; process group terminated",
                    )
                    self._emit(
                        "command_interrupted",
                        command,
                        status="interrupted",
                        returncode=returncode,
                        duration_seconds=duration,
                    )
                    raise
                except BaseException:
                    self._terminate_process_group(process)
                    raise

                outcome = CommandOutcome(
                    command=command,
                    returncode=returncode,
                    duration_seconds=max(0.0, self._clock() - started),
                    signal=_signal_from_returncode(returncode),
                )
                self._emit(
                    "command_finished",
                    command,
                    status=outcome.status,
                    returncode=returncode,
                    duration_seconds=outcome.duration_seconds,
                )
                return outcome
            finally:
                self._unregister_process(process)


class LocalDockerExecutor(LocalCommandExecutor):
    """Local executor restricted to already-assembled Docker CLI commands."""

    def __init__(
        self,
        *,
        docker_executable: str = "docker",
        termination_grace_seconds: float = 10.0,
        popen_factory: PopenFactory = subprocess.Popen,
        cleanup_runner: CleanupRunner = subprocess.run,
        cleanup_timeout_seconds: float = 30.0,
        clock: Clock = time.monotonic,
        event_sink: EventSink | None = None,
    ) -> None:
        if not docker_executable:
            raise ValueError("docker_executable must not be empty")
        if cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup_timeout_seconds must be greater than zero")
        super().__init__(
            termination_grace_seconds=termination_grace_seconds,
            popen_factory=popen_factory,
            clock=clock,
            event_sink=event_sink,
        )
        self.docker_executable = docker_executable
        self._cleanup_runner = cleanup_runner
        self.cleanup_timeout_seconds = cleanup_timeout_seconds

    _CONTAINER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

    def _named_run_container(self, command: CommandSpec) -> str | None:
        argv = command.argv
        if len(argv) < 3 or Path(argv[0]).name != Path(self.docker_executable).name:
            return None
        if argv[1] != "run":
            return None
        names: list[str] = []
        index = 2
        while index < len(argv):
            argument = argv[index]
            if argument == "--name":
                if index + 1 >= len(argv):
                    return None
                names.append(argv[index + 1])
                index += 2
                continue
            if argument.startswith("--name="):
                names.append(argument.partition("=")[2])
            index += 1
        if len(names) != 1 or self._CONTAINER_NAME.fullmatch(names[0]) is None:
            return None
        return names[0]

    def _cleanup_cancelled_processes(
        self,
        records: Sequence[_ActiveProcess],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        names = tuple(
            sorted(
                {
                    name
                    for record in records
                    if (name := self._named_run_container(record.command)) is not None
                }
            )
        )
        cleaned: list[str] = []
        errors: list[str] = []
        for name in names:
            command = [self.docker_executable, "rm", "-f", name]
            try:
                completed = self._cleanup_runner(
                    command,
                    shell=False,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.cleanup_timeout_seconds,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "").strip()
                errors.append(f"{name}: docker rm returned {completed.returncode}: {detail}")
                continue
            cleaned.append(name)
        return tuple(cleaned), tuple(errors)

    def run(self, command: CommandSpec, *, dry_run: bool = False) -> CommandOutcome:
        actual = Path(command.argv[0]).name
        expected = Path(self.docker_executable).name
        if actual != expected:
            raise ValueError(
                f"LocalDockerExecutor expected executable {expected!r}, got {actual!r}"
            )
        return super().run(command, dry_run=dry_run)


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Aggregate outcome for a bounded set of commands in one stage."""

    stage: str
    commands: tuple[CommandOutcome, ...]
    expected_commands: int
    duration_seconds: float
    status: Literal["skipped", "dry_run", "success", "partial", "error"]

    @property
    def succeeded(self) -> bool:
        return self.status in {"success", "dry_run", "skipped"}


def _stage_status(
    outcomes: Sequence[CommandOutcome],
    *,
    expected_commands: int,
) -> Literal["skipped", "dry_run", "success", "partial", "error"]:
    if expected_commands == 0:
        return "skipped"
    if outcomes and all(outcome.dry_run for outcome in outcomes):
        return "dry_run"
    succeeded = sum(outcome.succeeded for outcome in outcomes)
    failed = len(outcomes) - succeeded
    unstarted = expected_commands - len(outcomes)
    if succeeded == expected_commands:
        return "success"
    if succeeded > 0 and (failed > 0 or unstarted > 0):
        return "partial"
    return "error"


class StageRunner:
    """Sequential stage boundary for gradual workflow extraction.

    GPU sharding remains owned by the current scheduler.  A later integration
    can call one runner per scheduled shard (as the workflow already does in a
    thread pool) without introducing a second scheduler or a cross-stage DAG.
    """

    def __init__(
        self,
        executor: Executor,
        *,
        event_sink: EventSink | None = None,
        clock: Clock = time.monotonic,
    ) -> None:
        self.executor = executor
        self._event_sink = event_sink
        self._clock = clock

    def run(
        self,
        stage: str,
        commands: Sequence[CommandSpec],
        *,
        dry_run: bool = False,
        stop_on_failure: bool = False,
    ) -> StageOutcome:
        if not stage.strip():
            raise ValueError("stage must not be empty")
        if any(command.stage and command.stage != stage for command in commands):
            raise ValueError("every command must belong to the requested stage")

        started = self._clock()
        _emit_safely(
            self._event_sink,
            ExecutionEvent(
                kind="stage_started",
                stage=stage,
                status="running",
                timestamp_monotonic=started,
            ),
        )
        outcomes: list[CommandOutcome] = []
        try:
            for command in commands:
                outcome = self.executor.run(command, dry_run=dry_run)
                outcomes.append(outcome)
                if stop_on_failure and outcome.failed:
                    break
        except KeyboardInterrupt:
            _emit_safely(
                self._event_sink,
                ExecutionEvent(
                    kind="stage_finished",
                    stage=stage,
                    status="interrupted",
                    duration_seconds=max(0.0, self._clock() - started),
                    timestamp_monotonic=self._clock(),
                ),
            )
            raise
        except BaseException as exc:
            _emit_safely(
                self._event_sink,
                ExecutionEvent(
                    kind="stage_finished",
                    stage=stage,
                    status="error",
                    duration_seconds=max(0.0, self._clock() - started),
                    timestamp_monotonic=self._clock(),
                    detail=f"{type(exc).__name__}: {exc}",
                ),
            )
            raise

        duration = max(0.0, self._clock() - started)
        status = _stage_status(outcomes, expected_commands=len(commands))
        stage_outcome = StageOutcome(
            stage=stage,
            commands=tuple(outcomes),
            expected_commands=len(commands),
            duration_seconds=duration,
            status=status,
        )
        _emit_safely(
            self._event_sink,
            ExecutionEvent(
                kind="stage_finished",
                stage=stage,
                status=status,
                duration_seconds=duration,
                timestamp_monotonic=self._clock(),
            ),
        )
        return stage_outcome
