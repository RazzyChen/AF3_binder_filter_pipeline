from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from af3_binder_filter.execution import (
    CommandSpec,
    ExecutionEvent,
    LocalCommandExecutor,
    LocalDockerExecutor,
    StageRunner,
)


def _spec(
    tmp_path: Path,
    argv: tuple[str, ...],
    *,
    name: str = "command",
    timeout_seconds: float | None = None,
    stage: str = "prediction",
) -> CommandSpec:
    return CommandSpec.logged(
        argv,
        log_dir=tmp_path,
        name=name,
        timeout_seconds=timeout_seconds,
        stage=stage,
        shard_id="gpu_0",
    )


def _wait_for_active(executor: LocalCommandExecutor, expected: int) -> None:
    deadline = time.monotonic() + 5.0
    while executor.active_process_count != expected and time.monotonic() < deadline:
        time.sleep(0.01)
    assert executor.active_process_count == expected


def test_command_spec_is_immutable_and_copies_environment(tmp_path: Path) -> None:
    environment = {"AERITH_EXECUTION_TEST": "before"}
    command = CommandSpec.logged(
        (sys.executable, "-c", "pass"),
        log_dir=tmp_path,
        name="immutable",
        environment=environment,
    )
    environment["AERITH_EXECUTION_TEST"] = "after"

    assert command.environment == (("AERITH_EXECUTION_TEST", "before"),)
    with pytest.raises(FrozenInstanceError):
        command.name = "changed"  # type: ignore[misc]


def test_local_executor_streams_logs_and_emits_shard_events(tmp_path: Path) -> None:
    events: list[ExecutionEvent] = []
    command = _spec(
        tmp_path,
        (
            sys.executable,
            "-c",
            "import sys; print('stdout line'); print('stderr line', file=sys.stderr)",
        ),
    )

    executor = LocalCommandExecutor(event_sink=events.append)
    outcome = executor.run(command)

    assert outcome.succeeded
    assert outcome.status == "success"
    assert outcome.returncode == 0
    assert outcome.signal is None
    assert command.stdout_path.read_text() == "stdout line\n"
    assert command.stderr_path.read_text() == "stderr line\n"
    assert command.command_path is not None
    assert "stdout line" in command.command_path.read_text()
    assert [event.kind for event in events] == ["command_started", "command_finished"]
    assert all(event.stage == "prediction" for event in events)
    assert all(event.shard_id == "gpu_0" for event in events)
    assert executor.active_process_count == 0


@pytest.mark.skipif(os.name != "posix", reason="negative signal return codes are POSIX")
def test_negative_signal_return_code_is_failure(tmp_path: Path) -> None:
    command = _spec(
        tmp_path,
        (
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        ),
    )

    outcome = LocalCommandExecutor().run(command)

    assert outcome.returncode == -signal.SIGTERM
    assert outcome.signal == signal.SIGTERM
    assert outcome.failed
    assert outcome.status == "error"


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_timeout_terminates_process_group_and_is_not_success(tmp_path: Path) -> None:
    events: list[ExecutionEvent] = []
    command = _spec(
        tmp_path,
        (sys.executable, "-c", "import time; time.sleep(30)"),
        timeout_seconds=0.05,
    )

    executor = LocalCommandExecutor(
        termination_grace_seconds=0.05,
        event_sink=events.append,
    )
    outcome = executor.run(command)

    assert outcome.timed_out
    assert outcome.failed
    assert outcome.status == "timeout"
    assert outcome.returncode is not None and outcome.returncode != 0
    assert outcome.signal in {signal.SIGTERM, signal.SIGKILL}
    assert "exceeded timeout" in command.stderr_path.read_text()
    assert events[-1].kind == "command_timed_out"
    assert executor.active_process_count == 0


def test_keyboard_interrupt_terminates_process_group_and_closes_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent_signals: list[tuple[int, int]] = []

    class InterruptedProcess:
        pid = 424242
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            if timeout is None and self.returncode is None:
                raise KeyboardInterrupt
            self.returncode = -signal.SIGTERM
            return self.returncode

        def send_signal(self, signal_number: int) -> None:
            sent_signals.append((self.pid, signal_number))

    def popen(_argv: list[str], **_kwargs: Any) -> InterruptedProcess:
        return InterruptedProcess()

    monkeypatch.setattr(
        "af3_binder_filter.execution.os.killpg",
        lambda pid, signal_number: sent_signals.append((pid, signal_number)),
    )
    command = _spec(tmp_path, (sys.executable, "-c", "pass"))

    executor = LocalCommandExecutor(popen_factory=popen)  # type: ignore[arg-type]
    with pytest.raises(KeyboardInterrupt):
        executor.run(command)

    assert sent_signals == [(424242, signal.SIGTERM)]
    assert "process group terminated" in command.stderr_path.read_text()
    command.stdout_path.rename(tmp_path / "closed.stdout.log")
    command.stderr_path.rename(tmp_path / "closed.stderr.log")
    assert executor.active_process_count == 0


def test_dry_run_writes_only_command_record_and_never_starts_process(
    tmp_path: Path,
) -> None:
    def forbidden_popen(*_args: object, **_kwargs: object) -> subprocess.Popen[str]:
        raise AssertionError("dry-run must not start a subprocess")

    command = _spec(tmp_path, (sys.executable, "-c", "pass"))
    outcome = LocalCommandExecutor(popen_factory=forbidden_popen).run(  # type: ignore[arg-type]
        command,
        dry_run=True,
    )

    assert outcome.succeeded
    assert outcome.dry_run
    assert outcome.status == "dry_run"
    assert command.command_path is not None and command.command_path.is_file()
    assert not command.stdout_path.exists()
    assert not command.stderr_path.exists()


def test_local_docker_executor_validates_executable_without_invoking_docker(
    tmp_path: Path,
) -> None:
    command = _spec(tmp_path, (sys.executable, "-c", "pass"))

    with pytest.raises(ValueError, match="expected executable 'docker'"):
        LocalDockerExecutor().run(command, dry_run=True)


def test_start_error_never_leaks_registry_entry(tmp_path: Path) -> None:
    command = _spec(tmp_path, ("/aerith/does/not/exist",))
    executor = LocalCommandExecutor()

    outcome = executor.run(command)

    assert outcome.failed
    assert outcome.returncode is None
    assert executor.active_process_count == 0


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")
def test_cancel_all_reaps_two_concurrent_commands_and_closes_logs(
    tmp_path: Path,
) -> None:
    executor = LocalCommandExecutor(termination_grace_seconds=0.2)
    commands = tuple(
        _spec(
            tmp_path,
            (sys.executable, "-c", "import time; time.sleep(30)"),
            name=f"blocking_{index}",
        )
        for index in range(2)
    )
    outcomes: list[Any] = []
    errors: list[BaseException] = []

    def run(command: CommandSpec) -> None:
        try:
            outcomes.append(executor.run(command))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(command,)) for command in commands]
    for thread in threads:
        thread.start()
    _wait_for_active(executor, 2)
    with pytest.raises(RuntimeError, match="processes are active"):
        executor.reset_cancellation()

    report = executor.cancel_all()
    for thread in threads:
        thread.join(timeout=5)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    assert report.requested == 2
    assert report.term_signalled == 2
    assert report.reaped == 2
    assert report.still_running == 0
    assert len(outcomes) == 2
    assert all(outcome.failed and outcome.signal == signal.SIGTERM for outcome in outcomes)
    assert executor.active_process_count == 0
    for command in commands:
        command.stdout_path.rename(tmp_path / f"closed.{command.stdout_path.name}")
        command.stderr_path.rename(tmp_path / f"closed.{command.stderr_path.name}")

    second = executor.cancel_all()
    assert second.requested == 0
    assert second.reaped == 0


class _ControlledProcess:
    def __init__(self, pid: int, argv: list[str], *, ignore_term: bool = False) -> None:
        self.pid = pid
        self.argv = argv
        self.ignore_term = ignore_term
        self.returncode: int | None = None
        self._finished = threading.Event()
        self._lock = threading.Lock()

    def poll(self) -> int | None:
        with self._lock:
            return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if not self._finished.wait(timeout):
            raise subprocess.TimeoutExpired(self.argv, timeout)
        with self._lock:
            assert self.returncode is not None
            return self.returncode

    def send_signal(self, signal_number: int) -> None:
        with self._lock:
            if signal_number == signal.SIGTERM and self.ignore_term:
                return
            if self.returncode is None:
                self.returncode = -signal_number
                self._finished.set()


def test_cancellation_is_sticky_for_start_waiting_behind_cancel_all(
    tmp_path: Path,
) -> None:
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    popen_calls: list[list[str]] = []

    class PausedCancellationExecutor(LocalCommandExecutor):
        def _active_snapshot(self) -> tuple[Any, ...]:
            snapshot_entered.set()
            assert release_snapshot.wait(timeout=5)
            return super()._active_snapshot()

    def forbidden_popen(argv: list[str], **_kwargs: Any) -> subprocess.Popen[str]:
        popen_calls.append(argv)
        raise AssertionError("sticky cancellation must prevent Popen")

    executor = PausedCancellationExecutor(
        popen_factory=forbidden_popen,
        termination_grace_seconds=0.1,
    )
    cancel_reports: list[Any] = []
    cancel_thread = threading.Thread(target=lambda: cancel_reports.append(executor.cancel_all()))
    cancel_thread.start()
    assert snapshot_entered.wait(timeout=5)

    blocked_command = _spec(tmp_path, (sys.executable, "-c", "pass"), name="blocked")
    outcomes: list[Any] = []
    worker = threading.Thread(target=lambda: outcomes.append(executor.run(blocked_command)))
    worker.start()
    deadline = time.monotonic() + 5
    while not blocked_command.stderr_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert blocked_command.stderr_path.exists()

    release_snapshot.set()
    cancel_thread.join(timeout=5)
    worker.join(timeout=5)

    assert not cancel_thread.is_alive()
    assert not worker.is_alive()
    assert cancel_reports[0].requested == 0
    assert executor.cancellation_requested
    assert popen_calls == []
    assert len(outcomes) == 1
    assert outcomes[0].failed
    assert outcomes[0].returncode is None
    assert outcomes[0].error == "executor cancellation is active; command was not started"
    assert "command was not started" in blocked_command.stderr_path.read_text()
    assert executor.active_process_count == 0

    future_command = _spec(tmp_path, (sys.executable, "-c", "pass"), name="future")
    future_outcome = executor.run(future_command)
    assert future_outcome.failed
    assert future_outcome.returncode is None
    assert popen_calls == []
    assert "command was not started" in future_command.stderr_path.read_text()

    executor.reset_cancellation()
    assert not executor.cancellation_requested


def test_cancel_all_escalates_to_kill_against_one_shared_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _ControlledProcess(
        499999,
        [sys.executable, "-c", "pass"],
        ignore_term=True,
    )
    monkeypatch.setattr(
        "af3_binder_filter.execution.os.killpg",
        lambda _pid, signal_number: process.send_signal(signal_number),
    )
    executor = LocalCommandExecutor(
        popen_factory=lambda *_args, **_kwargs: process,  # type: ignore[arg-type]
        termination_grace_seconds=0.02,
    )
    command = _spec(tmp_path, (sys.executable, "-c", "pass"))
    outcomes: list[Any] = []
    thread = threading.Thread(target=lambda: outcomes.append(executor.run(command)))
    thread.start()
    _wait_for_active(executor, 1)

    report = executor.cancel_all()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert report.term_signalled == 1
    assert report.kill_signalled == 1
    assert report.reaped == 1
    assert report.still_running == 0
    assert outcomes[0].signal == signal.SIGKILL
    assert executor.active_process_count == 0


def test_docker_cancel_cleanup_is_named_bounded_and_concurrently_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processes: dict[int, _ControlledProcess] = {}
    next_pid = 500000
    process_lock = threading.Lock()
    cleanup_calls: list[tuple[list[str], dict[str, Any]]] = []

    def popen(argv: list[str], **_kwargs: Any) -> _ControlledProcess:
        nonlocal next_pid
        with process_lock:
            next_pid += 1
            process = _ControlledProcess(next_pid, argv)
            processes[next_pid] = process
            return process

    def killpg(pid: int, signal_number: int) -> None:
        processes[pid].send_signal(signal_number)

    def cleanup_runner(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        cleanup_calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "removed", "")

    monkeypatch.setattr("af3_binder_filter.execution.os.killpg", killpg)
    executor = LocalDockerExecutor(
        popen_factory=popen,  # type: ignore[arg-type]
        cleanup_runner=cleanup_runner,
        termination_grace_seconds=0.1,
        cleanup_timeout_seconds=1.25,
    )
    commands = (
        _spec(tmp_path, ("docker", "run", "--name", "aerith-alpha", "image"), name="alpha"),
        _spec(tmp_path, ("docker", "run", "--name=aerith-beta", "image"), name="beta"),
        _spec(tmp_path, ("docker", "run", "image"), name="unnamed"),
        _spec(tmp_path, ("docker", "ps", "--name=aerith-not-a-run"), name="not_run"),
    )
    outcomes: list[Any] = []
    run_threads = [
        threading.Thread(target=lambda command=command: outcomes.append(executor.run(command)))
        for command in commands
    ]
    for thread in run_threads:
        thread.start()
    _wait_for_active(executor, len(commands))

    barrier = threading.Barrier(3)
    reports: list[Any] = []

    def cancel() -> None:
        barrier.wait()
        reports.append(executor.cancel_all())

    cancel_threads = [threading.Thread(target=cancel) for _ in range(2)]
    for thread in cancel_threads:
        thread.start()
    barrier.wait()
    for thread in cancel_threads + run_threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in cancel_threads + run_threads)
    assert sorted(report.requested for report in reports) == [0, 4]
    effective_report = next(report for report in reports if report.requested)
    assert effective_report.cleaned_containers == ("aerith-alpha", "aerith-beta")
    assert effective_report.cleanup_errors == ()
    assert executor.active_process_count == 0
    assert len(outcomes) == 4
    assert [call[0] for call in cleanup_calls] == [
        ["docker", "rm", "-f", "aerith-alpha"],
        ["docker", "rm", "-f", "aerith-beta"],
    ]
    assert all(call[1]["shell"] is False for call in cleanup_calls)
    assert all(call[1]["timeout"] == 1.25 for call in cleanup_calls)
    assert all(call[1]["check"] is False for call in cleanup_calls)

    assert executor.cancel_all().requested == 0
    assert len(cleanup_calls) == 2


def test_stage_runner_reports_partial_and_preserves_command_order(
    tmp_path: Path,
) -> None:
    events: list[ExecutionEvent] = []
    commands = (
        _spec(tmp_path, (sys.executable, "-c", "pass"), name="success"),
        _spec(tmp_path, (sys.executable, "-c", "raise SystemExit(7)"), name="failure"),
    )
    runner = StageRunner(LocalCommandExecutor(), event_sink=events.append)

    outcome = runner.run("prediction", commands)

    assert outcome.status == "partial"
    assert [item.command.name for item in outcome.commands] == ["success", "failure"]
    assert [item.returncode for item in outcome.commands] == [0, 7]
    assert events[0].kind == "stage_started"
    assert events[-1].kind == "stage_finished"
    assert events[-1].status == "partial"
