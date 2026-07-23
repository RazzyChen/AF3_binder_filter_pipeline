from __future__ import annotations

import os
import signal
import subprocess
import sys
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

    outcome = LocalCommandExecutor(event_sink=events.append).run(command)

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

    outcome = LocalCommandExecutor(
        termination_grace_seconds=0.05,
        event_sink=events.append,
    ).run(command)

    assert outcome.timed_out
    assert outcome.failed
    assert outcome.status == "timeout"
    assert outcome.returncode is not None and outcome.returncode != 0
    assert outcome.signal in {signal.SIGTERM, signal.SIGKILL}
    assert "exceeded timeout" in command.stderr_path.read_text()
    assert events[-1].kind == "command_timed_out"


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

    with pytest.raises(KeyboardInterrupt):
        LocalCommandExecutor(popen_factory=popen).run(command)  # type: ignore[arg-type]

    assert sent_signals == [(424242, signal.SIGTERM)]
    assert "process group terminated" in command.stderr_path.read_text()
    command.stdout_path.rename(tmp_path / "closed.stdout.log")
    command.stderr_path.rename(tmp_path / "closed.stderr.log")


def test_dry_run_writes_only_command_record_and_never_starts_process(tmp_path: Path) -> None:
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


def test_stage_runner_reports_partial_and_preserves_command_order(tmp_path: Path) -> None:
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
