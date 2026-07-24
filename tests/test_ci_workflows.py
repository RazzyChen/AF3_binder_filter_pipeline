from __future__ import annotations

from pathlib import Path

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"


def _workflow(name: str) -> str:
    return (WORKFLOWS / name).read_text()


def test_cpu_ci_uses_locked_dependencies_and_real_python_matrix() -> None:
    workflow = _workflow("ci.yml")

    assert "uv sync --locked --dev" in workflow
    assert 'UV_PYTHON: "3.12"' in workflow
    assert "UV_PYTHON: ${{ matrix.python-version }}" in workflow
    assert 'pytest -m "not integration"' in workflow
    assert "ruff check ." in workflow
    assert "deptry --config pyproject.toml ." in workflow


def test_self_hosted_runtime_build_workflow_is_cache_backed_and_not_pr_triggered() -> None:
    workflow = _workflow("docker-build.yml")

    assert "pull_request:" not in workflow
    assert "runs-on: [self-hosted, linux, x64, aerith-build]" in workflow
    assert "AERITH_RUNTIME_SOURCE_BUNDLE" in workflow
    assert "snapshot_runtime_sources.py verify" in workflow
    assert "--cache-dir" in workflow
    assert "flock -n 9" in workflow
    assert "aerith/fold-runtime:ci-candidate" in workflow


def test_gpu_smoke_workflow_requires_exclusive_host_and_integration_contract() -> None:
    workflow = _workflow("gpu-smoke.yml")

    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "runs-on: [self-hosted, linux, x64, aerith-gpu]" in workflow
    assert "nvidia-smi --query-compute-apps" in workflow
    assert "flock -n 9" in workflow
    assert "--network none" in workflow
    assert "pytest -q -m integration tests/test_gpu_smoke_contract.py" in workflow
    assert "ci-last-known-good" in workflow
