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
    assert "scripts/runtime_sources.py validate" in workflow


def test_docker_contract_is_hosted_static_validation_only() -> None:
    workflow = _workflow("docker-contract.yml")

    assert "pull_request:" in workflow
    assert "push:" in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "scripts/runtime_sources.py metadata" in workflow
    assert workflow.count("docker build --check") == 2
    assert "--push" not in workflow
    assert "self-hosted" not in workflow


def test_runtime_build_is_manual_remote_private_and_component_staged() -> None:
    workflow = _workflow("runtime-build.yml")

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    assert "runs-on: ubuntu-24.04" in workflow
    assert "--driver remote" in workflow
    assert "cacert=" in workflow
    assert "AERITH_BUILDKIT_ENDPOINT" in workflow
    assert "$registry/aerith-runtime-uv:build-" in workflow
    assert "$registry/aerith-runtime-conda:build-" in workflow
    assert "uv-component" in workflow
    assert "conda-component" in workflow
    assert "runtime-base" in workflow
    assert "pids=()" in workflow
    assert "Dockerfile.assemble" in workflow
    assert "UV_COMPONENT_IMAGE_DIGEST" in workflow
    assert "CONDA_COMPONENT_IMAGE_DIGEST" in workflow
    assert "candidate-${GITHUB_RUN_ID}" in workflow
    assert "aerith-fold-runtime:latest" not in workflow


def test_gpu_smoke_requires_exclusive_host_and_promotes_tested_digest() -> None:
    workflow = _workflow("gpu-smoke.yml")

    assert "pull_request:" not in workflow
    assert "push:" not in workflow
    assert "schedule:" not in workflow
    assert "runs-on: [self-hosted, linux, x64, aerith-gpu]" in workflow
    assert "nvidia-smi --query-compute-apps" in workflow
    assert "flock -n 9" in workflow
    assert "--network none" in workflow
    assert "scripts/verify_runtime_image.py" in workflow
    assert "pytest -q -m integration tests/test_gpu_smoke_contract.py" in workflow
    assert "environment: runtime-release" in workflow
    assert workflow.index("scripts/verify_runtime_image.py") < workflow.index(
        "environment: runtime-release"
    )
    assert "$SOURCE_REPOSITORY@$SOURCE_DIGEST" in workflow
    assert '--tag "$target:stable"' in workflow
    assert "aerith-fold-runtime:latest" not in workflow
