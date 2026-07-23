"""Command line interface for Aerith."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from af3_binder_filter import __version__
from af3_binder_filter.csv_input import CsvInputError
from af3_binder_filter.progress import RichProgressReporter


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Aerith: AlphaFold 3 binder candidate orchestration pipeline.",
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def _root_callback(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, help="Show version and exit."),
    ] = False,
) -> None:
    """Aerith."""


def main() -> None:
    """Run the Aerith CLI."""
    app()


def _fail(message: str) -> None:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code=1)


def _parse_gpu_ids(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    try:
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise typer.BadParameter("--gpu-ids must be a comma-separated list of GPU indexes") from exc


# Hydra-native command surface ----------------------------------------------------

config_app = typer.Typer(
    name="config",
    no_args_is_help=True,
    help="Generate, validate, inspect, and diagnose Hydra configuration.",
)
app.add_typer(config_app, name="config")


@config_app.command("create")
def config_create(
    output: Annotated[Path, typer.Option("--output", help="Minimal YAML path to create.")] = Path(
        "config.yaml"
    ),
    project_root: Annotated[
        Path,
        typer.Option(
            "--project-root",
            help="Production screen root; work, outputs, and results are placed below it.",
        ),
    ] = Path("."),
    csv_path: Annotated[
        Path | None,
        typer.Option(
            "--csv",
            help="Binder CSV; defaults to <project-root>/input/screen.csv.",
        ),
    ] = None,
    secondary_backend: Annotated[
        str,
        typer.Option(
            "--secondary-backend",
            help="Cross-validation backend: none, protenix, or opendde.",
        ),
    ] = "opendde",
    gpu_ids: Annotated[
        str | None,
        typer.Option(
            "--gpu-ids",
            help="Comma-separated GPU allow-list; omit to use all available GPUs.",
        ),
    ] = None,
    epitope_residues: Annotated[
        str | None,
        typer.Option(
            "--epitope-residues",
            help="Optional target positions, for example 25-35,42,57.",
        ),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Replace an existing YAML.")] = False,
) -> None:
    """Create a non-interactive minimal config for a production screen."""

    from af3_binder_filter.config_tools import write_minimal_production_config

    if output.exists() and not force:
        _fail(f"configuration already exists: {output}; use --force to replace it")
    try:
        selected_gpus = _parse_gpu_ids(gpu_ids) or []
        write_minimal_production_config(
            output,
            project_root=project_root,
            csv_path=csv_path,
            secondary_backend=secondary_backend,
            gpu_ids=selected_gpus,
            epitope_residues=epitope_residues,
        )
    except ValueError as exc:
        _fail(str(exc))

    root = project_root.expanduser().resolve()
    resolved_csv = csv_path.expanduser().resolve() if csv_path else root / "input" / "screen.csv"
    gpu_summary = ",".join(map(str, selected_gpus)) if selected_gpus else "all available GPUs"
    console.print(f"Wrote minimal production configuration: {output}")
    console.print(f"Backends: alphafold3 + {secondary_backend}")
    console.print(f"Input CSV: {resolved_csv}")
    console.print(f"GPU selection: {gpu_summary}")


@config_app.command("init")
def config_init(
    output: Annotated[Path, typer.Option("--output", help="YAML path to create.")] = Path(
        "config.yaml"
    ),
    backend: Annotated[
        str,
        typer.Option("--backend", help="Primary backend; must be alphafold3."),
    ] = "alphafold3",
    secondary_backend: Annotated[
        str,
        typer.Option(
            "--secondary-backend",
            help="Initial cross-validation backend: none, protenix, or opendde.",
        ),
    ] = "none",
    csv_path: Annotated[
        str,
        typer.Option("--csv", help="Initial project CSV path."),
    ] = "all_seq_PD1_May12.csv",
    force: Annotated[bool, typer.Option("--force", help="Replace an existing YAML.")] = False,
) -> None:
    from af3_binder_filter.config_tools import detect_environment, write_initial_config

    if backend != "alphafold3":
        raise typer.BadParameter("--backend must be alphafold3")
    if secondary_backend not in {"none", "protenix", "opendde"}:
        raise typer.BadParameter(
            "--secondary-backend must be none, protenix, or opendde"
        )
    if output.exists() and not force:
        _fail(f"configuration already exists: {output}; use --force to replace it")
    detection = detect_environment()
    console.print(f"Docker: {detection.docker or 'not found'}")
    console.print(
        "GPUs: " + (",".join(map(str, detection.gpu_indexes)) if detection.gpu_indexes else "not found")
    )
    console.print(f"AF3 database: {detection.database_dir or 'not found'}")
    console.print("Foldseek: in-image GPU release 10-941cd33")
    console.print("MMseqs2: in-image GPU release 18-8cc5c")
    console.print(f"Rosetta: {detection.rosetta_binary or 'not found'}")
    console.print(f"Protenix: {detection.protenix_source or 'not found'}")
    console.print(
        f"OpenDDE: {detection.opendde_source or 'not found'}"
        + (f" ({detection.opendde_commit[:12]})" if detection.opendde_commit else "")
    )
    try:
        write_initial_config(
            output,
            detection,
            csv_path=csv_path,
            backend=backend,
            secondary_backend=secondary_backend,
        )
    except ValueError as exc:
        _fail(str(exc))
    console.print(f"Wrote Hydra configuration: {output}")


def _compose_cli_config(
    config_path: Path,
    *,
    backend: str | None = None,
    secondary_backend: str | None = None,
    overrides: list[str] | None = None,
):
    from af3_binder_filter.config import compose_hydra_config

    return compose_hydra_config(
        config_path,
        backend=backend,
        secondary_backend=secondary_backend,
        overrides=overrides or (),
    )


@config_app.command("validate")
def config_validate(
    config_path: Annotated[Path, typer.Option("--config", help="Hydra YAML.")] = Path(
        "config.yaml"
    ),
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Override the backend config group."),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", help="Repeatable Hydra override."),
    ] = None,
) -> None:
    from af3_binder_filter.config import ConfigError, validate_hydra_config
    from af3_binder_filter.jobs import build_job_plan

    try:
        config, _resolved = _compose_cli_config(
            config_path,
            backend=backend,
            overrides=override,
        )
        report = validate_hydra_config(config)
        plan = build_job_plan(config) if report.ok else None
    except (ConfigError, CsvInputError, ValueError) as exc:
        _fail(str(exc))
    for line in report.info:
        console.print(line)
    for line in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {line}")
    if plan is not None:
        console.print(
            f"Job plan: {len(plan.jobs)}/{plan.total_csv_jobs} jobs; "
            f"target length {len(plan.target_sequence)}"
        )
    if report.errors:
        for line in report.errors:
            console.print(f"[red]error:[/red] {line}")
        raise typer.Exit(code=1)


@config_app.command("doctor")
def config_doctor(
    config_path: Annotated[Path, typer.Option("--config", help="Hydra YAML.")] = Path(
        "config.yaml"
    ),
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Override the backend config group."),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", help="Repeatable Hydra override."),
    ] = None,
) -> None:
    from af3_binder_filter.config import ConfigError
    from af3_binder_filter.config_tools import doctor_config

    try:
        config, _resolved = _compose_cli_config(
            config_path,
            backend=backend,
            overrides=override,
        )
        report = doctor_config(config)
    except (ConfigError, ValueError) as exc:
        _fail(str(exc))
    for line in report.info:
        console.print(line)
    for line in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {line}")
    if report.errors:
        for line in report.errors:
            console.print(f"[red]error:[/red] {line}")
        raise typer.Exit(code=1)


@config_app.command("show")
def config_show(
    config_path: Annotated[Path, typer.Option("--config", help="Hydra YAML.")] = Path(
        "config.yaml"
    ),
    backend: Annotated[
        str | None,
        typer.Option("--backend", help="Override the backend config group."),
    ] = None,
    override: Annotated[
        list[str] | None,
        typer.Option("--override", help="Repeatable Hydra override."),
    ] = None,
) -> None:
    from omegaconf import OmegaConf

    from af3_binder_filter.config import ConfigError

    try:
        _config, resolved = _compose_cli_config(
            config_path,
            backend=backend,
            overrides=override,
        )
    except ConfigError as exc:
        _fail(str(exc))
    console.print(OmegaConf.to_yaml(resolved, resolve=True, sort_keys=False), markup=False)


HydraConfigPath = Annotated[Path, typer.Option("--config", help="Hydra YAML configuration.")]
HydraBackend = Annotated[
    str | None,
    typer.Option("--backend", help="Primary backend override (must remain alphafold3)."),
]
HydraSecondaryBackend = Annotated[
    str | None,
    typer.Option(
        "--secondary-backend",
        help="Cross-validation backend: none, protenix, or opendde.",
    ),
]
HydraOverrides = Annotated[
    list[str] | None,
    typer.Option("--override", help="Repeatable Hydra override, e.g. interface.distance=4.5."),
]


@app.command("build-runtime-image")
def build_runtime_image(
    config_path: HydraConfigPath = Path("config.yaml"),
    override: HydraOverrides = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Validate and print the unified image build command."),
    ] = False,
) -> None:
    """Build the isolated AF3/Protenix/OpenDDE/ESM runtime image."""

    from af3_binder_filter.backends import (
        BackendError,
        build_runtime_image_command,
        prepare_runtime_build_contexts,
    )
    from af3_binder_filter.config import ConfigError

    try:
        config, _resolved = _compose_cli_config(config_path, overrides=override)
        if dry_run:
            command = build_runtime_image_command(config)
            console.print(" ".join(command))
            return
        context_root = prepare_runtime_build_contexts(config)
        command = build_runtime_image_command(config, context_root=context_root)
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise BackendError(
                f"runtime image build failed with return code {completed.returncode}"
            )
    except (BackendError, ConfigError, OSError, subprocess.SubprocessError) as exc:
        _fail(str(exc))
    console.print(f"Built unified runtime image: {config.backend.image}")


@app.command("prepare-features")
def prepare_features_command(
    config_path: HydraConfigPath = Path("config.yaml"),
    backend: HydraBackend = None,
    override: HydraOverrides = None,
    dry_run: Annotated[
        bool | None,
        typer.Option("--dry-run/--no-dry-run", help="Override runtime.dry_run."),
    ] = None,
) -> None:
    from af3_binder_filter.workflow import (
        create_run_context,
        run_prepare_features_only,
    )

    try:
        context = create_run_context(
            config_path,
            backend=backend,
            overrides=override or (),
            dry_run=dry_run,
        )
        preparation = run_prepare_features_only(context)
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
    if preparation.bundle:
        console.print(
            f"Target features ready: {preparation.bundle.cache_dir}"
            + (" (cache hit)" if preparation.reused else "")
        )
    elif preparation.command:
        console.print("Dry-run command: " + " ".join(preparation.command))


@app.command("analyze-interface")
def analyze_interface_command(
    config_path: HydraConfigPath = Path("config.yaml"),
    backend: HydraBackend = None,
    override: HydraOverrides = None,
) -> None:
    from af3_binder_filter.workflow import create_run_context, run_interface_only

    try:
        context = create_run_context(
            config_path,
            backend=backend,
            overrides=override or (),
        )
        rows = run_interface_only(context)
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
    success = sum(row.get("interface_status") == "success" for row in rows)
    console.print(f"Interface analysis: {success}/{len(rows)} geometry successes in {context.results_dir}")


@app.command("cluster")
def cluster_command(
    config_path: HydraConfigPath = Path("config.yaml"),
    backend: HydraBackend = None,
    override: HydraOverrides = None,
    dry_run: Annotated[
        bool | None,
        typer.Option("--dry-run/--no-dry-run", help="Override runtime.dry_run."),
    ] = None,
) -> None:
    from af3_binder_filter.workflow import create_run_context, run_clustering_only

    try:
        context = create_run_context(
            config_path,
            backend=backend,
            overrides=override or (),
            dry_run=dry_run,
            initialize_run=False,
        )
        failed = run_clustering_only(context)
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
    if failed and not context.config.project.allow_partial:
        _fail(
            "Foldseek clustering failed; partial clustering audit outputs are in "
            f"{context.results_dir}"
        )
    console.print(f"Cluster outputs written to {context.results_dir}")


@app.command("pipeline")
def hydra_pipeline(
    config_path: HydraConfigPath = Path("config.yaml"),
    backend: HydraBackend = None,
    secondary_backend: HydraSecondaryBackend = None,
    override: HydraOverrides = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", min=1, help="Limit the immutable job plan."),
    ] = None,
    dry_run: Annotated[
        bool | None,
        typer.Option("--dry-run/--no-dry-run", help="Override runtime.dry_run."),
    ] = None,
) -> None:
    from af3_binder_filter.workflow import (
        PipelineExecutionError,
        create_run_context,
        run_pipeline,
    )

    reporter = RichProgressReporter(console)
    try:
        context = create_run_context(
            config_path,
            backend=backend,
            secondary_backend=secondary_backend,
            overrides=override or (),
            limit=limit,
            dry_run=dry_run,
        )
        run_pipeline(context, reporter=reporter)
    except PipelineExecutionError as exc:
        _fail(str(exc))
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
    finally:
        reporter.close()
