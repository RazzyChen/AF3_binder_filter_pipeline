"""Command line interface for Aerith."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from af3_binder_filter import __version__
from af3_binder_filter.af3_json import write_complex_inputs, write_target_input
from af3_binder_filter.af3_runner import (
    pending_input_jsons,
    prepare_shard_dirs,
    run_prepared_shards,
)
from af3_binder_filter.aggregate import aggregate_results
from af3_binder_filter.config import DEFAULT_COMPLEX_JOB_NAME_TEMPLATE, PipelineConfig
from af3_binder_filter.csv_input import CsvInputError, read_binder_csv, read_target_sequence
from af3_binder_filter.esm_score import score_esm_inputs
from af3_binder_filter.esmfold_score import score_esmfold_inputs
from af3_binder_filter.gpu import GPUError, query_gpus, select_free_gpus, shard_jobs
from af3_binder_filter.ipsae_score import score_ipsae_outputs
from af3_binder_filter.pipeline import (
    candidate_target_data_jsons,
    expected_target_data_json,
    run_preflight,
)
from af3_binder_filter.progress import RichProgressReporter
from af3_binder_filter.target_data import TargetDataError, extract_target_features


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


def _config_from_options(
    csv_path: Path,
    work_dir: Path,
    output_dir: Path,
    target_chain: str,
    binder_chain: str,
    gpu_busy_threshold_mib: int,
    job_name_template: str,
) -> PipelineConfig:
    return PipelineConfig(
        csv_path=csv_path,
        work_dir=work_dir,
        output_dir=output_dir,
        target_chain=target_chain,
        binder_chain=binder_chain,
        gpu_busy_threshold_mib=gpu_busy_threshold_mib,
        job_name_template=job_name_template,
    )


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


def _run_af3_input_dir(
    *,
    input_dir: Path,
    config: PipelineConfig,
    shard_name: str,
    dry_run: bool,
    force: bool,
    gpu_ids: str | None = None,
    input_jsons: list[Path] | None = None,
) -> None:
    if not input_dir.exists():
        _fail(f"AF3 input directory does not exist: {input_dir}")
    pending = pending_input_jsons(input_dir, config.output_dir, force=force, input_jsons=input_jsons)
    if not pending:
        console.print(f"No pending AF3 jobs in {input_dir}.")
        return
    try:
        free_gpus = select_free_gpus(
            query_gpus(),
            threshold_mib=config.gpu_busy_threshold_mib,
            allowed_gpu_ids=_parse_gpu_ids(gpu_ids),
        )
        shards = shard_jobs(pending, free_gpus)
    except GPUError as exc:
        _fail(str(exc))
    prepared = prepare_shard_dirs(
        shards,
        shard_root=config.work_dir / "shards" / shard_name,
        output_dir=config.output_dir,
        config=config.af3,
    )
    for shard in prepared:
        console.print(f"GPU {shard.gpu_index}: {len(shard.jobs)} jobs, input_dir={shard.input_dir}")
        console.print(" ".join(shard.command))
    return_code = run_prepared_shards(prepared, dry_run=dry_run)
    if return_code != 0:
        _fail(f"one or more AF3 Docker processes failed; worst return code {return_code}")


CommonCsv = Annotated[Path, typer.Option("--csv", help="Input binder CSV.")]
CommonWorkDir = Annotated[Path, typer.Option("--work-dir", help="Pipeline work directory.")]
CommonOutputDir = Annotated[Path, typer.Option("--output-dir", help="AF3 output directory.")]
CommonLimit = Annotated[int | None, typer.Option("--limit", min=1, help="Limit input jobs.")]
CommonForce = Annotated[bool, typer.Option("--force", help="Overwrite or rerun existing outputs.")]
CommonDryRun = Annotated[bool, typer.Option("--dry-run", help="Print/log commands without running.")]
CommonJobTemplate = Annotated[str, typer.Option("--job-name-template", help="Job name format string.")]
CommonTargetChain = Annotated[str, typer.Option("--target-chain", help="Target chain ID.")]
CommonBinderChain = Annotated[str, typer.Option("--binder-chain", help="Binder chain ID.")]
CommonGpuThreshold = Annotated[
    int, typer.Option("--gpu-busy-threshold-mib", help="GPU memory-use threshold for busy state.")
]


@app.command("check")
def check(
    csv_path: CommonCsv = Path("tests/AF3_pipeline_dev_sample.csv"),
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    target_chain: CommonTargetChain = "A",
    binder_chain: CommonBinderChain = "B",
    gpu_busy_threshold_mib: CommonGpuThreshold = 100,
    job_name_template: CommonJobTemplate = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE,
) -> None:
    config = _config_from_options(
        csv_path,
        work_dir,
        output_dir,
        target_chain,
        binder_chain,
        gpu_busy_threshold_mib,
        job_name_template,
    )
    try:
        report = run_preflight(config)
    except CsvInputError as exc:
        _fail(str(exc))
    for line in report.info:
        console.print(line)
    for line in report.warnings:
        console.print(f"[yellow]warning:[/yellow] {line}")
    if not report.ok:
        for line in report.errors:
            console.print(f"[red]error:[/red] {line}")
        raise typer.Exit(code=1)


@app.command("make-target")
def make_target(
    csv_path: CommonCsv = Path("tests/AF3_pipeline_dev_sample.csv"),
    work_dir: CommonWorkDir = Path("work"),
    force: CommonForce = False,
    target_chain: CommonTargetChain = "A",
    name: Annotated[str, typer.Option("--name", help="Target AF3 job name.")] = "target_A",
    seed: Annotated[int, typer.Option("--seed", help="AF3 model seed.")] = 42,
) -> None:
    try:
        target_sequence = read_target_sequence(csv_path)
        output_path = write_target_input(
            target_sequence=target_sequence,
            output_dir=work_dir / "target_input",
            name=name,
            target_chain=target_chain,
            seed=seed,
            force=force,
        )
    except (CsvInputError, ValueError) as exc:
        _fail(str(exc))
    console.print(f"Wrote target input: {output_path}")


@app.command("run-target")
def run_target(
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    dry_run: CommonDryRun = False,
    force: CommonForce = False,
    gpu_busy_threshold_mib: CommonGpuThreshold = 100,
    gpu_ids: Annotated[
        str | None,
        typer.Option("--gpu-ids", help="Comma-separated physical GPU indexes to consider."),
    ] = None,
) -> None:
    config = PipelineConfig(
        work_dir=work_dir,
        output_dir=output_dir,
        gpu_busy_threshold_mib=gpu_busy_threshold_mib,
    )
    _run_af3_input_dir(
        input_dir=config.target_input_dir,
        config=config,
        shard_name="target",
        dry_run=dry_run,
        force=force,
        gpu_ids=gpu_ids,
    )


@app.command("build-complex")
def build_complex(
    csv_path: CommonCsv = Path("tests/AF3_pipeline_dev_sample.csv"),
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    limit: CommonLimit = None,
    force: CommonForce = False,
    target_data_json: Annotated[
        Path | None,
        typer.Option(
            "--target-data-json",
            help="Target AF3 *_data.json used to externalize chain A MSA/template features.",
        ),
    ] = None,
    allow_empty_target_features: Annotated[
        bool,
        typer.Option(
            "--allow-empty-target-features",
            help="Allow complex JSON generation without target MSA/template data.",
        ),
    ] = False,
    target_chain: CommonTargetChain = "A",
    binder_chain: CommonBinderChain = "B",
    gpu_busy_threshold_mib: CommonGpuThreshold = 100,
    job_name_template: CommonJobTemplate = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE,
) -> None:
    config = _config_from_options(
        csv_path,
        work_dir,
        output_dir,
        target_chain,
        binder_chain,
        gpu_busy_threshold_mib,
        job_name_template,
    )
    try:
        rows = read_binder_csv(config.csv_path, limit=limit)
        if target_data_json is not None:
            target_features = extract_target_features(
                target_data_json,
                config.complex_input_dir,
                chain_id=config.target_chain,
                expected_sequence=rows[0].target_seq if rows else None,
                force=force,
            )
        elif allow_empty_target_features:
            target_features = None
            console.print("[yellow]warning:[/yellow] building complex inputs without target MSA/templates")
        else:
            _fail("--target-data-json is required unless --allow-empty-target-features is set")
        written = write_complex_inputs(
            rows,
            config.complex_input_dir,
            target_features=target_features,
            job_name_template=config.job_name_template,
            target_chain=config.target_chain,
            binder_chain=config.binder_chain,
            force=force,
            clean_stale=True,
        )
    except (CsvInputError, ValueError) as exc:
        _fail(str(exc))
    console.print(f"Wrote {len(written)} complex input JSON files to {config.complex_input_dir}")


@app.command("run-complex")
def run_complex(
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    dry_run: CommonDryRun = False,
    force: CommonForce = False,
    gpu_busy_threshold_mib: CommonGpuThreshold = 100,
    gpu_ids: Annotated[
        str | None,
        typer.Option("--gpu-ids", help="Comma-separated physical GPU indexes to consider."),
    ] = None,
) -> None:
    config = PipelineConfig(
        work_dir=work_dir,
        output_dir=output_dir,
        gpu_busy_threshold_mib=gpu_busy_threshold_mib,
    )
    _run_af3_input_dir(
        input_dir=config.complex_input_dir,
        config=config,
        shard_name="complex",
        dry_run=dry_run,
        force=force,
        gpu_ids=gpu_ids,
    )


@app.command("score-esm")
def score_esm(
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    binder_chain: CommonBinderChain = "B",
    dry_run: CommonDryRun = False,
    force: CommonForce = False,
    use_ray: Annotated[bool, typer.Option("--ray/--no-ray", help="Use Ray GPU tasks.")] = True,
) -> None:
    config = PipelineConfig(work_dir=work_dir, output_dir=output_dir, binder_chain=binder_chain)
    input_dir = config.complex_input_dir
    if not input_dir.exists():
        _fail(f"complex input directory does not exist: {input_dir}")
    rows = score_esm_inputs(
        input_dir=input_dir,
        af_output_dir=config.output_dir,
        score_dir=config.score_dir,
        chain_id=config.binder_chain,
        config=config.esm,
        dry_run=dry_run,
        force=force,
        use_ray=use_ray,
    )
    success = sum(1 for row in rows if row.get("esm_score_status") == "success")
    console.print(f"ESM scored {len(rows)} jobs ({success} success)")


@app.command("score-esmfold")
def score_esmfold(
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    binder_chain: CommonBinderChain = "B",
    dry_run: CommonDryRun = False,
    force: CommonForce = False,
    gpu_busy_threshold_mib: CommonGpuThreshold = 100,
    gpu_ids: Annotated[
        str | None,
        typer.Option("--gpu-ids", help="Comma-separated physical GPU indexes to consider."),
    ] = None,
) -> None:
    config = PipelineConfig(
        work_dir=work_dir,
        output_dir=output_dir,
        binder_chain=binder_chain,
        gpu_busy_threshold_mib=gpu_busy_threshold_mib,
    )
    input_dir = config.complex_input_dir
    if not input_dir.exists():
        _fail(f"complex input directory does not exist: {input_dir}")
    rows = score_esmfold_inputs(
        input_dir=input_dir,
        score_dir=config.score_dir,
        chain_id=config.binder_chain,
        config=config.esmfold,
        dry_run=dry_run,
        force=force,
        gpu_busy_threshold_mib=config.gpu_busy_threshold_mib,
        gpu_ids=_parse_gpu_ids(gpu_ids),
    )
    success = sum(1 for row in rows if row.get("esmfold_status") == "success")
    console.print(f"ESMFold scored {len(rows)} jobs ({success} success)")


@app.command("score-ipsae")
def score_ipsae(
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    pae_cutoff: Annotated[float, typer.Option("--pae-cutoff", help="PAE cutoff.")] = 10.0,
    dist_cutoff: Annotated[float, typer.Option("--dist-cutoff", help="Distance cutoff.")] = 15.0,
    target_chain: CommonTargetChain = "A",
    binder_chain: CommonBinderChain = "B",
    use_ray: Annotated[bool, typer.Option("--ray/--no-ray", help="Use Ray CPU tasks.")] = True,
) -> None:
    config = PipelineConfig(
        work_dir=work_dir,
        output_dir=output_dir,
        target_chain=target_chain,
        binder_chain=binder_chain,
    )
    input_dir = config.complex_input_dir
    if not input_dir.exists():
        _fail(f"complex input directory does not exist: {input_dir}")
    try:
        rows = score_ipsae_outputs(
            input_dir=input_dir,
            af_output_dir=config.output_dir,
            score_dir=config.score_dir,
            target_chain=config.target_chain,
            binder_chain=config.binder_chain,
            pae_cutoff=pae_cutoff,
            dist_cutoff=dist_cutoff,
            use_ray=use_ray,
        )
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
    success = sum(1 for row in rows if row.get("ipsae_score_status") == "success")
    console.print(f"ipSAE scored {len(rows)} jobs ({success} success)")


@app.command("aggregate")
def aggregate(
    csv_path: CommonCsv = Path("tests/AF3_pipeline_dev_sample.csv"),
    output_dir: CommonOutputDir = Path("af_output"),
    results_dir: Annotated[Path, typer.Option("--results-dir", help="Directory for result CSVs.")] = Path(
        "."
    ),
    score_dir: Annotated[
        Path,
        typer.Option("--score-dir", help="Directory containing ESM/ESMFold/ipSAE score summary CSVs."),
    ] = Path("work/scores"),
    job_name_template: CommonJobTemplate = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE,
    target_chain: CommonTargetChain = "A",
    binder_chain: CommonBinderChain = "B",
    sasa_point_number: Annotated[
        int,
        typer.Option("--sasa-point-number", min=1, help="Sphere point count for biotite SASA."),
    ] = 1000,
) -> None:
    try:
        rows = aggregate_results(
            csv_path=csv_path,
            af_output_dir=output_dir,
            results_dir=results_dir,
            score_dir=score_dir,
            job_name_template=job_name_template,
            target_chain=target_chain,
            binder_chain=binder_chain,
            sasa_point_number=sasa_point_number,
        )
    except Exception as exc:  # noqa: BLE001 - CLI should print a clear error
        _fail(str(exc))
    success = sum(1 for row in rows if row.get("job_status") == "success")
    console.print(f"Aggregated {len(rows)} jobs ({success} success) into {results_dir}")


@app.command("legacy-pipeline", hidden=True)
def pipeline(
    csv_path: CommonCsv = Path("tests/AF3_pipeline_dev_sample.csv"),
    work_dir: CommonWorkDir = Path("work"),
    output_dir: CommonOutputDir = Path("af_output"),
    limit: CommonLimit = None,
    dry_run: CommonDryRun = False,
    force: CommonForce = False,
    target_data_json: Annotated[
        Path | None,
        typer.Option("--target-data-json", help="Existing target AF3 *_data.json to reuse."),
    ] = None,
    target_name: Annotated[str, typer.Option("--target-name", help="Target AF3 job name.")] = "target_A",
    job_name_template: CommonJobTemplate = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE,
    target_chain: CommonTargetChain = "A",
    binder_chain: CommonBinderChain = "B",
    gpu_busy_threshold_mib: CommonGpuThreshold = 100,
    gpu_ids: Annotated[
        str | None,
        typer.Option("--gpu-ids", help="Comma-separated physical GPU indexes to consider."),
    ] = None,
    use_ray: Annotated[bool, typer.Option("--ray/--no-ray", help="Use Ray for scoring.")] = True,
    sasa_point_number: Annotated[
        int,
        typer.Option("--sasa-point-number", min=1, help="Sphere point count for biotite SASA."),
    ] = 1000,
) -> None:
    config = PipelineConfig(
        csv_path=csv_path,
        work_dir=work_dir,
        output_dir=output_dir,
        target_chain=target_chain,
        binder_chain=binder_chain,
        gpu_busy_threshold_mib=gpu_busy_threshold_mib,
        job_name_template=job_name_template,
    )

    report = run_preflight(config)
    for line in report.info:
        console.print(line)
    if not report.ok:
        for line in report.errors:
            console.print(f"[red]error:[/red] {line}")
        raise typer.Exit(code=1)

    try:
        target_sequence = read_target_sequence(config.csv_path)
        target_input = write_target_input(
            target_sequence=target_sequence,
            output_dir=config.target_input_dir,
            name=target_name,
            target_chain=config.target_chain,
            force=force,
        )
        console.print(f"Wrote target input: {target_input}")
    except (CsvInputError, ValueError) as exc:
        _fail(str(exc))

    active_target_data = target_data_json
    if active_target_data is None:
        _run_af3_input_dir(
            input_dir=config.target_input_dir,
            config=config,
            shard_name="target",
            dry_run=dry_run,
            force=force,
            gpu_ids=gpu_ids,
        )
        active_target_data = expected_target_data_json(config, target_name=target_name)

    if dry_run and not active_target_data.exists():
        console.print(
            "[yellow]dry-run:[/yellow] target data JSON is not available after target AF3 dry-run; "
            "stopping before build-complex."
        )
        return
    if not active_target_data.exists():
        candidates = ", ".join(str(path) for path in candidate_target_data_jsons(config, target_name=target_name))
        _fail(f"target data JSON does not exist after run-target; checked: {candidates}")

    try:
        rows = read_binder_csv(config.csv_path, limit=limit)
        target_features = extract_target_features(
            active_target_data,
            config.complex_input_dir,
            chain_id=config.target_chain,
            expected_sequence=target_sequence,
            force=force,
        )
        written = write_complex_inputs(
            rows,
            config.complex_input_dir,
            target_features=target_features,
            job_name_template=config.job_name_template,
            target_chain=config.target_chain,
            binder_chain=config.binder_chain,
            force=force,
            clean_stale=True,
        )
        console.print(f"Wrote {len(written)} complex input JSON files to {config.complex_input_dir}")
    except (CsvInputError, TargetDataError, ValueError) as exc:
        _fail(str(exc))

    _run_af3_input_dir(
        input_dir=config.complex_input_dir,
        config=config,
        shard_name="complex",
        dry_run=dry_run,
        force=force,
        gpu_ids=gpu_ids,
        input_jsons=written,
    )
    if dry_run:
        console.print("[yellow]dry-run:[/yellow] stopping before ESM/ESMFold/ipSAE/aggregate.")
        return

    score_esm_inputs(
        input_dir=config.complex_input_dir,
        input_jsons=written,
        af_output_dir=config.output_dir,
        score_dir=config.score_dir,
        chain_id=config.binder_chain,
        config=config.esm,
        force=force,
        use_ray=use_ray,
    )
    score_ipsae_outputs(
        input_dir=config.complex_input_dir,
        input_jsons=written,
        af_output_dir=config.output_dir,
        score_dir=config.score_dir,
        target_chain=config.target_chain,
        binder_chain=config.binder_chain,
        use_ray=use_ray,
    )
    score_esmfold_inputs(
        input_dir=config.complex_input_dir,
        input_jsons=written,
        score_dir=config.score_dir,
        chain_id=config.binder_chain,
        config=config.esmfold,
        force=force,
        gpu_busy_threshold_mib=config.gpu_busy_threshold_mib,
        gpu_ids=_parse_gpu_ids(gpu_ids),
    )
    rows = aggregate_results(
        csv_path=config.csv_path,
        af_output_dir=config.output_dir,
        score_dir=config.score_dir,
        job_name_template=config.job_name_template,
        target_chain=config.target_chain,
        binder_chain=config.binder_chain,
        sasa_point_number=sasa_point_number,
    )
    console.print(f"Pipeline complete: aggregated {len(rows)} jobs")


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


@app.command("build-backend-image")
def build_backend_image(
    config_path: HydraConfigPath = Path("config.yaml"),
    backend: HydraBackend = None,
    override: HydraOverrides = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the pinned local-source build command."),
    ] = False,
) -> None:
    """Build a Protenix or OpenDDE image from the configured local source."""

    from af3_binder_filter.backends import BackendError, build_backend_image_command
    from af3_binder_filter.config import ConfigError

    try:
        config, _resolved = _compose_cli_config(
            config_path,
            backend=backend,
            overrides=override,
        )
        command = build_backend_image_command(config)
        if config.backend.source_commit:
            source_dir = str(Path(config.backend.source_dir or "").expanduser().resolve())
            head = subprocess.run(
                ["git", "-C", source_dir, "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            actual = head.stdout.strip() if head.returncode == 0 else ""
            if not actual.startswith(config.backend.source_commit):
                raise BackendError(
                    "backend source commit mismatch: "
                    f"expected {config.backend.source_commit}, found {actual or 'unavailable'}"
                )
        if dry_run:
            console.print(" ".join(command))
            return
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise BackendError(
                f"backend image build failed with return code {completed.returncode}"
            )
    except (BackendError, ConfigError, OSError, subprocess.SubprocessError) as exc:
        _fail(str(exc))
    console.print(f"Built backend image: {config.backend.image}")


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
        )
        failed = run_clustering_only(context)
    except Exception as exc:  # noqa: BLE001
        _fail(str(exc))
    if failed and not context.config.project.allow_partial:
        _fail(f"Foldseek clustering failed; partial singleton clusters are in {context.results_dir}")
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
