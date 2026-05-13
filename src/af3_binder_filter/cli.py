"""Command line interface for the AF3 binder filter pipeline."""

from __future__ import annotations

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
from af3_binder_filter.config import PipelineConfig
from af3_binder_filter.csv_input import CsvInputError, read_binder_csv, read_target_sequence
from af3_binder_filter.esm_score import score_esm_inputs
from af3_binder_filter.gpu import GPUError, query_gpus, select_free_gpus, shard_jobs
from af3_binder_filter.ipsae_score import score_ipsae_outputs
from af3_binder_filter.target_data import TargetDataError, extract_target_features


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="AlphaFold 3 binder-target filtering pipeline.",
)
console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=version_callback, help="Show version and exit."),
    ] = False,
) -> None:
    """AF3 binder filter."""


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


def _not_implemented(command: str) -> None:
    console.print(f"[yellow]{command}[/yellow] is scaffolded but not implemented yet.")
    raise typer.Exit(code=2)


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
    job_name_template: CommonJobTemplate = "sample_{sample_no}_{run_name}",
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
        rows = read_binder_csv(config.csv_path)
    except CsvInputError as exc:
        _fail(str(exc))
    console.print(f"CSV OK: {config.csv_path} ({len(rows)} jobs)")


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
def run_target() -> None:
    _not_implemented("run-target")


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
    job_name_template: CommonJobTemplate = "sample_{sample_no}_{run_name}",
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
    input_dir = config.complex_input_dir
    if not input_dir.exists():
        _fail(f"complex input directory does not exist: {input_dir}")

    pending = pending_input_jsons(input_dir, config.output_dir, force=force)
    if not pending:
        console.print("No pending complex jobs.")
        return

    try:
        gpus = query_gpus()
        free_gpus = select_free_gpus(
            gpus,
            threshold_mib=config.gpu_busy_threshold_mib,
            allowed_gpu_ids=_parse_gpu_ids(gpu_ids),
        )
        shards = shard_jobs(pending, free_gpus)
    except GPUError as exc:
        _fail(str(exc))

    prepared = prepare_shard_dirs(
        shards,
        shard_root=config.work_dir / "shards" / "complex",
        output_dir=config.output_dir,
        config=config.af3,
    )
    for shard in prepared:
        console.print(
            f"GPU {shard.gpu_index}: {len(shard.jobs)} jobs, input_dir={shard.input_dir}"
        )
        console.print(" ".join(shard.command))

    return_code = run_prepared_shards(prepared, dry_run=dry_run)
    if return_code != 0:
        _fail(f"one or more AF3 Docker processes failed; worst return code {return_code}")


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
        typer.Option("--score-dir", help="Directory containing ESM/ipSAE score summary CSVs."),
    ] = Path("work/scores"),
    job_name_template: CommonJobTemplate = "sample_{sample_no}_{run_name}",
    target_chain: CommonTargetChain = "A",
    binder_chain: CommonBinderChain = "B",
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
        )
    except Exception as exc:  # noqa: BLE001 - CLI should print a clear error
        _fail(str(exc))
    success = sum(1 for row in rows if row.get("job_status") == "success")
    console.print(f"Aggregated {len(rows)} jobs ({success} success) into {results_dir}")


@app.command("pipeline")
def pipeline(dry_run: CommonDryRun = False, force: CommonForce = False) -> None:
    _ = (dry_run, force)
    _not_implemented("pipeline")
