"""Command line interface for the AF3 binder filter pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from af3_binder_filter import __version__
from af3_binder_filter.config import PipelineConfig


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
    _config_from_options(
        csv_path,
        work_dir,
        output_dir,
        target_chain,
        binder_chain,
        gpu_busy_threshold_mib,
        job_name_template,
    )
    _not_implemented("check")


@app.command("make-target")
def make_target() -> None:
    _not_implemented("make-target")


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
    target_chain: CommonTargetChain = "A",
    binder_chain: CommonBinderChain = "B",
    gpu_busy_threshold_mib: CommonGpuThreshold = 100,
    job_name_template: CommonJobTemplate = "sample_{sample_no}_{run_name}",
) -> None:
    _ = (limit, force)
    _config_from_options(
        csv_path,
        work_dir,
        output_dir,
        target_chain,
        binder_chain,
        gpu_busy_threshold_mib,
        job_name_template,
    )
    _not_implemented("build-complex")


@app.command("run-complex")
def run_complex(dry_run: CommonDryRun = False, force: CommonForce = False) -> None:
    _ = (dry_run, force)
    _not_implemented("run-complex")


@app.command("score-esm")
def score_esm(force: CommonForce = False) -> None:
    _ = force
    _not_implemented("score-esm")


@app.command("score-ipsae")
def score_ipsae(
    pae_cutoff: Annotated[float, typer.Option("--pae-cutoff", help="PAE cutoff.")] = 10.0,
    dist_cutoff: Annotated[float, typer.Option("--dist-cutoff", help="Distance cutoff.")] = 15.0,
    force: CommonForce = False,
) -> None:
    _ = (pae_cutoff, dist_cutoff, force)
    _not_implemented("score-ipsae")


@app.command("aggregate")
def aggregate(force: CommonForce = False) -> None:
    _ = force
    _not_implemented("aggregate")


@app.command("pipeline")
def pipeline(dry_run: CommonDryRun = False, force: CommonForce = False) -> None:
    _ = (dry_run, force)
    _not_implemented("pipeline")
