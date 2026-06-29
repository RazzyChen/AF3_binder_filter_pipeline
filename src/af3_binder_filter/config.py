"""Configuration defaults for Aerith."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


DEFAULT_COMPLEX_JOB_NAME_TEMPLATE = "sample_{sample_no}_binder_candiate_complex_pred"


class AF3DockerConfig(BaseModel):
    """Paths and image settings for AlphaFold 3 Docker execution."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    docker_bin: str = "docker"
    image: str = "alphafold3:latest"
    model_dir: Path = Path("/data/AF3_ckpt")
    database_dir: Path = Path("/data/AF3_database")
    jax_cache_dir: Path = Path("/home/structure/alphafold_jax_cache")
    xla_client_mem_fraction: str = "0.98"


class ESMConfig(BaseModel):
    """Settings for ESM inverse folding scoring."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    conda_bin: str = "conda"
    conda_env: str = "esm"
    scorer_path: Path = Path(
        "/home/structure/Software/esm/examples/inverse_folding/score_log_likelihoods.py"
    )


class ESMFoldConfig(BaseModel):
    """Settings for ESMFold single-chain structure prediction."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    conda_bin: str = "conda"
    conda_env: str = "esm"
    binary: str = "esm-fold"
    model_dir: Path | None = None
    num_recycles: int | None = None
    max_tokens_per_batch: int | None = None
    chunk_size: int | None = None
    cpu_only: bool = False
    cpu_offload: bool = False


class PipelineConfig(BaseModel):
    """Runtime configuration with cluster defaults."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    csv_path: Path = Path("tests/AF3_pipeline_dev_sample.csv")
    work_dir: Path = Path("work")
    output_dir: Path = Path("af_output")
    target_chain: str = "A"
    binder_chain: str = "B"
    gpu_busy_threshold_mib: int = 100
    job_name_template: str = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE
    af3: AF3DockerConfig = Field(default_factory=AF3DockerConfig)
    esm: ESMConfig = Field(default_factory=ESMConfig)
    esmfold: ESMFoldConfig = Field(default_factory=ESMFoldConfig)

    @property
    def complex_input_dir(self) -> Path:
        return self.work_dir / "complex_inputs"

    @property
    def target_input_dir(self) -> Path:
        return self.work_dir / "target_input"

    @property
    def score_dir(self) -> Path:
        return self.work_dir / "scores"
