"""Legacy options and Hydra structured configuration for Aerith.

The small :class:`PipelineConfig` model is retained for the original stage
commands.  New orchestration uses :class:`AerithConfig`, composed through
Hydra without allowing Hydra to take ownership of the process command line or
working directory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

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


# Hydra structured configuration -------------------------------------------------


@dataclass
class ProjectSettings:
    csv_path: str = "all_seq_PD1_May12.csv"
    work_dir: str = "work"
    output_dir: str = "af_output"
    results_dir: str = "results"
    target_chain: str = "A"
    binder_chain: str = "B"
    job_name_template: str = DEFAULT_COMPLEX_JOB_NAME_TEMPLATE
    seed: int = 42
    run_id: str | None = None
    limit: int | None = None
    prune: bool = False
    adopt_legacy: bool = False
    allow_partial: bool = False


@dataclass
class BackendSettings:
    name: str = "alphafold3"
    model: str = "alphafold3"
    image: str = "alphafold3:latest"
    image_id: str | None = None
    docker_bin: str = "docker"
    source_dir: str | None = None
    source_commit: str | None = None
    runtime_entry: str = "af3"
    target_data_json: str | None = None
    target_name: str = "target_A"
    model_dir: str = "/data/AF3_ckpt"
    checkpoint_path: str | None = None
    checkpoint_dir: str | None = None
    common_dir: str | None = None
    metadata_dir: str | None = None
    database_dir: str = "/data/AF3_database"
    command: list[str] = field(default_factory=list)
    minimum_primary_iptm: float = 0.70


@dataclass
class SecondaryBackendSettings(BackendSettings):
    """Optional cross-validation backend settings.

    Primary AlphaFold 3 execution is mandatory, so only a secondary backend
    has a meaningful enable/disable switch.
    """

    enabled: bool = False


@dataclass
class ESMToolSettings:
    enabled: bool = True
    run_on: str = "all"
    inverse_folding: bool = True
    esmfold: bool = True
    runtime_entry_if: str = "esm-if"
    runtime_entry_fold: str = "esmfold"
    model_cache: str = "/home/structure/.cache/torch/hub/checkpoints"
    inverse_folding_checkpoint: str = "esm_if1_gvp4_t16_142M_UR50.pt"
    esmfold_checkpoint: str = "esmfold_3B_v1.pt"
    timeout_seconds: int = 14400


@dataclass
class ScoringSettings:
    esm: ESMToolSettings = field(default_factory=ESMToolSettings)


@dataclass
class ConsensusSettings:
    anomaly_detection: bool = True
    anomaly_min_samples: int = 30
    robust_z_threshold: float = 3.5
    minimum_anomaly_metrics: int = 2
    explicit_different_epitope_jaccard: float = 0.10
    minimum_contact_residues_for_epitope_flag: int = 5
    target_alignment_min_residues: int = 20
    target_alignment_min_fraction: float = 0.70
    target_alignment_max_iterations: int = 3
    same_fold_tm_threshold: float = 0.50
    different_pose_rmsd_threshold: float = 5.0
    explicit_different_interface_pair_jaccard: float = 0.30


@dataclass
class FeatureSettings:
    name: str = "local_af3_db"
    image: str = "aerith/fold-runtime:local"
    image_id: str | None = None
    docker_bin: str = "docker"
    database_dir: str = "/data/AF3_database"
    mmseqs_binary: str | None = None
    mmseqs_id: str | None = None
    use_gpu: bool = True
    threads: int = 8
    split_memory_limit: str = "32G"
    iterations: int = 3
    primary_database: str = "uniref90_padded"
    environment_database: str = "mgnify_padded"
    template_database: str = "pdb_seqres_padded"
    use_environment_database: bool = True
    cache_dir: str = "${project.work_dir}/features"
    mmseqs_dir: str = "${features.database_dir}/mmseqs"
    pdb_seqres_fasta: str = "${features.database_dir}/pdb_seqres_2022_09_28.fasta"
    mmcif_dir: str = "${features.database_dir}/mmcif_files"
    timeout_seconds: int = 14400


@dataclass
class RosettaSettings:
    binary: str = (
        "/home/structure/Software/rosetta.source.release-408/main/source/bin/"
        "InterfaceAnalyzer.mpiserialization.linuxgccrelease"
    )
    database: str = "/home/structure/Software/rosetta.source.release-408/main/database"
    score_function: str = "ref2015"
    interface: str = "A_B"
    pack_input: bool = False
    pack_separated: bool = True
    compute_packstat: bool = True
    constant_seed: bool = True
    random_seed: int = 1111111
    timeout_seconds: int = 1800
    max_workers: int = 8


@dataclass
class InterfaceSettings:
    name: str = "biotite_rosetta"
    geometry_engine: str = "biotite"
    energy_engine: str = "rosetta_cli"
    distance: float = 5.0
    epitope_residues: str | None = None
    minimum_contact_pairs: int = 5
    minimum_epitope_coverage: float = 0.30
    # Purity remains an output annotation.  ``None`` keeps it out of the hard
    # shortlist gate, which is the appropriate default for narrow epitopes.
    minimum_epitope_purity: float | None = None
    sasa_point_number: int = 1000
    rosetta: RosettaSettings = field(default_factory=RosettaSettings)


@dataclass
class ClusteringSettings:
    name: str = "balanced"
    foldseek_binary: str = "foldseek"
    binder_tm_threshold: float = 0.50
    binder_coverage: float = 0.80
    multimer_tm_threshold: float = 0.65
    chain_tm_threshold: float = 0.50
    interface_lddt_threshold: float = 0.65
    epitope_jaccard_threshold: float = 0.50
    max_workers: int = 1


@dataclass
class RuntimeSettings:
    dry_run: bool = False
    force: bool = False
    gpu_ids: list[int] = field(default_factory=list)
    gpu_busy_threshold_mib: int = 100
    geometry_max_workers: int = 4
    dockerfile: str = "docker/runtime/Dockerfile"
    af3_source_dir: str = "/home/structure/Software/alphafold3-3.0.3"
    protenix_source_dir: str = "/home/structure/Software/Protenix-2.0.0"
    opendde_source_dir: str = "/home/structure/Software/OpenDDE"
    opendde_source_commit: str = "266ce4c49d492ad1077866000d83704999985f46"
    esm_source_dir: str = "/home/structure/Software/esm"
    esm_source_commit: str = "2b369911bb5b4b0dda914521b9475cad1656b2ac"
    mmseqs_release: str = "18-8cc5c"
    mmseqs_version: str = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
    mmseqs_archive_sha256: str = "83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7"
    foldseek_release: str = "10-941cd33"
    foldseek_version: str = "941cd33ff0771cd2e3f144e3293e22a2b87e9fda"
    foldseek_archive_sha256: str = (
        "af7a688ffd8625b356c380380fb5650ec811262a2d18bdb0faeda95cc4894a55"
    )
    minimum_build_free_gib: int = 45
    build_proxy: str | None = None
    build_add_host: str | None = None


@dataclass
class AerithConfig:
    project: ProjectSettings = field(default_factory=ProjectSettings)
    backend: BackendSettings = field(default_factory=BackendSettings)
    secondary_backend: SecondaryBackendSettings = field(
        default_factory=lambda: SecondaryBackendSettings(
            enabled=False,
            name="none",
            model="none",
            image="aerith/fold-runtime:local",
            runtime_entry="none",
        )
    )
    features: FeatureSettings = field(default_factory=FeatureSettings)
    scoring: ScoringSettings = field(default_factory=ScoringSettings)
    consensus: ConsensusSettings = field(default_factory=ConsensusSettings)
    interface: InterfaceSettings = field(default_factory=InterfaceSettings)
    clustering: ClusteringSettings = field(default_factory=ClusteringSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)


class ConfigError(ValueError):
    """Raised when a Hydra configuration cannot be composed or validated."""


@dataclass(frozen=True)
class ConfigValidationReport:
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    info: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def _register_config_store() -> None:
    """Register the root schema once for Hydra/OmegaConf type validation."""

    from hydra.core.config_store import ConfigStore

    ConfigStore.instance().store(name="aerith_schema", node=AerithConfig)


def _normalize_field_override(value: str) -> str:
    """Make dotted CLI overrides work with intentionally minimal YAML files.

    Hydra applies command-line overrides before the composed user config is
    merged with :class:`AerithConfig`. Consequently, a field supplied only by
    the Structured Config defaults (for example ``runtime.dry_run``) is not yet
    present when Hydra sees ``runtime.dry_run=true`` and strict composition
    rejects it. ``++`` has the desired semantics here: update the field when
    it is present, or add it temporarily when it is supplied by the schema.
    The subsequent structured merge still rejects misspelled or unknown
    fields.

    Config-group overrides and explicit Hydra ``+``, ``++`` or ``~`` syntax
    retain their native behavior.
    """

    override = value.strip()
    if not override or override.startswith(("+", "~")):
        return override
    key, separator, _raw_value = override.partition("=")
    if not separator:
        return override
    if "." in key and not key.startswith("hydra."):
        return f"++{override}"
    return override


def compose_hydra_config(
    config_path: Path,
    *,
    overrides: Iterable[str] = (),
    backend: str | None = None,
    secondary_backend: str | None = None,
) -> tuple[AerithConfig, Any]:
    """Compose a user YAML with built-in config groups and a structured schema.

    The returned tuple contains the dataclass object and the resolved OmegaConf
    object.  Relative paths deliberately remain relative to the invocation
    directory, matching normal CLI expectations.  ``hydra.job.chdir`` is
    forced off even if a caller supplies a conflicting Hydra default.
    """

    from hydra import compose, initialize_config_dir
    from hydra.errors import HydraException
    from omegaconf import DictConfig, OmegaConf

    path = config_path.expanduser().resolve()
    if not path.exists():
        raise ConfigError(f"configuration file does not exist: {config_path}")
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigError("Hydra configuration must be a YAML file (.yaml or .yml)")

    _register_config_store()
    hydra_overrides = [_normalize_field_override(value) for value in overrides]
    if backend:
        hydra_overrides.insert(0, f"backend={backend}")
    if secondary_backend:
        hydra_overrides.insert(0, f"secondary_backend={secondary_backend}")
    hydra_overrides.append("hydra.job.chdir=false")
    try:
        with initialize_config_dir(
            version_base="1.3",
            config_dir=str(path.parent),
            job_name="aerith",
        ):
            composed: DictConfig = compose(
                config_name=path.stem,
                overrides=hydra_overrides,
                return_hydra_config=False,
            )
        schema = OmegaConf.structured(AerithConfig)
        resolved = OmegaConf.merge(schema, composed)
        OmegaConf.resolve(resolved)
        obj = OmegaConf.to_object(resolved)
    except (
        HydraException,
        Exception,
    ) as exc:  # OmegaConf has several validation subclasses
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"invalid Hydra configuration {config_path}: {exc}") from exc
    if not isinstance(obj, AerithConfig):
        raise ConfigError(f"configuration did not resolve to AerithConfig: {config_path}")
    return obj, resolved


def config_as_dict(config: AerithConfig) -> dict[str, Any]:
    """Return a JSON/YAML-safe representation of a resolved config."""

    return asdict(config)


def validate_hydra_config(
    config: AerithConfig, *, check_paths: bool = True
) -> ConfigValidationReport:
    """Validate cross-field invariants and local resources.

    Docker/GPU execution checks live in ``config doctor``; this function keeps
    validation deterministic and side-effect free.
    """

    errors: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    project = config.project

    if not project.target_chain.strip() or not project.binder_chain.strip():
        errors.append("target_chain and binder_chain must be non-empty")
    elif project.target_chain == project.binder_chain:
        errors.append("target_chain and binder_chain must be different")
    if len(project.target_chain) != 1 or len(project.binder_chain) != 1:
        errors.append("target_chain and binder_chain must be one-character chain IDs")
    if project.limit is not None and project.limit < 1:
        errors.append("project.limit must be at least 1")
    if config.interface.distance <= 0:
        errors.append("interface.distance must be positive")
    if config.interface.geometry_engine != "biotite":
        errors.append("interface.geometry_engine must be biotite")
    if config.interface.energy_engine not in {"none", "rosetta_cli"}:
        errors.append("interface.energy_engine must be none or rosetta_cli")
    if config.interface.minimum_contact_pairs < 1:
        errors.append("interface.minimum_contact_pairs must be at least 1")
    if config.interface.sasa_point_number < 1:
        errors.append("interface.sasa_point_number must be at least 1")
    if config.interface.rosetta.max_workers < 1:
        errors.append("interface.rosetta.max_workers must be at least 1")
    if config.interface.rosetta.timeout_seconds < 1:
        errors.append("interface.rosetta.timeout_seconds must be at least 1")
    if config.interface.rosetta.random_seed < 1:
        errors.append("interface.rosetta.random_seed must be at least 1")
    if config.features.threads < 1:
        errors.append("features.threads must be at least 1")
    if config.features.iterations < 1:
        errors.append("features.iterations must be at least 1")
    if config.features.timeout_seconds < 1:
        errors.append("features.timeout_seconds must be at least 1")
    if config.scoring.esm.timeout_seconds < 1:
        errors.append("scoring.esm.timeout_seconds must be at least 1")
    if config.runtime.geometry_max_workers < 1:
        errors.append("runtime.geometry_max_workers must be at least 1")
    expected_rosetta_interface = f"{project.target_chain}_{project.binder_chain}"
    if (
        config.interface.energy_engine == "rosetta_cli"
        and config.interface.rosetta.interface != expected_rosetta_interface
    ):
        errors.append(
            "interface.rosetta.interface must match configured chains: "
            f"expected {expected_rosetta_interface}"
        )
    if config.interface.minimum_epitope_purity is None:
        warnings.append(
            "interface.minimum_epitope_purity is deprecated and has no effect; "
            "remove it from configuration"
        )
    else:
        errors.append(
            "interface.minimum_epitope_purity is no longer supported; "
            "only epitope coverage is used for filtering"
        )
    for name, value in (
        ("minimum_epitope_coverage", config.interface.minimum_epitope_coverage),
        ("binder_tm_threshold", config.clustering.binder_tm_threshold),
        ("binder_coverage", config.clustering.binder_coverage),
        ("multimer_tm_threshold", config.clustering.multimer_tm_threshold),
        ("chain_tm_threshold", config.clustering.chain_tm_threshold),
        ("interface_lddt_threshold", config.clustering.interface_lddt_threshold),
        ("epitope_jaccard_threshold", config.clustering.epitope_jaccard_threshold),
    ):
        if value is not None and not 0 <= value <= 1:
            errors.append(f"{name} must be between 0 and 1")
    if config.backend.name != "alphafold3":
        errors.append("the primary backend must be alphafold3")
    if config.secondary_backend.name not in {"none", "protenix", "opendde"}:
        errors.append(f"unsupported secondary backend: {config.secondary_backend.name}")
    if config.secondary_backend.enabled and config.secondary_backend.name == "none":
        errors.append("secondary_backend.enabled cannot be true when name is none")
    if not config.secondary_backend.enabled and config.secondary_backend.name != "none":
        errors.append("secondary_backend must be enabled when a secondary backend is selected")
    if config.secondary_backend.enabled and config.secondary_backend.name == config.backend.name:
        errors.append("primary and secondary backends must be different")
    if config.secondary_backend.enabled and config.secondary_backend.image != config.backend.image:
        errors.append("primary and secondary backends must use the same unified image")
    if config.backend.name == "alphafold3" and not config.backend.target_name.strip():
        errors.append("backend.target_name must be non-empty for AlphaFold 3")
    if config.scoring.esm.run_on != "all":
        errors.append("scoring.esm.run_on must be all")
    if not 0 <= config.secondary_backend.minimum_primary_iptm <= 1:
        errors.append("secondary_backend.minimum_primary_iptm must be between 0 and 1")
    if config.consensus.anomaly_min_samples < 1:
        errors.append("consensus.anomaly_min_samples must be positive")
    if config.consensus.minimum_anomaly_metrics < 1:
        errors.append("consensus.minimum_anomaly_metrics must be positive")
    if not 0 < config.consensus.target_alignment_min_fraction <= 1:
        errors.append("consensus.target_alignment_min_fraction must be in (0, 1]")
    if not 0 <= config.consensus.same_fold_tm_threshold <= 1:
        errors.append("consensus.same_fold_tm_threshold must be between 0 and 1")
    if not 0 <= config.consensus.explicit_different_interface_pair_jaccard <= 1:
        errors.append("consensus.explicit_different_interface_pair_jaccard must be between 0 and 1")
    if config.consensus.different_pose_rmsd_threshold <= 0:
        errors.append("consensus.different_pose_rmsd_threshold must be positive")
    if config.runtime.minimum_build_free_gib < 1:
        errors.append("runtime.minimum_build_free_gib must be positive")

    if check_paths:
        csv_path = Path(project.csv_path).expanduser()
        if not csv_path.is_file():
            errors.append(f"project.csv_path does not exist: {csv_path}")
        database_dir = Path(config.features.database_dir).expanduser()
        if not database_dir.is_dir():
            errors.append(f"features.database_dir does not exist: {database_dir}")
        else:
            mmseqs_dir = Path(config.features.mmseqs_dir).expanduser()
            required = [
                mmseqs_dir / config.features.primary_database,
                mmseqs_dir / config.features.template_database,
                Path(config.features.pdb_seqres_fasta).expanduser(),
                Path(config.features.mmcif_dir).expanduser(),
            ]
            if config.features.use_environment_database:
                required.append(mmseqs_dir / config.features.environment_database)
            for required_path in required:
                if not required_path.exists():
                    errors.append(f"required AF3 database path does not exist: {required_path}")
        if config.interface.energy_engine == "rosetta_cli":
            binary = Path(config.interface.rosetta.binary).expanduser()
            database = Path(config.interface.rosetta.database).expanduser()
            if not binary.is_file():
                errors.append(f"Rosetta binary does not exist: {binary}")
            if not database.is_dir():
                errors.append(f"Rosetta database does not exist: {database}")
        if config.backend.source_dir:
            source = Path(config.backend.source_dir).expanduser()
            if not source.is_dir():
                errors.append(f"backend.source_dir does not exist: {source}")
        if config.backend.name == "alphafold3" and config.backend.target_data_json:
            target_data = Path(config.backend.target_data_json).expanduser()
            if not target_data.is_file():
                errors.append(f"backend.target_data_json does not exist: {target_data}")
        model_dir = Path(config.backend.model_dir).expanduser()
        if not model_dir.is_dir():
            errors.append(f"backend.model_dir does not exist: {model_dir}")
        elif config.backend.name == "opendde":
            checkpoint = Path(
                config.backend.checkpoint_path or model_dir / "checkpoint" / "opendde.pt"
            ).expanduser()
            if not checkpoint.is_file():
                errors.append(f"OpenDDE checkpoint does not exist: {checkpoint}")
        secondary = config.secondary_backend
        if secondary.enabled:
            if secondary.name == "protenix":
                checkpoint = Path(
                    secondary.checkpoint_path or "/home/structure/checkpoint/protenix-v2.pt"
                ).expanduser()
                common = Path(secondary.common_dir or "/home/structure/common").expanduser()
                metadata = Path(
                    secondary.metadata_dir or "/home/structure/Software/OpenDDE/common"
                ).expanduser()
                for path in (
                    checkpoint,
                    common / "components.cif",
                    common / "components.cif.rdkit_mol.pkl",
                    common / "obsolete_release_date.csv",
                    common / "clusters-by-entity-40.txt",
                    metadata / "release_date_cache.json",
                    metadata / "obsolete_to_successor.json",
                ):
                    if not path.exists():
                        errors.append(f"required Protenix runtime asset does not exist: {path}")
            elif secondary.name == "opendde":
                checkpoint = Path(
                    secondary.checkpoint_path
                    or "/home/structure/Software/OpenDDE/checkpoint/opendde.pt"
                ).expanduser()
                if not checkpoint.is_file():
                    errors.append(f"OpenDDE checkpoint does not exist: {checkpoint}")
                common = Path(
                    secondary.common_dir or "/home/structure/Software/OpenDDE/common"
                ).expanduser()
                for path in (
                    common / "components.cif",
                    common / "release_date_cache.json",
                    common / "obsolete_to_successor.json",
                ):
                    if not path.exists():
                        errors.append(f"required OpenDDE runtime asset does not exist: {path}")
        if config.scoring.esm.enabled:
            model_cache = Path(config.scoring.esm.model_cache).expanduser()
            if not model_cache.is_dir():
                errors.append(f"ESM model cache does not exist: {model_cache}")
            else:
                for filename in (
                    config.scoring.esm.inverse_folding_checkpoint,
                    config.scoring.esm.esmfold_checkpoint,
                ):
                    if not (model_cache / filename).is_file():
                        errors.append(
                            f"required ESM checkpoint does not exist: {model_cache / filename}"
                        )
        if config.features.mmseqs_binary:
            errors.append(
                "features.mmseqs_binary host overrides are disabled; "
                "use the pinned GPU MMseqs2 binary inside features.image"
            )

    if not errors:
        info.append(f"configuration valid for backend {config.backend.name}")
    return ConfigValidationReport(tuple(errors), tuple(warnings), tuple(info))
