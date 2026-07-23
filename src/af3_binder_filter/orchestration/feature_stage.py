"""Cohesive feature stage orchestration boundary."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from af3_binder_filter.backends import build_backend_command
from af3_binder_filter.af3_json import write_target_input
from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeaturePreparation,
    prepare_target_features,
)
from af3_binder_filter.io_utils import (
    atomic_write_json,
    atomic_write_text,
)
from af3_binder_filter.jobs import sequence_sha256
from af3_binder_filter.target_data import extract_target_features
from af3_binder_filter.orchestration.command_runtime import _run_prediction_command
from af3_binder_filter.orchestration.context import (
    PipelineExecutionError,
    RunContext,
    _af3_feature_fingerprint,
    _container_name,
    _context_feature_fingerprint,
    _existing_or_new_manifest,
    _expected_feature_cache_dir,
    _runtime_gpus,
)
from af3_binder_filter.orchestration.feature_identity import (
    _absolute_target_features,
    _af3_bundle_artifact_identity,
    _af3_bundle_from_manifest,
    _bind_feature_content,
)


def prepare_features_stage(context: RunContext) -> FeaturePreparation:
    log_dir = context.layout.stage("features").logs
    if (
        context.config.backend.name == "alphafold3"
        and context.config.backend.target_data_json
    ):
        preparation = _prepare_af3_target_features(context)
        if preparation.command:
            atomic_write_text(
                log_dir / "prepare_features.command.txt",
                " ".join(preparation.command) + "\n",
            )
        return preparation
    feature_gpu = _runtime_gpus(
        context,
        job_count=1,
        stage_name="features",
    )[0]
    preparation = prepare_target_features(
        context.config.features,
        context.plan.target_sequence,
        dry_run=context.config.runtime.dry_run,
        force=context.config.runtime.force,
        gpu_index=feature_gpu.index,
        container_name=_container_name(
            context,
            "feature-builder",
            feature_gpu.index,
        ),
        log_dir=log_dir,
        database_identity=context.feature_database_identity,
    )
    if preparation.command:
        atomic_write_text(
            log_dir / "prepare_features.command.txt",
            " ".join(preparation.command) + "\n",
        )
    return preparation


def run_prepare_features_only(context: RunContext) -> FeaturePreparation:
    feature_fingerprint = _context_feature_fingerprint(context)
    manifest = _existing_or_new_manifest(context, feature_fingerprint)
    manifest.stage_status["features"] = "running"
    manifest.status = "running"
    manifest.write(context.manifest_path)
    try:
        preparation = prepare_features_stage(context)
        if preparation.bundle is not None:
            _bind_feature_content(manifest, preparation.bundle)
        manifest.stage_status["features"] = (
            "dry_run" if context.config.runtime.dry_run else "success"
        )
        manifest.status = (
            "dry_run" if context.config.runtime.dry_run else "success"
        )
        manifest.write(context.manifest_path)
        return preparation
    except KeyboardInterrupt:
        message = "feature preparation interrupted by user"
        manifest.stage_status["features"] = "interrupted"
        manifest.status = "interrupted"
        if message not in manifest.errors:
            manifest.errors.append(message)
        manifest.write(context.manifest_path)
        raise
    except Exception as exc:
        manifest.stage_status["features"] = "error"
        manifest.status = "error"
        manifest.errors.append(str(exc))
        manifest.write(context.manifest_path)
        raise


def _prepare_af3_target_features(context: RunContext) -> FeaturePreparation:
    config = context.config
    target_sequence = context.plan.target_sequence
    fingerprint = _context_feature_fingerprint(context)
    if fingerprint != _af3_feature_fingerprint(config, target_sequence):
        raise PipelineExecutionError("AF3 target feature fingerprint changed during setup")
    cache_dir = _expected_feature_cache_dir(
        config,
        target_sequence,
        fingerprint=fingerprint,
    )
    cached = None if config.runtime.force else _af3_bundle_from_manifest(
        cache_dir,
        target_sequence=target_sequence,
        fingerprint=fingerprint,
    )
    if cached is not None:
        return FeaturePreparation(cached, None, reused=True)

    cache_dir.mkdir(parents=True, exist_ok=True)
    configured_data = config.backend.target_data_json
    source_data: Path
    command: list[str] | None = None
    build_root: Path | None = None
    if configured_data:
        source_data = Path(configured_data).expanduser().resolve()
        if not source_data.is_file():
            raise PipelineExecutionError(
                f"configured AF3 target data JSON does not exist: {source_data}"
            )
    else:
        if config.runtime.dry_run:
            build_root = cache_dir / ".target-only-dry-run"
            build_root.mkdir(parents=True, exist_ok=True)
        else:
            build_root = Path(
                tempfile.mkdtemp(prefix=".target-only-", dir=cache_dir)
            )
        input_dir = build_root / "input"
        output_dir = build_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        target_input = write_target_input(
            target_sequence=target_sequence,
            output_dir=input_dir,
            name=config.backend.target_name,
            target_chain=config.project.target_chain,
            seed=config.project.seed,
            force=True,
        )
        feature_gpu = _runtime_gpus(
            context,
            job_count=1,
            stage_name="features",
        )[0]
        gpu_index = feature_gpu.index
        command = build_backend_command(
            config,
            input_dir=input_dir,
            output_dir=output_dir,
            gpu_index=gpu_index,
            container_name=_container_name(
                context,
                "target_features",
                gpu_index,
            ),
        )
        if config.runtime.dry_run:
            return FeaturePreparation(None, tuple(command), reused=False)
        return_code = _run_prediction_command(
            context,
            command,
            name="target_features",
            timeout_seconds=config.features.timeout_seconds,
        )
        if return_code != 0:
            shutil.rmtree(build_root, ignore_errors=True)
            raise PipelineExecutionError(
                f"AF3 target-only feature command returned {return_code}"
            )
        target_name = target_input.stem
        candidates = [
            output_dir / f"{target_name}_data.json",
            output_dir / target_name / f"{target_name}_data.json",
            *sorted(output_dir.rglob(f"{target_name}_data.json")),
        ]
        source_data = next((path for path in candidates if path.is_file()), candidates[0])
        if not source_data.is_file():
            shutil.rmtree(build_root, ignore_errors=True)
            raise PipelineExecutionError(
                f"AF3 target-only run did not produce {target_name}_data.json"
            )

    try:
        extracted = extract_target_features(
            source_data,
            cache_dir,
            chain_id=config.project.target_chain,
            prefix=config.backend.target_name,
            expected_sequence=target_sequence,
            force=True,
        )
        absolute_features = _absolute_target_features(extracted, cache_dir)
        target_payload = json.loads(source_data.read_text(encoding="utf-8"))
        atomic_write_json(cache_dir / "target_data.json", target_payload)
        bundle = AF3FeatureBundle(
            sequence_sha256=sequence_sha256(target_sequence),
            cache_dir=cache_dir,
            target_data_json=cache_dir / "target_data.json",
            features=absolute_features,
            fingerprint=fingerprint,
        )
        bundle.validate()
        atomic_write_json(
            cache_dir / "manifest.json",
            {
                "version": 1,
                "mode": "alphafold3_target_only",
                "fingerprint": fingerprint,
                "sequence_sha256": sequence_sha256(target_sequence),
                "source_target_data_json": str(source_data),
                "backend_image": config.backend.image,
                "backend_image_id": config.backend.image_id,
                "features": {
                    "unpaired_msa_path": absolute_features.unpaired_msa_path,
                    "paired_msa_path": absolute_features.paired_msa_path,
                    "templates": absolute_features.templates,
                },
                "artifact_identity": _af3_bundle_artifact_identity(bundle),
            },
        )
        return FeaturePreparation(
            bundle,
            tuple(command) if command is not None else None,
            reused=False,
        )
    finally:
        if build_root is not None:
            shutil.rmtree(build_root, ignore_errors=True)
