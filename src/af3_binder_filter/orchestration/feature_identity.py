"""Cohesive feature identity orchestration boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from af3_binder_filter.af3_json import TargetFeatures
from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeatureBundle,
    af3_feature_bundle_artifact_identity,
    cached_target_features,
    feature_bundle_content_sha256,
)
from af3_binder_filter.jobs import sequence_sha256
from af3_binder_filter.manifest import (
    RunManifest,
    load_manifest,
)
from af3_binder_filter.secondary_features import (
    SecondaryFeatureBundle,
    secondary_feature_bundle_content_sha256,
)
from af3_binder_filter.orchestration.context import (
    RunContext,
    context_feature_fingerprint,
    expected_feature_cache_dir,
)


def target_feature_cache_hit(context: RunContext) -> bool:
    """Return whether the exact target feature bundle is reusable."""

    if context.config.runtime.force:
        return False
    if (
        context.config.backend.name == "alphafold3"
        and context.config.backend.target_data_json
    ):
        fingerprint = context_feature_fingerprint(context)
        return (
            af3_bundle_from_manifest(
                expected_feature_cache_dir(
                    context.config,
                    context.plan.target_sequence,
                    fingerprint=fingerprint,
                ),
                target_sequence=context.plan.target_sequence,
                fingerprint=fingerprint,
            )
            is not None
        )
    return (
        cached_target_features(
            context.config.features,
            context.plan.target_sequence,
            database_identity=context.feature_database_identity,
        )
        is not None
    )


def absolute_target_features(
    features: TargetFeatures,
    root: Path,
) -> TargetFeatures:
    def absolute(value: str | None) -> str | None:
        if not value:
            return None
        path = Path(value)
        return str(path if path.is_absolute() else (root / path).resolve())

    templates: list[dict[str, Any]] = []
    for template in features.templates:
        converted = dict(template)
        converted["mmcifPath"] = absolute(str(template.get("mmcifPath") or ""))
        templates.append(converted)
    return TargetFeatures(
        unpaired_msa_path=absolute(features.unpaired_msa_path),
        paired_msa_path=absolute(features.paired_msa_path),
        templates=templates,
    )


def af3_bundle_from_manifest(
    cache_dir: Path,
    *,
    target_sequence: str,
    fingerprint: str,
) -> AF3FeatureBundle | None:
    payload = load_manifest(cache_dir / "manifest.json")
    if (
        payload is None
        or payload.get("fingerprint") != fingerprint
        or payload.get("sequence_sha256") != sequence_sha256(target_sequence)
    ):
        return None
    feature_payload = payload.get("features")
    if not isinstance(feature_payload, dict):
        return None
    bundle = AF3FeatureBundle(
        sequence_sha256=sequence_sha256(target_sequence),
        cache_dir=cache_dir,
        target_data_json=cache_dir / "target_data.json",
        features=TargetFeatures(
            unpaired_msa_path=feature_payload.get("unpaired_msa_path"),
            paired_msa_path=feature_payload.get("paired_msa_path"),
            templates=list(feature_payload.get("templates") or []),
        ),
        fingerprint=fingerprint,
    )
    try:
        bundle.validate()
    except Exception:
        return None
    if payload.get("artifact_identity") != af3_bundle_artifact_identity(bundle):
        return None
    return bundle


def af3_bundle_artifact_identity(bundle: AF3FeatureBundle) -> dict[str, Any]:
    return dict(af3_feature_bundle_artifact_identity(bundle))


def bind_feature_content(
    manifest: RunManifest,
    bundle: FeatureBundle | AF3FeatureBundle,
) -> str:
    """Bind the exact prepared MSA/template bytes to the run manifest."""

    digest = feature_bundle_content_sha256(bundle)
    manifest.feature_content_sha256 = digest
    manifest.artifact_sha256["target_features"] = digest
    return digest


def prediction_feature_identity(
    bundle: FeatureBundle | AF3FeatureBundle | SecondaryFeatureBundle,
) -> str:
    """Bind prediction reuse to generation settings and exact feature bytes."""

    content_sha256 = (
        secondary_feature_bundle_content_sha256(bundle)
        if isinstance(bundle, SecondaryFeatureBundle)
        else feature_bundle_content_sha256(bundle)
    )
    return sequence_sha256(
        json.dumps(
            {
                "generation_fingerprint": bundle.fingerprint,
                "content_sha256": content_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )


def primary_prediction_feature_identity(context: RunContext) -> str:
    """Recover the exact primary feature identity for standalone validation."""

    feature_fingerprint = context_feature_fingerprint(context)
    manifest_path = getattr(context, "manifest_path", None)
    payload = load_manifest(manifest_path) if isinstance(manifest_path, Path) else None
    content_sha256 = (
        payload.get("feature_content_sha256")
        if isinstance(payload, dict)
        else None
    )
    if not isinstance(content_sha256, str) or not content_sha256:
        # Compatibility for direct library callers that predate/run outside a
        # validated RunContext. Production standalone CLI manifests require
        # this field whenever the feature stage succeeded.
        return feature_fingerprint
    return sequence_sha256(
        json.dumps(
            {
                "generation_fingerprint": feature_fingerprint,
                "content_sha256": content_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    )
