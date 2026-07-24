"""Prediction input/output adapters for AlphaFold 3, Protenix, and OpenDDE."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from af3_binder_filter.config import AerithConfig, BackendSettings
from af3_binder_filter.features import (
    AF3FeatureBundle,
    FeatureBundle,
    write_query_only_msa,
)
from af3_binder_filter.io_utils import atomic_write_json
from af3_binder_filter.jobs import JobSpec


class BackendError(RuntimeError):
    """Raised for an invalid backend contract or an unparseable prediction."""


RUNTIME_SOURCE_CONTEXTS = (
    "af3-src",
    "protenix-src",
    "opendde-src",
    "esm-src",
)
RUNTIME_SOURCE_BUNDLE_SCHEMA = "aerith.runtime-source-bundle.v1"
RUNTIME_SOURCE_BUNDLE_MANIFEST = "manifest.json"


@dataclass(frozen=True, slots=True)
class RuntimeSourceBundle:
    """A verified, portable set of filtered BuildKit source contexts."""

    root: Path
    bundle_sha256: str
    context_paths: dict[str, Path]
    context_sha256: dict[str, str]
    manifest: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UnifiedPrediction:
    job_id: str
    backend: str
    status: str
    best_model_path: Path | None = None
    summary_path: Path | None = None
    confidence_path: Path | None = None
    ranking_score: float | None = None
    iptm: float | None = None
    ptm: float | None = None
    plddt: float | None = None
    pae: Any | None = None
    error: str | None = None
    fingerprint_valid: bool = False

    def as_row(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("best_model_path", "summary_path", "confidence_path"):
            if data[key] is not None:
                data[key] = str(data[key])
        return data


class OutputAdapter(Protocol):
    backend_name: str

    def parse(self, job: JobSpec, output_dir: Path) -> UnifiedPrediction: ...


def _number(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, list):
            numbers = [_number(item) for item in value]
            present = [number for number in numbers if number is not None]
            return sum(present) / len(present) if present else None
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BackendError(f"confidence JSON must be an object: {path}")
    return payload


def _matching_model(summary_path: Path, cif_paths: Sequence[Path]) -> Path | None:
    stem = summary_path.stem
    model_stem = stem.replace("_summary_confidences", "_model")
    model_stem = model_stem.replace("_summary_confidence", "")
    direct = summary_path.with_name(model_stem + ".cif")
    if direct.exists():
        return direct
    sample_match = re.search(r"_sample_(\d+)$", stem)
    if sample_match:
        sample_suffix = f"_sample_{sample_match.group(1)}"
        matching = [path for path in cif_paths if sample_suffix in path.stem]
        if matching:
            return sorted(matching)[0]
    same_parent = [path for path in cif_paths if path.parent == summary_path.parent]
    return sorted(same_parent or list(cif_paths))[0] if (same_parent or cif_paths) else None


def _path_belongs_to_job(path: Path, job_id: str) -> bool:
    return job_id in path.parts or path.stem == job_id or path.stem.startswith(f"{job_id}_")


class RankedJsonOutputAdapter:
    """Shared Protenix/OpenDDE output contract."""

    def __init__(self, backend_name: str):
        self.backend_name = backend_name

    def parse(self, job: JobSpec, output_dir: Path) -> UnifiedPrediction:
        roots = [output_dir / job.job_id, output_dir]
        summary_paths: list[Path] = []
        cif_paths: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            summary_paths.extend(root.rglob("*summary_confidence*.json"))
            cif_paths.extend(root.rglob("*.cif"))
            if summary_paths and cif_paths:
                break
        summary_paths = sorted(set(summary_paths))
        cif_paths = sorted(set(cif_paths))
        # A batched output directory can contain artifacts for many jobs. The
        # global fallback is useful for backends that add their own nesting,
        # but it must never make a failed job inherit another job's model.
        summary_paths = [path for path in summary_paths if _path_belongs_to_job(path, job.job_id)]
        cif_paths = [path for path in cif_paths if _path_belongs_to_job(path, job.job_id)]
        if not summary_paths:
            return UnifiedPrediction(
                job.job_id,
                self.backend_name,
                "missing",
                error=(
                    f"no summary confidence JSON for job {job.job_id!r} found under {output_dir}"
                ),
            )

        candidates: list[tuple[float, Path, dict[str, Any]]] = []
        parse_errors: list[str] = []
        for path in summary_paths:
            try:
                payload = _load_object(path)
                score = _number(payload.get("ranking_score"))
                candidates.append((score if score is not None else -math.inf, path, payload))
            except (OSError, json.JSONDecodeError, BackendError) as exc:
                parse_errors.append(f"{path}: {exc}")
        if not candidates:
            return UnifiedPrediction(
                job.job_id,
                self.backend_name,
                "error",
                error="; ".join(parse_errors) or "all summary confidence JSON files were invalid",
            )
        _score, summary_path, summary = max(candidates, key=lambda item: (item[0], str(item[1])))
        model_path = _matching_model(summary_path, cif_paths)
        if model_path is None or not model_path.is_file():
            return UnifiedPrediction(
                job.job_id,
                self.backend_name,
                "missing",
                summary_path=summary_path,
                ranking_score=_number(summary.get("ranking_score")),
                error="best summary has no matching CIF model",
            )
        expected_full_data = summary_path.with_name(
            summary_path.name.replace("_summary_confidences_", "_full_data_").replace(
                "_summary_confidence_", "_full_data_"
            )
        )
        full_data_candidates = sorted(summary_path.parent.glob("*full_data*.json"))
        confidence_path = (
            expected_full_data
            if expected_full_data.exists()
            else (full_data_candidates[0] if full_data_candidates else summary_path)
        )
        confidence: dict[str, Any] = {}
        if confidence_path != summary_path:
            try:
                confidence = _load_object(confidence_path)
            except Exception:
                confidence = {}
        return UnifiedPrediction(
            job_id=job.job_id,
            backend=self.backend_name,
            status="success",
            best_model_path=model_path,
            summary_path=summary_path,
            confidence_path=confidence_path,
            ranking_score=_number(summary.get("ranking_score")),
            iptm=_number(summary.get("iptm")),
            ptm=_number(summary.get("ptm")),
            plddt=_number(summary.get("plddt")),
            pae=(
                confidence.get("pae")
                or confidence.get("predicted_aligned_error")
                or confidence.get("token_pair_pae")
            ),
        )


class AlphaFold3OutputAdapter:
    backend_name = "alphafold3"

    def parse(self, job: JobSpec, output_dir: Path) -> UnifiedPrediction:
        job_dir = output_dir / job.job_id
        if not job_dir.is_dir():
            return UnifiedPrediction(
                job.job_id,
                self.backend_name,
                "missing",
                error=f"missing output directory: {job_dir}",
            )
        ranking_path = job_dir / f"{job.job_id}_ranking_scores.csv"
        best_seed: str | None = None
        best_sample: str | None = None
        best_score: float | None = None
        if ranking_path.exists():
            import csv

            with ranking_path.open(encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    score = _number(row.get("ranking_score"))
                    if score is not None and (best_score is None or score > best_score):
                        best_score = score
                        best_seed = row.get("seed")
                        best_sample = row.get("sample")
        roots = [job_dir]
        if best_seed is not None and best_sample is not None:
            roots.insert(0, job_dir / f"seed-{best_seed}_sample-{best_sample}")
        ranked_basename = (
            f"{job.job_id}_seed-{best_seed}_sample-{best_sample}"
            if best_seed is not None and best_sample is not None
            else None
        )
        summary_path = next(
            (
                path
                for root in roots
                for path in [
                    root
                    / (
                        f"{ranked_basename}_summary_confidences.json"
                        if ranked_basename and root != job_dir
                        else f"{job.job_id}_summary_confidences.json"
                    )
                ]
                if path.exists()
            ),
            None,
        )
        confidence_path = next(
            (
                path
                for root in roots
                for path in [
                    root
                    / (
                        f"{ranked_basename}_confidences.json"
                        if ranked_basename and root != job_dir
                        else f"{job.job_id}_confidences.json"
                    )
                ]
                if path.exists()
            ),
            None,
        )
        model_path = next(
            (
                path
                for root in roots
                for path in [
                    root
                    / (
                        f"{ranked_basename}_model.cif"
                        if ranked_basename and root != job_dir
                        else f"{job.job_id}_model.cif"
                    )
                ]
                if path.exists()
            ),
            None,
        )
        if summary_path is None:
            # AF3 sample filenames include the seed/sample in the basename.
            found = sorted(job_dir.rglob("*_summary_confidences.json"))
            summary_path = found[0] if found else None
        if confidence_path is None:
            found = sorted(job_dir.rglob("*_confidences.json"))
            confidence_path = next((path for path in found if "summary" not in path.name), None)
        if model_path is None:
            found = sorted(job_dir.rglob("*_model.cif"))
            model_path = found[0] if found else None
        if summary_path is None:
            return UnifiedPrediction(
                job.job_id,
                self.backend_name,
                "missing",
                error="AF3 output has no summary confidences",
            )
        try:
            summary = _load_object(summary_path)
            confidence = _load_object(confidence_path) if confidence_path else {}
        except Exception as exc:
            return UnifiedPrediction(job.job_id, self.backend_name, "error", error=str(exc))
        plddt = confidence.get("atom_plddts") or confidence.get("plddt")
        summary_ranking_score = _number(summary.get("ranking_score"))
        return UnifiedPrediction(
            job_id=job.job_id,
            backend=self.backend_name,
            status=("success" if model_path is not None else "missing"),
            best_model_path=model_path,
            summary_path=summary_path,
            confidence_path=confidence_path,
            ranking_score=(
                summary_ranking_score if summary_ranking_score is not None else best_score
            ),
            iptm=_number(summary.get("iptm")),
            ptm=_number(summary.get("ptm")),
            plddt=_number(plddt),
            pae=confidence.get("pae") or confidence.get("predicted_aligned_error"),
            error=(
                None
                if model_path is not None and confidence_path is not None
                else "AF3 output has confidence metrics but no complete model/confidence artifact"
            ),
        )


def output_adapter(name: str) -> OutputAdapter:
    if name == "alphafold3":
        return AlphaFold3OutputAdapter()
    if name in {"protenix", "opendde"}:
        return RankedJsonOutputAdapter(name)
    raise BackendError(f"unsupported backend: {name}")


def _protein_chain(
    sequence: str,
    *,
    paired_msa: Path | None,
    unpaired_msa: Path,
    templates: Path | None,
    validate_paths: bool = True,
) -> dict[str, Any]:
    if validate_paths and not unpaired_msa.is_file():
        raise BackendError(f"unpaired MSA does not exist: {unpaired_msa}")
    protein: dict[str, Any] = {
        "sequence": sequence,
        "count": 1,
        "unpairedMsaPath": str(unpaired_msa.resolve()),
    }
    if paired_msa is not None:
        if validate_paths and not paired_msa.is_file():
            raise BackendError(f"paired MSA does not exist: {paired_msa}")
        protein["pairedMsaPath"] = str(paired_msa.resolve())
    if templates is not None:
        if validate_paths and not templates.is_file():
            raise BackendError(f"template result does not exist: {templates}")
        protein["templatesPath"] = str(templates.resolve())
    return {"proteinChain": protein}


def make_protenix_style_input(
    job: JobSpec,
    *,
    target_features: Any,
    binder_msa_path: Path,
    binder_templates_path: Path,
    validate_paths: bool = True,
) -> dict[str, Any]:
    """Create the common local-feature contract used by Protenix/OpenDDE."""

    return {
        "name": job.job_id,
        "modelSeeds": [job.seed],
        "sequences": [
            _protein_chain(
                job.target_sequence,
                paired_msa=target_features.pairing_a3m,
                unpaired_msa=target_features.non_pairing_a3m,
                templates=target_features.hmmsearch_a3m,
                validate_paths=validate_paths,
            ),
            _protein_chain(
                job.binder_sequence,
                paired_msa=None,
                unpaired_msa=binder_msa_path,
                # Protenix performs an automatic template search whenever
                # use_template=true and templatesPath is absent. An explicit
                # query-only A3M means "no Binder template hits" and keeps the
                # secondary backend fully offline while target templates remain
                # enabled.
                templates=binder_templates_path,
                validate_paths=validate_paths,
            ),
        ],
    }


def write_backend_inputs(
    jobs: Sequence[JobSpec],
    config: AerithConfig,
    *,
    input_dir: Path,
    target_features: FeatureBundle | AF3FeatureBundle | None,
    backend_settings: BackendSettings | None = None,
    force: bool = False,
    allow_missing_features: bool = False,
) -> list[Path]:
    backend = backend_settings or config.backend
    input_dir.mkdir(parents=True, exist_ok=True)
    current: list[Path] = []
    if backend.name == "alphafold3":
        from af3_binder_filter.af3_json import TargetFeatures

        for job in jobs:
            feature_refs = TargetFeatures()
            if isinstance(target_features, AF3FeatureBundle):
                feature_refs = target_features.features
            elif isinstance(target_features, FeatureBundle):
                feature_refs = TargetFeatures(
                    unpaired_msa_path=str(target_features.non_pairing_a3m.resolve()),
                    paired_msa_path=str(target_features.pairing_a3m.resolve()),
                    templates=target_features.af3_templates(),
                )
            payload = {
                "dialect": "alphafold3",
                "version": 4,
                "name": job.job_id,
                "modelSeeds": [job.seed],
                "sequences": [
                    {
                        "protein": {
                            "id": job.target_chain,
                            "sequence": job.target_sequence,
                            "modifications": [],
                            "templates": feature_refs.templates,
                            **(
                                {"unpairedMsaPath": feature_refs.unpaired_msa_path}
                                if feature_refs.unpaired_msa_path
                                else {}
                            ),
                            **(
                                {"pairedMsaPath": feature_refs.paired_msa_path}
                                if feature_refs.paired_msa_path
                                else {}
                            ),
                        }
                    },
                    {
                        "protein": {
                            "id": job.binder_chain,
                            "sequence": job.binder_sequence,
                            "modifications": [],
                            "templates": [],
                            "unpairedMsa": f">query\n{job.binder_sequence}\n",
                            "pairedMsa": "",
                        }
                    },
                ],
            }
            path = input_dir / f"{job.job_id}.json"
            atomic_write_json(path, payload)
            current.append(path)
    else:
        if target_features is None:
            raise BackendError(f"{backend.name} requires local target features")
        payloads: list[dict[str, Any]] = []
        for job in jobs:
            binder_feature_dir = input_dir / "binder_msas" / job.job_id
            binder_msa = write_query_only_msa(
                job.binder_sequence,
                binder_feature_dir / "non_pairing.a3m",
                force=True,
            )
            binder_templates = write_query_only_msa(
                job.binder_sequence,
                binder_feature_dir / "hmmsearch.a3m",
                force=True,
            )
            payloads.append(
                make_protenix_style_input(
                    job,
                    target_features=target_features,
                    binder_msa_path=binder_msa,
                    binder_templates_path=binder_templates,
                    validate_paths=not allow_missing_features,
                )
            )
        path = input_dir / f"{backend.name}_jobs.json"
        atomic_write_json(path, payloads)
        current.append(path)

    if config.project.prune:
        keep = set(current)
        for old in input_dir.glob("*.json"):
            if old not in keep:
                old.unlink()
    return current


def build_backend_image_command(config: AerithConfig) -> list[str]:
    if not config.backend.source_dir:
        raise BackendError(f"{config.backend.name} has no local source_dir configured")
    return [
        config.backend.docker_bin,
        "build",
        "--tag",
        config.backend.image,
        str(Path(config.backend.source_dir).expanduser().resolve()),
    ]


_RUNTIME_IGNORED_NAMES = (
    ".git",
    ".venv",
    "__pycache__",
    "*.pyc",
    "checkpoint",
    "ckpt",
    "output",
    "outputs",
    "PD1_output",
    "test_outputs",
    "search_database",
    "examples",
    "tests",
    "docs",
    "benchmarks",
    "assets",
    "build",
    ".github",
    ".pytest_cache",
)


def _runtime_source_paths(config: AerithConfig) -> dict[str, Path]:
    return {
        "af3-src": Path(config.runtime.af3_source_dir).expanduser().resolve(),
        "protenix-src": Path(config.runtime.protenix_source_dir).expanduser().resolve(),
        "opendde-src": Path(config.runtime.opendde_source_dir).expanduser().resolve(),
        "esm-src": Path(config.runtime.esm_source_dir).expanduser().resolve(),
    }


def _git_head(source: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    actual = completed.stdout.strip() if completed.returncode == 0 else ""
    return actual or None


def _validated_runtime_source_heads(
    config: AerithConfig, sources: dict[str, Path]
) -> dict[str, str | None]:
    for name, source in sources.items():
        if not source.is_dir():
            raise BackendError(f"runtime build context {name} does not exist: {source}")
    heads = {name: _git_head(source) for name, source in sources.items()}
    expected_commits = {
        "opendde-src": config.runtime.opendde_source_commit,
        "esm-src": config.runtime.esm_source_commit,
    }
    for name, expected in expected_commits.items():
        actual = heads[name]
        if actual != expected:
            raise BackendError(
                f"runtime build context {name} commit mismatch: "
                f"expected {expected}, found {actual or 'unavailable'}"
            )
    return heads


def _update_hash_field(digest: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    digest.update(len(encoded).to_bytes(8, byteorder="big"))
    digest.update(encoded)


def _hash_runtime_context(root: Path) -> dict[str, int | str]:
    """Hash all BuildKit-visible paths, including names and permission bits."""

    if not root.is_dir():
        raise BackendError(f"runtime source context does not exist: {root}")
    digest = hashlib.sha256(b"aerith-runtime-context-v1\0")
    file_count = 0
    size_bytes = 0
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = metadata.st_mode & 0o7777
        if path.is_symlink():
            target = os.readlink(path)
            _update_hash_field(digest, "symlink")
            _update_hash_field(digest, relative)
            _update_hash_field(digest, f"{mode:o}")
            _update_hash_field(digest, target)
            file_count += 1
            size_bytes += len(os.fsencode(target))
        elif path.is_dir():
            _update_hash_field(digest, "directory")
            _update_hash_field(digest, relative)
            _update_hash_field(digest, f"{mode:o}")
        elif path.is_file():
            _update_hash_field(digest, "file")
            _update_hash_field(digest, relative)
            _update_hash_field(digest, f"{mode:o}")
            _update_hash_field(digest, str(metadata.st_size))
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            file_count += 1
            size_bytes += metadata.st_size
        else:
            raise BackendError(f"unsupported file in runtime source context: {path}")
    return {
        "sha256": digest.hexdigest(),
        "file_count": file_count,
        "size_bytes": size_bytes,
    }


def _runtime_bundle_digest(contexts: dict[str, dict[str, Any]]) -> str:
    identity = {
        "schema": RUNTIME_SOURCE_BUNDLE_SCHEMA,
        "contexts": {
            name: {
                "path": contexts[name]["path"],
                "sha256": contexts[name]["sha256"],
                "file_count": contexts[name]["file_count"],
                "size_bytes": contexts[name]["size_bytes"],
            }
            for name in RUNTIME_SOURCE_CONTEXTS
        },
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_runtime_source_bundle(
    config: AerithConfig,
    destination: Path,
    *,
    force: bool = False,
) -> RuntimeSourceBundle:
    """Copy four filtered source trees into an atomic, content-hashed bundle."""

    sources = _runtime_source_paths(config)
    heads = _validated_runtime_source_heads(config, sources)
    expanded_destination = destination.expanduser()
    if expanded_destination.is_symlink():
        raise BackendError(
            f"refusing to replace a symlink as a runtime source bundle: {expanded_destination}"
        )
    target_root = expanded_destination.resolve()
    if target_root.exists() and not force:
        raise BackendError(
            f"runtime source bundle already exists: {target_root}; use force to replace it"
        )
    if target_root.exists() and force:
        existing_manifest = target_root / RUNTIME_SOURCE_BUNDLE_MANIFEST
        try:
            existing_payload = json.loads(existing_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing_payload = None
        if (
            not isinstance(existing_payload, dict)
            or existing_payload.get("schema") != RUNTIME_SOURCE_BUNDLE_SCHEMA
        ):
            raise BackendError(f"refusing to replace a non-bundle path: {target_root}")
    target_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{target_root.name}.staging-",
            dir=target_root.parent,
        )
    )
    ignored = shutil.ignore_patterns(*_RUNTIME_IGNORED_NAMES)
    # AF3's src/alphafold3/common is package code and must be retained. The
    # OpenDDE root-level common directory contains mounted inference data.
    opendde_ignored = shutil.ignore_patterns(*_RUNTIME_IGNORED_NAMES, "common")
    try:
        contexts: dict[str, dict[str, Any]] = {}
        for name in RUNTIME_SOURCE_CONTEXTS:
            source = sources[name]
            copied = temporary / name
            shutil.copytree(
                source,
                copied,
                ignore=opendde_ignored if name == "opendde-src" else ignored,
            )
            context = _hash_runtime_context(copied)
            contexts[name] = {
                "path": name,
                **context,
                "source_path": str(source),
                "source_git_commit": heads[name],
            }
        bundle_sha256 = _runtime_bundle_digest(contexts)
        manifest: dict[str, Any] = {
            "schema": RUNTIME_SOURCE_BUNDLE_SCHEMA,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "bundle_sha256": bundle_sha256,
            "contexts": contexts,
            "filters": {
                "all_contexts": list(_RUNTIME_IGNORED_NAMES),
                "opendde-src_additional": ["common"],
            },
        }
        atomic_write_json(temporary / RUNTIME_SOURCE_BUNDLE_MANIFEST, manifest)
        if target_root.exists():
            if target_root.is_dir() and not target_root.is_symlink():
                shutil.rmtree(target_root)
            else:
                target_root.unlink()
        temporary.replace(target_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return RuntimeSourceBundle(
        root=target_root,
        bundle_sha256=bundle_sha256,
        context_paths={name: target_root / name for name in RUNTIME_SOURCE_CONTEXTS},
        context_sha256={name: str(contexts[name]["sha256"]) for name in RUNTIME_SOURCE_CONTEXTS},
        manifest=manifest,
    )


def verify_runtime_source_bundle(bundle_root: Path) -> RuntimeSourceBundle:
    """Verify a bundle manifest and every BuildKit context before use."""

    root = bundle_root.expanduser().resolve()
    manifest_path = root / RUNTIME_SOURCE_BUNDLE_MANIFEST
    if not manifest_path.is_file():
        raise BackendError(f"runtime source bundle manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BackendError(
            f"invalid runtime source bundle manifest: {manifest_path}: {exc}"
        ) from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != RUNTIME_SOURCE_BUNDLE_SCHEMA:
        raise BackendError(
            f"unsupported runtime source bundle schema: {manifest.get('schema') if isinstance(manifest, dict) else None}"
        )
    contexts = manifest.get("contexts")
    if not isinstance(contexts, dict) or set(contexts) != set(RUNTIME_SOURCE_CONTEXTS):
        raise BackendError("runtime source bundle must contain exactly four named contexts")
    verified_contexts: dict[str, dict[str, Any]] = {}
    context_paths: dict[str, Path] = {}
    for name in RUNTIME_SOURCE_CONTEXTS:
        declared = contexts[name]
        if not isinstance(declared, dict) or declared.get("path") != name:
            raise BackendError(f"runtime source bundle context {name} has an invalid path")
        path = root / name
        actual = _hash_runtime_context(path)
        for field in ("sha256", "file_count", "size_bytes"):
            if declared.get(field) != actual[field]:
                raise BackendError(
                    f"runtime source bundle context {name} {field} mismatch: "
                    f"expected {declared.get(field)}, found {actual[field]}"
                )
        verified_contexts[name] = {**declared, **actual, "path": name}
        context_paths[name] = path
    actual_bundle_sha256 = _runtime_bundle_digest(verified_contexts)
    if manifest.get("bundle_sha256") != actual_bundle_sha256:
        raise BackendError(
            "runtime source bundle digest mismatch: "
            f"expected {manifest.get('bundle_sha256')}, found {actual_bundle_sha256}"
        )
    return RuntimeSourceBundle(
        root=root,
        bundle_sha256=actual_bundle_sha256,
        context_paths=context_paths,
        context_sha256={
            name: str(verified_contexts[name]["sha256"]) for name in RUNTIME_SOURCE_CONTEXTS
        },
        manifest=manifest,
    )


def prepare_runtime_build_contexts(config: AerithConfig) -> Path:
    """Stage a verified source bundle in the work directory for local builds."""

    target_root = (
        Path(config.project.work_dir).expanduser().resolve() / "runtime-build" / "contexts"
    )
    # This exact work path predates the portable bundle manifest, so it may be
    # an old unmanaged staging directory. It is disposable local build state.
    if target_root.exists():
        if target_root.is_dir() and not target_root.is_symlink():
            shutil.rmtree(target_root)
        else:
            target_root.unlink()
    return create_runtime_source_bundle(config, target_root).root


def _runtime_recipe_sha256(dockerfile: Path) -> str:
    repository_root = dockerfile.parents[2]
    recipe_paths = (
        dockerfile,
        dockerfile.parent / "entrypoint.sh",
        dockerfile.parent / "esm_if_batch.py",
        dockerfile.parent / "openfold-cuda-11.6.patch",
        repository_root / "docker" / "feature-builder" / "build_local_features.py",
        repository_root / "docker" / "feature-builder" / "mmseqs_wrapper.py",
        repository_root / "docker" / "feature-builder" / "convert_af3_templates.py",
    )
    digest = hashlib.sha256(b"aerith-runtime-recipe-v1\0")
    for path in recipe_paths:
        if not path.is_file():
            raise BackendError(f"runtime build recipe file does not exist: {path}")
        _update_hash_field(digest, path.relative_to(repository_root).as_posix())
        _update_hash_field(digest, path.read_bytes())
    return digest.hexdigest()


def build_runtime_image_command(
    config: AerithConfig,
    *,
    context_root: Path | None = None,
    source_bundle: Path | None = None,
) -> list[str]:
    """Build from local sources, staged contexts, or a verified source bundle."""

    dockerfile = Path(config.runtime.dockerfile).expanduser().resolve()
    if context_root is not None and source_bundle is not None:
        raise BackendError("context_root and source_bundle are mutually exclusive")
    verified_bundle: RuntimeSourceBundle | None = None
    if source_bundle is not None:
        verified_bundle = verify_runtime_source_bundle(source_bundle)
        sources = verified_bundle.context_paths
    elif context_root is None:
        sources = _runtime_source_paths(config)
    else:
        resolved_context_root = context_root.expanduser().resolve()
        if (resolved_context_root / RUNTIME_SOURCE_BUNDLE_MANIFEST).is_file():
            verified_bundle = verify_runtime_source_bundle(resolved_context_root)
            sources = verified_bundle.context_paths
        else:
            sources = {name: resolved_context_root / name for name in RUNTIME_SOURCE_CONTEXTS}
    if not dockerfile.is_file():
        raise BackendError(f"runtime Dockerfile does not exist: {dockerfile}")
    for name, source in sources.items():
        if not source.is_dir():
            raise BackendError(f"runtime build context {name} does not exist: {source}")
    free_gib = shutil.disk_usage(dockerfile.parent).free / (1024**3)
    if free_gib < config.runtime.minimum_build_free_gib:
        raise BackendError(
            f"runtime image build needs at least {config.runtime.minimum_build_free_gib} GiB "
            f"free; found {free_gib:.1f} GiB"
        )
    lock_root = dockerfile.parent / "locks"
    lock_paths = sorted(lock_root.glob("*.lock"))
    if not lock_paths:
        raise BackendError(f"runtime lock files do not exist: {lock_root}")
    lock_hash = hashlib.sha256(b"aerith-runtime-locks-v1\0")
    for path in lock_paths:
        _update_hash_field(lock_hash, path.name)
        _update_hash_field(lock_hash, path.read_bytes())
    recipe_sha256 = _runtime_recipe_sha256(dockerfile)
    command = [config.backend.docker_bin, "build", "--progress", "plain"]
    if config.runtime.build_add_host:
        command.extend(["--add-host", config.runtime.build_add_host])
    if config.runtime.build_proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            command.extend(["--build-arg", f"{name}={config.runtime.build_proxy}"])
        direct_hosts = (
            "localhost,127.0.0.1,archive.ubuntu.com,security.ubuntu.com,"
            "developer.download.nvidia.com,developer.download.nvidia.cn"
        )
        for name in ("NO_PROXY", "no_proxy"):
            command.extend(["--build-arg", f"{name}={direct_hosts}"])
    command.extend(
        [
            "--build-arg",
            f"OPENDDE_COMMIT={config.runtime.opendde_source_commit}",
            "--build-arg",
            f"ESM_COMMIT={config.runtime.esm_source_commit}",
            "--build-arg",
            f"MMSEQS_RELEASE={config.runtime.mmseqs_release}",
            "--build-arg",
            f"MMSEQS_VERSION={config.runtime.mmseqs_version}",
            "--build-arg",
            f"MMSEQS_SHA256={config.runtime.mmseqs_archive_sha256}",
            "--build-arg",
            f"FOLDSEEK_RELEASE={config.runtime.foldseek_release}",
            "--build-arg",
            f"FOLDSEEK_VERSION={config.runtime.foldseek_version}",
            "--build-arg",
            f"FOLDSEEK_SHA256={config.runtime.foldseek_archive_sha256}",
            "--build-arg",
            f"RUNTIME_LOCK_SHA256={lock_hash.hexdigest()}",
            "--build-arg",
            f"RUNTIME_RECIPE_SHA256={recipe_sha256}",
            "--build-arg",
            "RUNTIME_SOURCE_BUNDLE_SHA256="
            + (verified_bundle.bundle_sha256 if verified_bundle else "unavailable"),
        ]
    )
    source_hash_arguments = {
        "af3-src": "AF3_SOURCE_SHA256",
        "protenix-src": "PROTENIX_SOURCE_SHA256",
        "opendde-src": "OPENDDE_SOURCE_SHA256",
        "esm-src": "ESM_SOURCE_SHA256",
    }
    for context_name in RUNTIME_SOURCE_CONTEXTS:
        source_sha256 = (
            verified_bundle.context_sha256[context_name]
            if verified_bundle is not None
            else "unavailable"
        )
        command.extend(["--build-arg", f"{source_hash_arguments[context_name]}={source_sha256}"])
    for name, source in sources.items():
        command.extend(["--build-context", f"{name}={source}"])
    command.extend(
        [
            "--file",
            str(dockerfile),
            "--tag",
            config.backend.image,
            str(dockerfile.parents[2]),
        ]
    )
    return command


def build_backend_command(
    config: AerithConfig,
    *,
    input_dir: Path,
    output_dir: Path,
    gpu_index: int,
    feature_dir: Path | None = None,
    backend_settings: BackendSettings | None = None,
    template_mmcif_dir: Path | None = None,
    container_name: str | None = None,
) -> list[str]:
    backend = backend_settings or config.backend
    inputs = input_dir.resolve()
    outputs = output_dir.resolve()
    base = [
        backend.docker_bin,
        "run",
        "--rm",
    ]
    if container_name:
        base.extend(["--name", container_name])
    base.extend(
        [
            "--network",
            "none",
            "--gpus",
            f"device={gpu_index}",
            "--shm-size",
            "4g",
            "--volume",
            f"{inputs}:/inputs:ro",
            "--volume",
            f"{inputs}:{inputs}:ro",
            "--volume",
            f"{outputs}:/outputs",
        ]
    )
    if feature_dir is not None:
        feature_path = feature_dir.resolve()
        # Preserve absolute paths embedded in local-feature JSON.
        base.extend(["--volume", f"{feature_path}:{feature_path}:ro"])
    database = Path(config.features.database_dir).expanduser().resolve()
    base.append(backend.image)

    if backend.command:
        return base + [
            value.format(input_dir="/inputs", output_dir="/outputs", model=backend.model)
            for value in backend.command
        ]
    if backend.name == "alphafold3":
        model_dir = Path(backend.model_dir).expanduser().resolve()
        base[-1:-1] = [
            "--volume",
            f"{model_dir}:/root/models:ro",
            "--volume",
            f"{database}:{database}:ro",
        ]
        return base + [
            backend.runtime_entry,
            "--input_dir=/inputs",
            "--output_dir=/outputs",
            "--model_dir=/root/models",
            f"--db_dir={database}",
            "--norun_data_pipeline",
            "--gpu_device=0",
        ]
    input_json = f"/inputs/{backend.name}_jobs.json"
    use_template = template_mmcif_dir is not None
    if backend.name == "protenix":
        checkpoint_dir = Path(backend.checkpoint_dir or backend.model_dir).expanduser().resolve()
        common_dir = Path(backend.common_dir or backend.model_dir).expanduser().resolve()
        metadata_dir = Path(backend.metadata_dir or common_dir).expanduser().resolve()
        common_assets = (
            (common_dir / "components.cif", "components.cif"),
            (
                common_dir / "components.cif.rdkit_mol.pkl",
                "components.cif.rdkit_mol.pkl",
            ),
            (
                common_dir / "clusters-by-entity-40.txt",
                "clusters-by-entity-40.txt",
            ),
            (
                common_dir / "obsolete_release_date.csv",
                "obsolete_release_date.csv",
            ),
            (
                metadata_dir / "release_date_cache.json",
                "release_date_cache.json",
            ),
            (
                metadata_dir / "obsolete_to_successor.json",
                "obsolete_to_successor.json",
            ),
        )
        for source, _filename in common_assets:
            if not source.is_file():
                raise BackendError(f"Protenix common asset does not exist: {source}")
        base[-1:-1] = [
            "--env",
            "PROTENIX_ROOT_DIR=/protenix_data",
            "--volume",
            f"{checkpoint_dir}:/protenix_data/checkpoint:ro",
        ]
        # Mount files separately. Mounting common_dir read-only first and then
        # overlaying metadata beneath it makes runc fail while creating the
        # nested mountpoints.
        for source, filename in common_assets:
            base[-1:-1] = [
                "--volume",
                f"{source}:/protenix_data/common/{filename}:ro",
            ]
        if template_mmcif_dir is not None:
            base[-1:-1] = [
                "--volume",
                f"{template_mmcif_dir.resolve()}:/protenix_data/mmcif:ro",
            ]
        return base + [
            backend.runtime_entry,
            "pred",
            "-i",
            input_json,
            "-o",
            "/outputs",
            "-n",
            backend.model,
            "--use_msa",
            "true",
            "--use_template",
            "true" if use_template else "false",
            "--use_seeds_in_json",
            "true",
            "--need_atom_confidence",
            "true",
        ]
    if backend.name == "opendde":
        checkpoint = (
            Path(
                backend.checkpoint_path or "/home/structure/Software/OpenDDE/checkpoint/opendde.pt"
            )
            .expanduser()
            .resolve()
        )
        container_checkpoint = f"/opendde_data/checkpoint/{checkpoint.name}"
        common_dir = Path(backend.common_dir or backend.model_dir).expanduser().resolve()
        base[-1:-1] = [
            "--env",
            "OPENDDE_ROOT_DIR=/opendde_data",
            "--env",
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
            "--volume",
            f"{checkpoint}:{container_checkpoint}:ro",
            "--volume",
            f"{common_dir}:/opendde_data/common:ro",
        ]
        if template_mmcif_dir is not None:
            base[-1:-1] = [
                "--volume",
                f"{template_mmcif_dir.resolve()}:/opendde_data/search_database/mmcif:ro",
            ]
        return base + [
            backend.runtime_entry,
            "pred",
            "-i",
            input_json,
            "-o",
            "/outputs",
            "-n",
            backend.model,
            "--dtype",
            "bf16",
            "--load_checkpoint_path",
            container_checkpoint,
            "--use_msa",
            "true",
            "--use_template",
            "true" if use_template else "false",
            "--use_rna_msa",
            "false",
            "--need_atom_confidence",
            "true",
        ]
    raise BackendError(f"unsupported backend: {backend.name}")
