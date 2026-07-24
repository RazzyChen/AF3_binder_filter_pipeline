from __future__ import annotations

from pathlib import Path

import biotite.structure as struc
import biotite.structure.io as strucio
import numpy as np

from af3_binder_filter.af3_json import TargetFeatures
from af3_binder_filter.config import ConsensusSettings
from af3_binder_filter.consensus import add_anomaly_flags, structure_consensus_metrics
from af3_binder_filter.features import AF3FeatureBundle, FeatureBundle
from af3_binder_filter.secondary_features import (
    adapt_af3_features_for_secondary,
    adapt_local_features_for_secondary,
)


def _protein(
    path: Path,
    *,
    binder_shift: float = 0.0,
    missing_position: int | None = None,
) -> Path:
    residue_count = 25
    positions = [
        position for position in range(1, residue_count + 1) if position != missing_position
    ]
    array = struc.AtomArray(len(positions) * 2)
    array.coord = np.asarray(
        [[float(position - 1), 0.0, 0.0] for position in positions]
        + [[float(position - 1), 4.0 + binder_shift, 0.0] for position in positions]
    )
    array.chain_id = np.asarray(["A"] * len(positions) + ["B"] * len(positions))
    array.res_id = np.asarray(positions * 2)
    array.res_name = np.asarray(["ALA"] * len(positions) * 2)
    array.atom_name = np.asarray(["CA"] * len(positions) * 2)
    array.element = np.asarray(["C"] * len(positions) * 2)
    array.hetero = np.asarray([False] * len(positions) * 2)
    strucio.save_structure(str(path), array)
    return path


def test_af3_template_adapter_reuses_msa_disables_pairing_and_stages_template(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "af3"
    cache.mkdir()
    msa = cache / "target.a3m"
    msa.write_text(">query\n" + "A" * 25 + "\n")
    cif = _protein(cache / "1abc.cif")
    target_data = cache / "target_data.json"
    target_data.write_text("{}")
    af3 = AF3FeatureBundle(
        sequence_sha256="digest",
        cache_dir=cache,
        target_data_json=target_data,
        features=TargetFeatures(
            unpaired_msa_path=str(msa),
            paired_msa_path=None,
            templates=[
                {
                    "mmcifPath": str(cif),
                    "queryIndices": list(range(25)),
                    "templateIndices": list(range(25)),
                }
            ],
        ),
        fingerprint="af3-fingerprint",
    )

    adapted = adapt_af3_features_for_secondary(af3, "A" * 25)

    assert adapted.non_pairing_a3m != msa
    assert adapted.non_pairing_a3m.read_text() == msa.read_text()
    assert adapted.non_pairing_a3m.parent == adapted.cache_dir
    assert adapted.pairing_a3m is None
    assert adapted.templates_enabled is True
    assert adapted.template_count == 1
    assert adapted.hmmsearch_a3m is not None
    text = adapted.hmmsearch_a3m.read_text()
    assert ">query\n" + "A" * 25 in text
    assert any(adapted.template_mmcif_dir.glob("*.cif"))


def test_invalid_af3_templates_fall_back_to_msa_only(tmp_path: Path) -> None:
    cache = tmp_path / "af3"
    cache.mkdir()
    msa = cache / "target.a3m"
    msa.write_text(">query\nAAAA\n")
    cif = _protein(cache / "1abc.cif")
    target_data = cache / "target_data.json"
    target_data.write_text("{}")
    af3 = AF3FeatureBundle(
        "digest",
        cache,
        target_data,
        TargetFeatures(
            unpaired_msa_path=str(msa),
            templates=[{"mmcifPath": str(cif), "queryIndices": [], "templateIndices": []}],
        ),
        "fingerprint",
    )

    adapted = adapt_af3_features_for_secondary(af3, "AAAA")

    assert adapted.templates_enabled is False
    assert adapted.hmmsearch_a3m is None
    assert adapted.template_count == 0


def test_local_feature_adapter_stages_msa_and_reuses_hmmsearch_templates(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "features"
    cache.mkdir()
    pairing = cache / "pairing.a3m"
    non_pairing = cache / "non_pairing.a3m"
    hmmsearch = cache / "hmmsearch.a3m"
    pairing.write_text(">query\nAAAA\n")
    non_pairing.write_text(">query\nAAAA\n>homolog\nAAAA\n")
    hmmsearch.write_text(">query\nAAAA\n>1abc_A/1-4 mol:protein length:4\nAAAA\n")
    staged_templates = cache / "templates"
    staged_templates.mkdir()
    _protein(staged_templates / "1abc_A_1.cif")
    source_mmcif_dir = tmp_path / "mmcif"
    source_mmcif_dir.mkdir()
    _protein(source_mmcif_dir / "1abc.cif")
    template_json = cache / "af3_templates.json"
    template_json.write_text(
        """{
  "version": 1,
  "templates": [
    {
      "pdbId": "1abc",
      "authChainId": "A",
      "mmcifFile": "1abc_A_1.cif",
      "queryIndices": [0, 1, 2, 3],
      "templateIndices": [0, 1, 2, 3]
    }
  ]
}
"""
    )
    local = FeatureBundle(
        sequence_sha256="digest",
        cache_dir=cache,
        pairing_a3m=pairing,
        non_pairing_a3m=non_pairing,
        hmmsearch_a3m=hmmsearch,
        fingerprint="local-fingerprint",
        af3_templates_json=template_json,
        template_mmcif_dir=staged_templates,
        source_mmcif_dir=source_mmcif_dir,
    )

    adapted = adapt_local_features_for_secondary(local, "AAAA")

    assert adapted.non_pairing_a3m.parent == adapted.cache_dir
    assert adapted.non_pairing_a3m.read_text() == non_pairing.read_text()
    assert adapted.pairing_a3m is None
    assert adapted.templates_enabled is True
    assert adapted.template_count == 1
    assert adapted.template_mmcif_dir == source_mmcif_dir
    assert adapted.hmmsearch_a3m is not None
    assert ">1abc_A/1-4" in adapted.hmmsearch_a3m.read_text()


def test_consensus_uses_target_frame_and_separates_fold_from_pose(
    tmp_path: Path,
) -> None:
    primary = _protein(tmp_path / "primary.pdb")
    secondary = _protein(tmp_path / "secondary.pdb", binder_shift=2.0)
    contacts = frozenset(range(1, 11))

    metrics = structure_consensus_metrics(
        primary,
        secondary,
        target_chain="A",
        binder_chain="B",
        primary_target_contacts=contacts,
        secondary_target_contacts=contacts,
        primary_binder_contacts=contacts,
        secondary_binder_contacts=contacts,
        settings=ConsensusSettings(),
    )

    assert metrics["consensus_target_alignment_rmsd"] < 1e-6
    assert metrics["consensus_binder_fixed_frame_rmsd"] == 2.0
    assert metrics["consensus_binder_fold_rmsd"] < 1e-6
    assert metrics["consensus_binder_fold_tm"] > 0.99
    assert metrics["consensus_epitope_jaccard"] == 1.0


def test_raw_consensus_pairs_missing_residues_by_residue_position(
    tmp_path: Path,
) -> None:
    primary = _protein(tmp_path / "primary.pdb")
    secondary = _protein(
        tmp_path / "secondary_missing.pdb",
        binder_shift=2.0,
        missing_position=10,
    )
    contacts = frozenset(range(1, 10))

    metrics = structure_consensus_metrics(
        primary,
        secondary,
        target_chain="A",
        binder_chain="B",
        primary_target_contacts=contacts,
        secondary_target_contacts=contacts,
        primary_binder_contacts=contacts,
        secondary_binder_contacts=contacts,
        settings=ConsensusSettings(),
    )

    assert metrics["consensus_coordinate_source"] == "raw_structure"
    assert metrics["consensus_target_alignment_residues"] == 24
    assert metrics["consensus_target_alignment_rmsd"] < 1e-6
    assert metrics["consensus_binder_fixed_frame_rmsd"] == 2.0


def test_multimetric_robust_anomaly_is_review_only() -> None:
    rows = []
    for index in range(30):
        outlier = index == 29
        rows.append(
            {
                "job_name": str(index),
                "secondary_backend": "protenix",
                "consensus_status": "success",
                "consensus_binder_fixed_frame_rmsd": 10.0 if outlier else 1.0,
                "consensus_interface_fixed_frame_rmsd": 12.0 if outlier else 1.0,
                "consensus_binder_center_displacement": 1.0,
                "consensus_epitope_disagreement": 0.0,
                "consensus_fold_disagreement": 0.0,
                "primary_target_interface_residues": "1,2,3,4,5",
                "secondary_target_interface_residues": "1,2,3,4,5",
                "consensus_epitope_jaccard": 1.0,
                "candidate_pool": True,
            }
        )

    flagged = add_anomaly_flags(rows, ConsensusSettings())[-1]

    assert flagged["manual_review"] is True
    assert flagged["manual_review_reason"] == ("different_pose;robust_multimetric_anomaly")
    assert flagged["candidate_pool"] is True


def test_small_cohort_still_flags_explicit_fold_and_pose_disagreement() -> None:
    rows = [
        {
            "job_name": "different-fold",
            "secondary_backend": "opendde",
            "consensus_status": "success",
            "consensus_binder_fold_tm": 0.35,
            "consensus_binder_fixed_frame_rmsd": 24.0,
            "consensus_epitope_jaccard": 0.60,
            "primary_target_interface_residues": "1,2,3,4,5",
            "secondary_target_interface_residues": "1,2,3,4,5",
        },
        {
            "job_name": "different-pose",
            "secondary_backend": "opendde",
            "consensus_status": "success",
            "consensus_binder_fold_tm": 0.95,
            "consensus_binder_fixed_frame_rmsd": 6.0,
            "consensus_epitope_jaccard": 0.80,
            "primary_target_interface_residues": "1,2,3,4,5",
            "secondary_target_interface_residues": "1,2,3,4,5",
        },
    ]

    fold, pose = add_anomaly_flags(rows, ConsensusSettings())

    assert fold["consensus_different_binder_fold"] is True
    assert fold["consensus_different_pose"] is True
    assert fold["manual_review_reason"] == "different_binder_fold;different_pose"
    assert pose["consensus_different_binder_fold"] is False
    assert pose["consensus_different_pose"] is True
    assert pose["manual_review_reason"] == "different_pose"
