from pathlib import Path

from af3_binder_filter.config import ESMFoldConfig
from af3_binder_filter.esmfold_score import (
    build_esmfold_command,
    parse_esmfold_plddt,
    score_esmfold_inputs,
)
from af3_binder_filter.gpu import GPUInfo


def _atom_line(serial: int, residue: int, bfactor: float) -> str:
    return (
        f"ATOM  {serial:5d}  CA  ALA A{residue:4d}    "
        f"{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{bfactor:6.2f}           C\n"
    )


def test_parse_esmfold_plddt_uses_atom_b_factors(tmp_path):
    pdb_path = tmp_path / "model.pdb"
    pdb_path.write_text(
        "REMARK ignored\n" + _atom_line(1, 1, 80.0) + _atom_line(2, 2, 90.0) + "HETATM ignored\n"
    )

    assert parse_esmfold_plddt(pdb_path) == 85.0


def test_build_esmfold_command_includes_configured_options(tmp_path):
    config = ESMFoldConfig(
        conda_bin="conda",
        conda_env="esm",
        binary="esm-fold",
        model_dir=Path("/models"),
        num_recycles=2,
        max_tokens_per_batch=128,
        chunk_size=64,
        cpu_only=True,
    )

    command = build_esmfold_command(
        fasta_path=tmp_path / "input.fasta",
        pdb_dir=tmp_path / "pdb",
        config=config,
    )

    assert command[:5] == ["conda", "run", "-n", "esm", "esm-fold"]
    assert "--model-dir" in command
    assert "--num-recycles" in command
    assert "--max-tokens-per-batch" in command
    assert "--chunk-size" in command
    assert "--cpu-only" in command


def test_score_esmfold_dry_run_shards_only_free_gpus(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    for index in range(2):
        (input_dir / f"job_{index}.json").write_text(
            '{"name":"job_%d","sequences":[{"protein":{"id":"B","sequence":"ACDE"}}]}' % index
        )

    monkeypatch.setattr(
        "af3_binder_filter.esmfold_score.query_gpus",
        lambda: [
            GPUInfo(index=0, name="free", memory_used_mib=100, memory_total_mib=24000),
            GPUInfo(index=1, name="busy", memory_used_mib=101, memory_total_mib=24000),
        ],
    )

    rows = score_esmfold_inputs(
        input_dir=input_dir,
        score_dir=tmp_path / "scores",
        chain_id="B",
        config=ESMFoldConfig(),
        dry_run=True,
        force=True,
        gpu_busy_threshold_mib=100,
    )

    assert len(rows) == 2
    assert all(row["esmfold_status"] == "skipped" for row in rows)
    assert all("CUDA_VISIBLE_DEVICES=0" in row["esmfold_command"] for row in rows)
    assert (tmp_path / "scores" / "esmfold" / "shards" / "gpu_0" / "input.fasta").exists()
    assert not (tmp_path / "scores" / "esmfold" / "shards" / "gpu_1").exists()
