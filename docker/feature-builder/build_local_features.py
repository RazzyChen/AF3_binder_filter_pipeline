"""Offline-only feature-builder container entry point.

This wrapper deliberately has no HTTP client path. The image carries the
ColabFold search CLI, while AF3 flat padded MMseqs databases are searched
directly because they do not contain ColabFold expansion databases.
The resulting A3M then drives a local HMMER template search.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def _a3m_records(path: Path) -> list[tuple[str, str]]:
    records: list[tuple[str, str]] = []
    header: str | None = None
    sequence: list[str] = []
    for line in path.read_text().splitlines():
        line = line.replace("\x00", "")
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(sequence)))
            header = line
            sequence = []
        elif line.strip():
            sequence.append(line.strip())
    if header is not None:
        records.append((header, "".join(sequence)))
    return records


def _merge_a3ms(paths: list[Path], query: str, output_path: Path) -> None:
    seen_sequences = {query}
    merged = [(">query", query)]
    for path in paths:
        for header, sequence in _a3m_records(path):
            if sequence and sequence not in seen_sequences:
                seen_sequences.add(sequence)
                merged.append((header, sequence))
    output_path.write_text("".join(f"{header}\n{sequence}\n" for header, sequence in merged))


def _run_mmseqs_search(
    *,
    wrapper: str,
    query_db: Path,
    database: Path,
    output_root: Path,
    label: str,
    threads: int,
    iterations: int,
    split_memory_limit: str,
    use_gpu: bool,
    environment: dict[str, str],
) -> Path:
    result_db = output_root / f"{label}_result"
    temporary = output_root / f"{label}_tmp"
    output_a3m = output_root / f"{label}.a3m"
    search_command = [
        wrapper,
        "search",
        str(query_db),
        str(database),
        str(result_db),
        str(temporary),
        "--threads",
        str(threads),
        "--num-iterations",
        str(iterations),
        "--split-memory-limit",
        split_memory_limit,
        "-a",
        "-e",
        "0.1",
        "--max-seqs",
        "10000",
    ]
    if use_gpu:
        search_command.extend(["--gpu", "1"])
    subprocess.run(
        search_command,
        check=True,
        env=environment,
    )
    subprocess.run(
        [
            wrapper,
            "result2msa",
            str(query_db),
            str(database),
            str(result_db),
            str(output_a3m),
            "--msa-format-mode",
            "5",
            "--threads",
            str(threads),
        ],
        check=True,
        env=environment,
    )
    return output_a3m


def _aligned_fasta_from_a3m(a3m_path: Path, output_path: Path) -> None:
    # A few hundred representatives are enough for a robust query HMM and
    # avoid making hmmbuild scale with every hit in a deep target MSA.
    records = _a3m_records(a3m_path)[:300]
    output: list[str] = []
    for header, sequence in records:
        output.extend([header, re.sub(r"[a-z]", "", sequence.replace("\x00", ""))])
    output_path.write_text("\n".join(output) + "\n")


def _template_search(msa_path: Path, database: Path, output_path: Path) -> None:
    work = output_path.parent / "hmmsearch_work"
    work.mkdir(exist_ok=True)
    aligned_fasta = work / "query_alignment.fasta"
    hmm_path = work / "query.hmm"
    table_path = work / "hits.tbl"
    hits_fasta = work / "hits.fasta"
    _aligned_fasta_from_a3m(msa_path, aligned_fasta)
    subprocess.run(
        ["hmmbuild", "--informat", "afa", str(hmm_path), str(aligned_fasta)],
        check=True,
    )
    subprocess.run(
        [
            "hmmsearch",
            "--noali",
            "--tblout",
            str(table_path),
            "--F1",
            "0.1",
            "--F2",
            "0.1",
            "--F3",
            "0.1",
            "-E",
            "100",
            "--incE",
            "100",
            str(hmm_path),
            str(database),
        ],
        check=True,
    )
    hit_ids: list[str] = []
    for line in table_path.read_text().splitlines():
        if line and not line.startswith("#"):
            hit_ids.append(line.split()[0])
        if len(hit_ids) >= 100:
            break
    selected = set(hit_ids)
    found: list[str] = []
    current_id: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_id in selected and current_lines:
            found.append(f">{current_id}\n{''.join(current_lines)}\n")

    with database.open() as handle:
        for line in handle:
            if line.startswith(">"):
                flush()
                current_id = line[1:].split()[0]
                current_lines = []
            elif current_id in selected:
                current_lines.append(line.strip())
        flush()
    query = aligned_fasta.read_text().splitlines()[1]
    if not found:
        output_path.write_text(f">query\n{query}\n")
        shutil.rmtree(work)
        return
    af3_hits: list[str] = []
    for record in found:
        lines = record.splitlines()
        identifier = lines[0].removeprefix(">")
        sequence = "".join(lines[1:])
        af3_hits.append(
            f">{identifier}/1-{len(sequence)} mol:protein length:{len(sequence)}\n{sequence}\n"
        )
    hits_fasta.write_text("".join(af3_hits))
    aligned = subprocess.run(
        ["hmmalign", "--outformat", "A2M", str(hmm_path), str(hits_fasta)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    output_path.write_text(f">query\n{query}\n{aligned}")
    shutil.rmtree(work)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mmseqs-db", required=True)
    parser.add_argument("--pdb-seqres", required=True)
    parser.add_argument("--mmcif-dir", required=True)
    parser.add_argument("--mmseqs-binary", default="mmseqs")
    parser.add_argument("--use-gpu", type=int, choices=(0, 1), default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--split-memory-limit", default="32G")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--primary-database", default="uniref90_padded")
    parser.add_argument("--environment-database", default="mgnify_padded")
    parser.add_argument("--template-database", default="pdb_seqres_padded")
    parser.add_argument("--use-environment-database", type=int, choices=(0, 1), default=1)
    args = parser.parse_args()

    output = Path(args.output)
    search_output = output / "mmseqs_search"
    search_output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["AERITH_MMSEQS_BINARY"] = args.mmseqs_binary
    environment["AERITH_MMSEQS_SPLIT_MEMORY_LIMIT"] = args.split_memory_limit
    environment["AERITH_MMSEQS_NUM_ITERATIONS"] = str(args.iterations)
    wrapper = "/opt/aerith/mmseqs_wrapper.py"
    query_db = search_output / "query_db"
    subprocess.run(
        [
            wrapper,
            "createdb",
            args.query,
            str(query_db),
        ],
        check=True,
        env=environment,
    )
    msa_paths = [
        _run_mmseqs_search(
            wrapper=wrapper,
            query_db=query_db,
            database=Path(args.mmseqs_db) / args.primary_database,
            output_root=search_output,
            label="primary",
            threads=args.threads,
            iterations=args.iterations,
            split_memory_limit=args.split_memory_limit,
            use_gpu=bool(args.use_gpu),
            environment=environment,
        )
    ]
    if args.use_environment_database:
        msa_paths.append(
            _run_mmseqs_search(
                wrapper=wrapper,
                query_db=query_db,
                database=Path(args.mmseqs_db) / args.environment_database,
                output_root=search_output,
                label="environment",
                threads=args.threads,
                iterations=args.iterations,
                split_memory_limit=args.split_memory_limit,
                use_gpu=bool(args.use_gpu),
                environment=environment,
            )
        )
    query = Path(args.query).read_text().splitlines()[1].strip()
    _merge_a3ms(msa_paths, query, output / "non_pairing.a3m")
    (output / "pairing.a3m").write_text(f">query\n{query}\n")

    _template_search(
        output / "non_pairing.a3m",
        Path(args.pdb_seqres),
        output / "hmmsearch.a3m",
    )
    subprocess.run(
        [
            "/opt/envs/af3/bin/python",
            "/opt/aerith/convert_af3_templates.py",
            "--query",
            args.query,
            "--hmmsearch-a3m",
            str(output / "hmmsearch.a3m"),
            "--mmcif-dir",
            args.mmcif_dir,
            "--output-dir",
            str(output / "templates"),
            "--output-json",
            str(output / "af3_templates.json"),
        ],
        check=True,
    )
    shutil.rmtree(search_output)


if __name__ == "__main__":
    main()
