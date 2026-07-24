"""Convert a local HMMsearch A3M into AlphaFold 3 template JSON assets."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from alphafold3.constants import mmcif_names
from alphafold3.data import msa_config, structure_stores
from alphafold3.data import templates as templates_lib


def _without_query_record(a3m: str) -> str:
    records: list[str] = []
    current: list[str] = []
    for line in a3m.splitlines():
        if line.startswith(">") and current:
            records.append("\n".join(current) + "\n")
            current = []
        if line.strip():
            current.append(line)
    if current:
        records.append("\n".join(current) + "\n")
    return "".join(
        record
        for record in records
        if record.splitlines()[0].removeprefix(">").split()[0] != "query"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--hmmsearch-a3m", required=True)
    parser.add_argument("--mmcif-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-template-date", default="2021-09-30")
    parser.add_argument("--max-templates", type=int, default=4)
    args = parser.parse_args()

    query = "".join(Path(args.query).read_text().splitlines()[1:]).strip().upper()
    a3m = _without_query_record(Path(args.hmmsearch_a3m).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_json = Path(args.output_json)
    maximum_date = datetime.date.fromisoformat(args.max_template_date)

    if not a3m.strip():
        payload = {
            "version": 1,
            "max_template_date": args.max_template_date,
            "templates": [],
        }
        output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return

    filter_config = msa_config.TemplateFilterConfig(
        max_subsequence_ratio=0.95,
        min_align_ratio=0.1,
        min_hit_length=10,
        deduplicate_sequences=True,
        max_hits=args.max_templates,
        max_template_date=maximum_date,
    )
    templates = templates_lib.Templates.from_hmmsearch_a3m(
        query_sequence=query,
        a3m=a3m,
        max_template_date=maximum_date,
        structure_store=structure_stores.StructureStore(args.mmcif_dir),
        filter_config=filter_config,
        chain_poly_type=mmcif_names.PROTEIN_CHAIN,
    )

    records = []
    for index, (hit, structure) in enumerate(
        templates.get_hits_with_structures(),
        start=1,
    ):
        filename = f"{hit.pdb_id}_{hit.auth_chain_id}_{index}.cif"
        (output_dir / filename).write_text(structure.to_mmcif())
        mapping = hit.query_to_hit_mapping
        records.append(
            {
                "pdbId": hit.pdb_id,
                "authChainId": hit.auth_chain_id,
                "mmcifFile": filename,
                "queryIndices": list(mapping.keys()),
                "templateIndices": list(mapping.values()),
            }
        )
    payload = {
        "version": 1,
        "max_template_date": args.max_template_date,
        "templates": records,
    }
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
