# AF3 Binder Filter Pipeline

Python/uv pipeline for AlphaFold 3 binder-target filtering. It reads a binder CSV,
builds AF3 target and complex JSON inputs, runs AF3 through Docker with GPU
sharding, scores best complex models with ESM inverse folding and ipSAE-style
metrics, and writes new CSV outputs without modifying the input CSV.

## Install

```bash
uv sync
```

The console entrypoint is:

```bash
uv run af3-binder-filter --help
```

`main.py` is only a compatibility wrapper around the package CLI.

## Local Validation

The local `tests/` directory is intentionally ignored by git. With local fixtures
present, run:

```bash
python -m compileall -q src main.py
python main.py check --csv tests/AF3_pipeline_dev_sample.csv
```

External AF3/ESM/GPU integration checks should be opt-in only. Do not run real
Docker AF3 or ESM scorer in default unit tests.

## Build Inputs

Create the target-only AF3 input:

```bash
python main.py make-target \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work \
  --force
```

After target AF3 has produced a target `*_data.json`, build complex inputs:

```bash
python main.py build-complex \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work \
  --target-data-json af_output/target_A/target_A_data.json \
  --limit 3 \
  --force
```

Each complex JSON is a single AF3 job object. Chain `A` uses externalized
target MSA/template paths under the complex input directory, and chain `B` has
empty MSA/template fields.

## Run AF3

Dry-run complex AF3 commands:

```bash
python main.py run-complex --work-dir work --output-dir af_output --dry-run
```

Run the full pipeline with an existing target data JSON:

```bash
python main.py pipeline \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work \
  --output-dir af_output \
  --target-data-json af_output/target_A/target_A_data.json
```

Without `--target-data-json`, `pipeline` runs `make-target` and `run-target`
first, then expects `af_output/target_A/target_A_data.json` before building
complex inputs.

## GPU Sharding

The runner queries:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
```

A GPU is busy when memory used is greater than `100 MiB` by default. Pending
jobs are sharded over free physical GPU indexes in ascending order. Docker uses
`--gpus device=<host_gpu>` and AF3 inside the container always receives
`--gpu_device=0`.

## Resumability

Subcommands skip existing successful outputs unless `--force` is passed.
Aggregation continues through missing or failed jobs and records status/error
columns for follow-up reruns.

## Outputs

Aggregation writes:

```text
aggregate_results.csv
input_with_af3_metrics.csv
best_models/
```

ESM and ipSAE scoring write summary CSVs under `work/scores/`, which
`aggregate` merges when present.

Successful AF3 jobs also receive SASA metrics during aggregation:

- `sasa_target_chain` and `sasa_binder_chain`: chain SASA in the predicted complex
- `sasa_target_free` and `sasa_binder_free`: isolated-chain SASA
- `dsasa` / `dsasa_interface`: buried interface SASA, computed as free total minus complex total

SASA uses Biotite and defaults to `--sasa-point-number 1000`.

`modin[ray]` is included in the project dependencies for larger post-processing
workloads that should use a pandas-compatible API backed by Ray.
