# Aerith

Aerith is a Hydra-configured binder prediction and filtering pipeline. AlphaFold
3 is the fixed primary backend; one optional secondary backend (`protenix-v2` or
OpenDDE) cross-validates AF3 candidates. ESMFold and ESM-IF provide soft
sequence/fold annotations. Biotite, Rosetta InterfaceAnalyzer, Foldseek, and
target-contact fingerprints handle interface analysis and three-layer clustering.

Run all commands from the repository root. Production inference is containerized;
the host Python environment only provides the `aerith` orchestrator.

## Production quick start

The supported production path starts from the committed Dockerfile and lock files:

~~~bash
git switch Dev
uv venv --prompt aerith .venv
uv sync
source .venv/bin/activate

aerith config create \
  --output /ssd/aerith_screens/HER3/config.yaml \
  --project-root /ssd/aerith_screens/HER3 \
  --secondary-backend opendde \
  --gpu-ids 0,1,2

aerith config validate --config /ssd/aerith_screens/HER3/config.yaml
aerith build-runtime-image --config /ssd/aerith_screens/HER3/config.yaml
aerith config doctor --config /ssd/aerith_screens/HER3/config.yaml
aerith pipeline --config /ssd/aerith_screens/HER3/config.yaml --dry-run
aerith pipeline --config /ssd/aerith_screens/HER3/config.yaml
~~~

The default input is `/ssd/aerith_screens/HER3/input/screen.csv`. It must contain
`sample_no`, `run_name`, `binder_sequence`, and `target_seq`. All rows in one run
must share exactly one target sequence. Add `--csv PATH` or
`--epitope-residues 405,409,436` to `config create` when needed.

The production execution model is:

| Component | Production role | Isolation |
|---|---|---|
| AlphaFold 3 | Required primary fold backend | One GPU-isolated container per shard |
| Protenix-v2 or OpenDDE | Optional secondary cross-validation backend | Selected with `secondary_backend`; one GPU-isolated container per shard |
| GPU MMseqs2 + template search | Shared target feature preparation | Runs inside the same image; databases mounted read-only; network disabled |
| ESMFold + ESM-IF | Binder fold and sequence/backbone annotations | Dedicated image entrypoints using the ESM conda environment |
| Biotite + Rosetta | Interface geometry, epitope coverage, energy and packing | Host orchestration with deterministic Rosetta seed |
| GPU Foldseek + contact fingerprints | Binder-fold, complex-pose and epitope clustering | Image tool plus deterministic contact-set clustering |

AF3 is always the primary backend. The optional secondary backend reuses AF3's
target MSA/templates, while de novo Binder chains intentionally remain query-only
with no paired MSA or templates. A secondary result is cross-validation evidence,
not a vote requiring both backends to pass: the exact candidate and consensus rules
are documented below and in [SHORTLIST_COLUMNS.md](SHORTLIST_COLUMNS.md).

## Current validation status

- Automated suite: 81 tests passing; source, scripts and Docker helper modules pass
  `compileall`; the Git diff passes whitespace validation.
- Runtime image: `aerith/fold-runtime:local`, built reproducibly from
  `docker/runtime/Dockerfile` with pinned dependency locks and verified GPU
  MMseqs2/Foldseek archives.
- Real HER3 acceptance: 119/119 AF3 predictions, 72/72 gated OpenDDE
  predictions, 72 candidates and 44 final diversity representatives.
- Recovery verification: two consecutive recoveries produced byte-identical
  `all_results.csv`, `candidates.csv`, and `final_shortlist.csv`; Rosetta uses a
  fixed seed and ESM cache reuse validates full job/artifact coverage.
- Public output contract: exactly three run-root CSV files with the same 83-column
  schema; stage logs, tables and artifacts live under `stages/01_*` through
  `stages/10_*`.

Read [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the architecture and
acceptance contract. Read [SHORTLIST_COLUMNS.md](SHORTLIST_COLUMNS.md) before
interpreting shortlist metrics or changing ranking thresholds.

## Hydra workflow

Create a non-interactive minimal configuration for a production screen. The
CSV defaults to `<project-root>/input/screen.csv`:

~~~bash
aerith config create \
  --output /ssd/aerith_screens/HER3/config.yaml \
  --project-root /ssd/aerith_screens/HER3 \
  --secondary-backend opendde \
  --gpu-ids 0,1,2

aerith config validate --config /ssd/aerith_screens/HER3/config.yaml
aerith config doctor --config /ssd/aerith_screens/HER3/config.yaml
aerith config show --config /ssd/aerith_screens/HER3/config.yaml
~~~

Use `--csv PATH` when the CSV is not at the default location, and
`--epitope-residues 25-35,42,57` when a reference epitope is available. Omit
`--gpu-ids` to let Aerith use all currently available GPUs. Existing YAML files
are protected unless `--force` is supplied.

The resulting YAML contains only screen-specific decisions; the Structured
Config schema supplies stable defaults:

~~~yaml
defaults:
  - backend: alphafold3
  - secondary_backend: opendde
  - features: local_af3_db
  - interface: biotite_rosetta
  - clustering: balanced
  - _self_

project:
  csv_path: /ssd/aerith_screens/HER3/input/screen.csv
  work_dir: /ssd/aerith_screens/HER3/work
  output_dir: /ssd/aerith_screens/HER3/outputs
  results_dir: /ssd/aerith_screens/HER3/results

runtime:
  gpu_ids: [0, 1, 2]

interface:
  epitope_residues: null

hydra:
  searchpath:
    - pkg://af3_binder_filter.conf
  job:
    chdir: false
~~~

For a fully expanded host snapshot with automatic GPU, Docker, database, and
Rosetta detection, keep using the separate initialization command:

~~~bash
aerith config init --output config.full.yaml
~~~

Run AF3 only, or choose one secondary backend:

~~~bash
aerith pipeline --config config.yaml
aerith pipeline --config config.yaml --secondary-backend protenix
aerith pipeline --config config.yaml --secondary-backend opendde \
  --override interface.distance=4.5

aerith prepare-features --config config.yaml
aerith analyze-interface --config config.yaml
aerith cluster --config config.yaml
~~~

For the current 300-design production screen, run from the screen directory:

~~~bash
cd /ssd/CSX/Jun5_bindcraft_design/424_425_426_binder_filter
aerith config validate --config config.yaml
aerith config doctor --config config.yaml
aerith pipeline --config config.yaml --secondary-backend opendde
~~~

The configured GPUs are isolated per container with `docker run --gpus
device=<host-gpu>`. AF3 is always primary; the secondary backend starts only
for jobs that pass the AF3 confidence gate and reuses AF3's target MSA and
template features.

`--backend` is retained only as an explicit primary override and must remain
`alphafold3`. `--secondary-backend` selects `none`, `protenix`, or `opendde`.
`--override` is repeatable. Hydra is composed through its Python API and never
changes the process working directory.

OpenDDE uses the general-purpose `opendde.pt` checkpoint by default. The
antibody-antigen-tuned weights remain an explicit opt-in:

~~~bash
aerith pipeline --config config.yaml --secondary-backend opendde \
  --override secondary_backend.checkpoint_path=/home/structure/Software/OpenDDE/checkpoint/opendde_abag.pt
~~~

Every run is fingerprinted from sequences, chains, both backend selections,
models, seeds, AF3 features, ESM/consensus settings, source revisions, and image
IDs. Output is reused only when its manifest matches and its adapter can parse
the artifacts. A changed sequence, backend, checkpoint, feature, or image cannot
silently reuse an older result.

~~~text
results/run-<fingerprint>/
├── resolved_config.yaml
├── manifest.json
├── all_results.csv
├── candidates.csv
├── final_shortlist.csv
└── stages/
    ├── 01_preflight/{logs,tables,artifacts}/
    ├── 02_features/{logs,tables,artifacts}/
    ├── 03_primary_prediction/{logs,tables,artifacts}/
    ├── 04_primary_interface/{logs,tables,artifacts}/
    ├── 05_esm/{logs,tables,artifacts}/
    ├── 06_secondary_features/{logs,tables,artifacts}/
    ├── 07_secondary_prediction/{logs,tables,artifacts}/
    ├── 08_secondary_interface/{logs,tables,artifacts}/
    ├── 09_consensus/{logs,tables,artifacts}/
    └── 10_clustering/{logs,tables,artifacts}/
~~~

These are the only three CSV files at the run root, and they always have the
same ordered schema:

- `all_results.csv` has one row for every planned input job, including failed
  jobs and empty metrics for unavailable stages.
- `candidates.csv` contains exactly rows with `candidate_pass=true`, enriched
  with Binder-fold, complex-pose, target-contact and diversity-cell cluster IDs.
- `final_shortlist.csv` contains one quality representative per unique
  Binder/pose/epitope diversity cell.

Detailed, wide tables remain under each stage's `tables/` directory. Commands,
stdout and stderr are under that stage's `logs/`; Rosetta inputs/scores, ESM
models, Foldseek files, cluster TSVs and representative FASTA files are under
`artifacts/`. Raw backend predictions remain in
`outputs/<run_id>/<backend>/` and are not duplicated into results.

Public residue fields use input-sequence 1-based positions and always include
the chain: target residues `A:424;A:425`, Binder residues `B:76;B:80`, and
contact pairs `A:424-B:76;A:425-B:80`. The manifest records
`output_schema_version: 2`.

Biotite geometry always runs before Rosetta for both prediction backends. A
Rosetta failure or timeout is recorded without discarding valid geometry.
Enabled-stage failures preserve partial outputs and return non-zero unless
`project.allow_partial=true`.

When a reference epitope is configured, the hard epitope gate uses coverage:
`overlapping target residues / configured epitope residues`.  With three
configured residues, one hit is therefore `1/3 = 0.333` and passes the default
`minimum_epitope_coverage: 0.30`.  Interface purity is still reported as
`overlap / all target interface residues` in detailed interface tables, but it
is absent from the public CSV schema and never filters or ranks candidates. An
old `minimum_epitope_purity` YAML value is accepted for compatibility but
ignored.

The in-image GPU MMseqs2 feature builder is the default feature source. Its
target MSA and AF3-native template mappings are cached by target-sequence
SHA-256. AF3 reads the external A3M/mmCIF assets with
`--norun_data_pipeline`, so it cannot fall back to Jackhmmer. The secondary
adapter reuses the same target unpaired MSA and up to four staged templates;
Binder B always uses query-only unpaired MSA, no paired MSA, and no templates.
Runtime containers use `--network none`.

The secondary gate requires a fingerprint-valid AF3 confidence result with
`ipTM >= 0.70`. It deliberately does not require a valid AF3 CIF, interface, or
epitope result, so Protenix/OpenDDE can rescue an AF3 structure failure. In dual
mode the candidate pool requires secondary success and a geometry/epitope pass
from either backend. Consensus disagreement is continuous ranking/annotation
data and can only add a row to
`stages/09_consensus/tables/manual_review.csv`; it is not a hard filter.
A Binder fold TM-score below `consensus.same_fold_tm_threshold`, or a
target-frame Binder RMSD above `consensus.different_pose_rmsd_threshold`
when the fold is otherwise the same, is flagged deterministically even when
the cohort is too small for robust-z anomaly detection.

## Unified runtime image

AF3 (uv), OpenDDE (uv), Protenix (conda), ESM (conda), and the offline
MSA/template toolchain are isolated inside one CUDA 12.6.3 image. The pinned
GPU-enabled MMseqs2 build (`8cc5ce…`), patched HMMER 3.4, Kalign,
ColabFold search, and the local feature adapter share the image without adding
another Torch environment. MMseqs searches padded databases with `--gpu 1`
on one selected GPU and do not silently fall back to CPU. Each stage starts a
separate container, so Python and torch/JAX dependencies do not leak between
tools. Checkpoints, databases, features, model cache, inputs, and outputs remain
host mounts; weights and large databases are not baked into the image.

### Build from the Dockerfile

The reproducible build consumes four explicit, source-only BuildKit contexts.
The local conda/uv environments are not copied into the image: dependencies are
re-created from the committed locks in `docker/runtime/locks`.

Required source locations in the default configuration:

~~~text
/home/structure/Software/alphafold3-3.0.3
/home/structure/Software/Protenix-2.0.0
/home/structure/Software/OpenDDE
/home/structure/Software/esm
~~~

[MMseqs2 18-8cc5c GPU](https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-gpu.tar.gz)
and [Foldseek 10-941cd33 GPU](https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-gpu.tar.gz)
are downloaded from their official release assets during the build and verified
with pinned SHA-256 checksums. Neither tool is taken from the host filesystem.

The recommended build stages filtered contexts, verifies source commits and the
GPU MMseqs2 checksum, then invokes the Dockerfile:

~~~bash
aerith config validate --config config.yaml
aerith build-runtime-image --config config.yaml --dry-run
aerith build-runtime-image --config config.yaml
~~~

The equivalent raw Docker command, run from this repository root, is:

~~~bash
DOCKER_BUILDKIT=1 docker build --progress plain \
  --build-context af3-src=/home/structure/Software/alphafold3-3.0.3 \
  --build-context protenix-src=/home/structure/Software/Protenix-2.0.0 \
  --build-context opendde-src=/home/structure/Software/OpenDDE \
  --build-context esm-src=/home/structure/Software/esm \
  --file docker/runtime/Dockerfile \
  --tag aerith/fold-runtime:local \
  .
~~~

When downloads require the local V2Ray endpoint on `127.0.0.1:8889`, expose it
only through a host address reachable by rootless BuildKit, then set
`runtime.build_proxy` to that relay URL. Do not put proxy settings into the
finished image. `runtime.build_add_host` is available when a stable host alias
is preferred.

The build stages source-only copies under `work/runtime-build/contexts`, excluding
local `.venv`, conda environments, checkpoints, and prediction outputs. The
image dispatcher exposes `af3`, `protenix`, `opendde`, `esmfold`, `esm-if`,
and `prepare-features`. Feature preparation mounts `/data/AF3_database`
read-only and runs with `--network none`. OpenDDE always receives the explicit
checkpoint selected by Hydra; the packaged default is the general-purpose
`opendde.pt`.

Verify both the image identity and all isolated environments after building:

~~~bash
docker image inspect aerith/fold-runtime:local
docker run --rm --gpus all --network none \
  aerith/fold-runtime:local doctor
aerith config doctor --config config.yaml
~~~

### Docker storage on this host

The rootless Docker daemon is configured in
`/home/structure/.config/docker/daemon.json` with
`data-root=/ssd/docker-rootless` and `storage-driver=fuse-overlayfs`.
The previous Docker data root was removed only after the new daemon and image
were verified. Confirm the active location before a large rebuild:

~~~bash
docker info --format '{{.DockerRootDir}} {{.Driver}}'
~~~

The expected result on this workstation is:

~~~text
/ssd/docker-rootless fuse-overlayfs
~~~

This is daemon storage configuration, not part of the Dockerfile. Model
checkpoints, databases, run inputs, and outputs remain explicit read-only or
read-write mounts and are never copied into the image layer.

At runtime Aerith mounts these host assets instead of copying them:

- `/data/AF3_database` and model/checkpoint directories read-only;
- cached target MSA/templates read-only for prediction containers;
- per-shard inputs read-only and output directories read-write;
- ESM model cache read-only.

### Multi-GPU Docker execution

`aerith pipeline` discovers GPU memory usage at the beginning of every GPU
stage. `runtime.gpu_ids` is an allow-list; an empty list means all discovered
GPUs are eligible. A device with memory usage above
`runtime.gpu_busy_threshold_mib` is excluded.

~~~yaml
runtime:
  gpu_ids: [0, 1, 2]
  gpu_busy_threshold_mib: 100
~~~

Pending jobs are assigned deterministically and round-robin to the free allowed
GPUs. Aerith starts one Docker container per selected physical GPU:

~~~text
host GPU 0 -> docker --gpus device=0 -> backend-visible GPU 0
host GPU 1 -> docker --gpus device=1 -> backend-visible GPU 0
host GPU 2 -> docker --gpus device=2 -> backend-visible GPU 0
~~~

AF3 shards run concurrently. After the primary stage completes, eligible jobs
are independently re-sharded for Protenix or OpenDDE. ESMFold and ESM-IF use the
same isolation policy; MSA/template preparation uses one free GPU because there
is only one shared target. Each shard has its own input directory, container
name, command record, stdout log, and stderr log. The run manifest records the
physical GPU-to-job assignment.

Stages retain their dependency barriers: a secondary backend never starts
before AF3 confidence gating, and interface/consensus analysis never reads a
still-running prediction. A shard failure does not discard artifacts produced
by other shards, but the pipeline exits nonzero unless
`project.allow_partial=true`.

Inspect all generated container commands without running inference:

~~~bash
aerith pipeline --config config.yaml --dry-run
~~~

Run the real ten-row backend acceptance CSV with separate, auditable run IDs:

~~~bash
TEST_PARENT=/ssd/CSX/Jun5_bindcraft_design/424_425_426_binder_filter
TEST_ROOT=${TEST_PARENT}/aerith_backend_test

aerith pipeline --config config.yaml \
  --secondary-backend protenix \
  --override project.csv_path=${TEST_PARENT}/backend_test_file.csv \
  --override project.work_dir=${TEST_ROOT}/work \
  --override project.output_dir=${TEST_ROOT}/outputs \
  --override project.results_dir=${TEST_ROOT}/results \
  --override project.run_id=backend-test-protenix-gpu-features

aerith pipeline --config config.yaml \
  --secondary-backend opendde \
  --override project.csv_path=${TEST_PARENT}/backend_test_file.csv \
  --override project.work_dir=${TEST_ROOT}/work \
  --override project.output_dir=${TEST_ROOT}/outputs \
  --override project.results_dir=${TEST_ROOT}/results \
  --override project.run_id=backend-test-opendde-gpu-features
~~~

Both commands use the target feature cache under
`${TEST_ROOT}/work/features/<target-sequence-sha256>/`. The primary and
secondary prediction manifests remain separate, so an output from one backend
combination cannot be adopted by the other merely because the job name is the
same.

### Real-data acceptance result (2026-07-13)

The commands above were run against all ten rows, not a synthetic fixture. The
validated unified image was
`sha256:f1b4007199272574f62bee9702e66a42feac9522a32bea1c74101d5d7333cbec`
(32.6 GB). It was the only image present in the active rootless Docker daemon.
This historical OpenDDE acceptance run used `opendde_abag.pt`; the packaged
default was subsequently changed to `opendde.pt`. Checkpoint identity is part
of the run and per-job fingerprints, so the general model cannot reuse those
ABAG prediction artifacts.

The shared 602-residue target produced a GPU MMseqs2 A3M with 13,312 raw
records; inference retained 13,277 MSA rows. Four local target templates were
selected. Every AF3 log reported `Skipping data pipeline...`, every target
chain reported four templates in Protenix/OpenDDE, and every Binder chain
reported zero templates. All prediction containers ran with
`--network none`.

Both result manifests finished with every required stage marked `success`:

- `${TEST_ROOT}/results/backend-test-protenix-gpu-features`: AF3 10/10,
  Protenix 10/10, primary/secondary Biotite 10/10, primary/secondary Rosetta
  10/10, consensus 10/10, ESM and all three clustering layers successful.
- `${TEST_ROOT}/results/backend-test-opendde-gpu-features`: the same 10/10
  stage counts for AF3 plus OpenDDE.

On the three RTX 3090 workers, Protenix-v2 model forward time was approximately
125–136 seconds per complex. OpenDDE `bf16` took approximately 219–242
seconds per complex. Observed peak memory remained below 24 GB. GPU 3 was busy
with an unrelated process and was excluded throughout by the configured busy
threshold.

The explicit cross-backend review logic found no deterministic disagreement in
the Protenix run. In the OpenDDE run, sample 6 is retained in the candidate
pool but written to `stages/09_consensus/tables/manual_review.csv` with
`manual_review_reason=different_binder_fold`: Binder fold TM-score was about
0.355 and target-frame Binder RMSD was about 24.74 Å. This is an annotation,
not a hard filter.

After correcting the secondary job fingerprint to include the actual
secondary feature bundle and secondary JobSpec, both completed runs were
resumed with 10 reusable and zero pending secondary jobs. Their final
manifests therefore contain no `secondary_prediction` GPU assignment for the
resume pass.

The detailed, reviewed architecture and acceptance criteria are in
[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md).

### 300-design production acceptance (2026-07-16)

The current production command was run from
`/ssd/CSX/Jun5_bindcraft_design/424_425_426_binder_filter` after deleting all
regenerable feature, prediction and result directories. The run therefore
rebuilt the 602-residue target features from the local GPU MMseqs2 databases;
the validated bundle contains an 8.5 MB target MSA and four local templates.

Run `run-a817f3c5a90b` completed with exit code zero and every manifest stage
marked `success`, with no manifest errors:

- AF3 prediction, primary Biotite, primary Rosetta, ESMFold and ESM-IF:
  300/300 successful.
- AF3-gated OpenDDE (`opendde.pt`), secondary Biotite and secondary Rosetta:
  237/237 successful.
- Public rows: 300 `all_results`, 37 `candidates`, and 28
  `final_shortlist` representatives.
- Candidate diversity: 23 Binder-fold clusters, 28 complex-pose clusters and
  one target-contact epitope cluster; every final row has a unique three-layer
  diversity cell.
- All public residue/contact values passed chain-qualified format validation;
  the three CSV headers are identical and contain no purity column.

The accepted outputs are under
`results/run-a817f3c5a90b/` relative to that production directory.

### 119-design HER3 production acceptance (2026-07-17; recovery verified 2026-07-21)

The unified image was rebuilt from `docker/runtime/Dockerfile` as
`aerith/fold-runtime:local` with immutable image ID
`sha256:15abac4ce089426f33301b36898d2ffee2dead01783a6cc82fb27c8c6024b901`.
The in-image doctor passed AF3, OpenDDE CUDA, Protenix CUDA, ESM CUDA, GPU
MMseqs2 and GPU Foldseek checks on an RTX 3090.

The production command was run from
`/ssd/CSX/Jun5_bindcraft_design/424_428_455_binder_filter`:

```bash
aerith pipeline --config config.yaml --secondary-backend opendde
```

The input contained 119 unique Binders against one 624-residue HER3 ECD
target, with reference epitope positions `405,409,436`. The first one-job
smoke test built the reusable target feature cache with 13,751 unpaired MSA
records, 101 HMM/template-search records and four local target templates.
Binder chains remained query-only, with no paired MSA and no templates.

Run `run-ee29bd9dd241` completed in 7,991 seconds with exit code zero, no
manifest errors and every enabled stage marked `success`:

- AF3, primary Biotite, primary Rosetta, ESMFold and ESM-IF: 119/119.
- The AF3 `ipTM >= 0.70` gate selected 72/119 designs.
- General-model OpenDDE (`opendde.pt`), secondary Biotite and secondary
  Rosetta: 72/72.
- Consensus: 72 successful comparisons and 47 explicit `not_available` rows
  for designs that did not enter the secondary backend.
- Public rows: 119 `all_results`, 72 `candidates` and 44
  `final_shortlist` representatives.
- Candidate diversity: 42 Binder-fold clusters, 44 complex-pose clusters,
  two target-contact epitope clusters and 44 unique diversity cells.
- Seven candidates were annotated for manual review; these annotations did
  not hard-filter candidates.

All three public CSV files have the same schema. Every target residue, Binder
residue and residue pair passed chain-qualified formatting checks such as
`A:405`, `B:15` and `A:405-B:15`. The run used 3.8 GB under `outputs/`,
55 MB under its result directory and left no running containers or GPU
workloads; all four GPUs returned to the approximately 2 MiB idle baseline.

The accepted outputs are under:

```text
/ssd/CSX/Jun5_bindcraft_design/424_428_455_binder_filter/results/run-ee29bd9dd241/
```

Recovery acceptance also exposed and fixed two reproducibility issues. Rosetta
InterfaceAnalyzer now defaults to `constant_seed: true` and
`random_seed: 1111111`, so separated-state repacking and packstat are
deterministic. A same-run ESM table is reused only when all expected jobs,
finite ESM-IF scores and ESMFold PDB artifacts validate; otherwise the entire
ESM stage is recomputed.

The final recovery reported feature cache hit 1/1, AF3 cache hit 119/119, ESM
cache hit 119/119 and OpenDDE cache hit 72/72:

- `all_results.csv`: `be682d17c3b51c79aad4c1ba535797fae05f5e7ab291253f48b5c3d72cd0ad14`
- `candidates.csv`: `61c3bd86933545b36b691afd3ba3c802be82492e91b4bea51a6839eed5a3377e`
- `final_shortlist.csv`: `2b402de8a233980481dad6f4dcf2f8211c984843a16d4ac518a7c7380fea259b`

The direction, unit, interpretation and caveats for all 83 public columns are
documented in [SHORTLIST_COLUMNS.md](SHORTLIST_COLUMNS.md), including the exact
candidate gate, representative selection and final-rank ordering.

## Legacy stage commands

The original AF3/ESM/ESMFold/ipSAE stage commands remain available for existing
workflows. New runs should use the Hydra commands above.

## Environment

Create the uv environment with the Aerith prompt, then sync dependencies:

```bash
uv venv --prompt aerith .venv
uv sync
```

Activate the project environment:

```bash
while [[ -n "${CONDA_PREFIX:-}" ]]; do conda deactivate; done
source .venv/bin/activate
```

The package entrypoint is:

```bash
aerith --help
```

Typical full run:

```bash
uv venv --prompt aerith .venv
uv sync
source .venv/bin/activate
aerith pipeline --config config.yaml
```

`main.py` remains a compatibility wrapper around the same package CLI:

```bash
python main.py --help
```

The ESM inverse-folding scorer and ESMFold are called through the configured
conda environment:

```text
conda run -n esm python /home/structure/Software/esm/examples/inverse_folding/score_log_likelihoods.py
```

## Local Files

Local generated inputs under `tests/`, `goal.md`, `.venv/`, `work/`, and
`af_output/` are ignored by git. Committed unit tests may still live under
`tests/`. The sample CSV and fixture AF3 JSON/CIF files can be used for local
validation, but they are intentionally not versioned here.

Default local CSV:

```text
tests/AF3_pipeline_dev_sample.csv
```

Required CSV columns:

```text
sample_no, run_name, binder_sequence, target_seq
```

The target sequence is read from `target_seq`; each binder sequence is read from
`binder_sequence`.

## Preflight Check

The startup check is intentionally narrow. It only validates the CSV path,
readability, schema, and parsed job count:

```bash
aerith check --csv tests/AF3_pipeline_dev_sample.csv
```

Expected output on the cluster:

```text
CSV OK: tests/AF3_pipeline_dev_sample.csv (5 jobs)
```

Docker, conda, ESM, GPU, and cache checks are left to the subcommands that need
them.

## Target AF3 Input

The first target-only AF3 run is used only to make AF3 compute target MSA and
template features. Its JSON is AF3 dialect version `1`, with a list-valued chain
ID and no explicit MSA/template fields:

```bash
aerith make-target \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work \
  --name target_A \
  --force
```

This writes:

```text
work/target_input/target_A.json
```

Shape:

```json
{
  "name": "target_A",
  "sequences": [
    {
      "protein": {
        "id": ["A"],
        "sequence": "..."
      }
    }
  ],
  "modelSeeds": [42],
  "dialect": "alphafold3",
  "version": 1
}
```

Run target AF3:

```bash
aerith run-target --work-dir work --output-dir af_output
```

After this target run, the pipeline accepts either target data layout:

```text
af_output/target_A_data.json
af_output/target_A/target_A_data.json
```

That `*_data.json` must contain the target chain MSA/template data generated by
AF3.

## Complex AF3 Inputs

Build binder-target complex inputs after the target `*_data.json` exists:

```bash
aerith build-complex \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work \
  --target-data-json af_output/target_A_data.json \
  --force
```

Default complex job names are:

```text
sample_{sample_no}_binder_candiate_complex_pred
```

For the sample CSV, this writes:

```text
work/complex_inputs/sample_1_binder_candiate_complex_pred.json
work/complex_inputs/sample_2_binder_candiate_complex_pred.json
...
```

For a quick construction from the local fixture:

```bash
aerith build-complex \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work/fixture \
  --limit 1 \
  --force \
  --target-data-json tests/ref/af_output/sample_1_PD1_binder_candiate_complex_pred/sample_1_PD1_binder_candiate_complex_pred_data.json \
  --job-name-template "sample_{sample_no}_PD1_binder_candiate_complex_pred"
```

Each complex JSON is a single AF3 job object under:

```text
work/complex_inputs/
```

Chain `A` is the target. Its MSA/template content is externalized beside the
complex JSON:

```text
work/complex_inputs/msas/
work/complex_inputs/templates/
```

Chain `B` is the binder. It intentionally has no `unpairedMsa`, `pairedMsa`,
`unpairedMsaPath`, or `pairedMsaPath` fields, so AF3 can build/search binder MSA.
It also has no templates:

```json
{
  "protein": {
    "id": "B",
    "sequence": "...",
    "modifications": [],
    "templates": []
  }
}
```

## Run Complex AF3

Dry-run the Docker commands:

```bash
aerith run-complex --work-dir work --output-dir af_output --dry-run
```

Run pending complex AF3 jobs:

```bash
aerith run-complex --work-dir work --output-dir af_output
```

GPU assignment uses:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits
```

A GPU is treated as busy when used memory is greater than `100 MiB`. Jobs are
sharded only over free physical GPU IDs. Docker receives
`--gpus device=<host_gpu>`, while AF3 inside the container receives
`--gpu_device=0`.

## Scoring

ESM scoring creates a temporary FASTA from the AF3 complex input JSON, not from
the CSV. The FASTA header is:

```text
>job_name|chain_B
```

Run ESM on the best CIF for each job:

```bash
aerith score-esm \
  --work-dir work \
  --output-dir af_output \
  --no-ray
```

Run ESMFold on the design chain and report its mean pLDDT:

```bash
aerith score-esmfold \
  --work-dir work \
  --output-dir af_output \
  --no-ray
```

Run ipSAE-style metrics:

```bash
aerith score-ipsae \
  --work-dir work \
  --output-dir af_output \
  --no-ray
```

Ray is enabled by default for `score-esm`, `score-esmfold`, and `score-ipsae`;
pass `--no-ray` for serial local validation. `modin[ray]` is included for
pandas-compatible large table work backed by Ray.

## Aggregate

Aggregate AF3, ESM, ESMFold, sequence pI, ipSAE, and SASA/BSA metrics:

```bash
aerith aggregate \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --output-dir af_output \
  --results-dir results \
  --score-dir work/scores
```

Outputs:

```text
results/aggregate_results.csv
results/input_with_af3_metrics.csv
results/candiate.csv
results/best_models/
```

Aggregation does not modify the input CSV. Missing or failed jobs stay in the
result table with status/error columns so they can be rerun.

Main AF3 fields include:

```text
sample_no, run_name, job_status, job_name, best_seed, best_sample, best_model_path
ranking_score, iptm, ptm, fraction_disordered, has_clash
chain_iptm, chain_pair_iptm, chain_pair_pae_min, chain_ptm
plddt_global_mean, normalized_plddt_global_mean
plddt_global_min, plddt_chain_A_mean, plddt_chain_B_mean
ipae_A_to_B_mean, ipae_A_to_B_min, ipae_B_to_A_mean, ipae_B_to_A_min
```

Sequence and ESM fields include:

```text
design_chain_pi
esm_log_likelihood, esm_perplexity, esm_score_status
esm_fasta_path, esm_score_csv, esm_error
esmfold_plddt_mean, esmfold_status
esmfold_fasta_path, esmfold_pdb_path, esmfold_error
```

ipSAE fields include A-to-B, B-to-A, and max values for:

```text
ipSAE, ipSAE_d0chn, ipSAE_d0dom, ipTM_af, ipTM_d0chn
pDockQ, pDockQ2, LIS
```

SASA/BSA fields include:

```text
sasa_target, sasa_binder, sasa_complex
bsa, bsa_interface
sasa_status, sasa_error
```

SASA uses Biotite and defaults to `--sasa-point-number 1000`. `bsa` is calculated
as `sasa_target + sasa_binder - sasa_complex`, using the current complex
conformation with the other chain deleted for the target-only and binder-only SASA.

## Candidate Filter

`candiate.csv` is filtered from `aggregate_results.csv`. The spelling is
intentional and matches the pipeline output.

A row is included only when all fields are present and all conditions pass:

```text
(ipae_A_to_B_mean + ipae_B_to_A_mean) / 2 <= 1.9
iptm >= 0.80
normalized_plddt_global_mean >= 0.85
exp(-esm_log_likelihood) < 10
```

`candiate.csv` keeps the complete `aggregate_results.csv` header, preserves
aggregate row order, does not rebuild columns, and does not sort by metrics.

## One-command Pipeline

Default `pipeline --csv ...` runs the complete workflow: target JSON, target-only
AF3, target data extraction, complex JSON, complex AF3, ESM/ipSAE, aggregation,
and candidate filtering. Only pass `--target-data-json` when reusing existing
target data and intentionally skipping target-only AF3.

Run the full workflow:

```bash
aerith pipeline \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work \
  --output-dir af_output
```

Run with an existing target data JSON:

```bash
aerith pipeline \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work \
  --output-dir af_output \
  --target-data-json af_output/target_A_data.json
```

Complex AF3 output directories use the same default job names:

```text
af_output/sample_1_binder_candiate_complex_pred/
af_output/sample_2_binder_candiate_complex_pred/
...
```

Without `--target-data-json`, `pipeline` writes the target input, runs target AF3,
then looks for:

```text
af_output/target_A_data.json
af_output/target_A/target_A_data.json
```

It stops before complex AF3 if the target data JSON is still missing.

## Verified Fixture Workflow

This project was validated with the local fixture data using the uv environment
and the real ESM/ipSAE/SASA code paths:

```bash
while [[ -n "${CONDA_PREFIX:-}" ]]; do conda deactivate; done
source .venv/bin/activate

python -m compileall -q src main.py

aerith check --csv tests/AF3_pipeline_dev_sample.csv

aerith make-target \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work/readme_fixture \
  --force

aerith build-complex \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --work-dir work/readme_fixture \
  --limit 1 \
  --force \
  --target-data-json tests/ref/af_output/sample_1_PD1_binder_candiate_complex_pred/sample_1_PD1_binder_candiate_complex_pred_data.json \
  --job-name-template "sample_{sample_no}_PD1_binder_candiate_complex_pred"

aerith score-ipsae \
  --work-dir work/readme_fixture \
  --output-dir tests/ref/af_output \
  --no-ray

aerith score-esmfold \
  --work-dir work/readme_fixture \
  --output-dir tests/ref/af_output \
  --no-ray \
  --force

aerith score-esm \
  --work-dir work/readme_fixture \
  --output-dir tests/ref/af_output \
  --no-ray \
  --force

aerith aggregate \
  --csv tests/AF3_pipeline_dev_sample.csv \
  --output-dir tests/ref/af_output \
  --results-dir work/readme_fixture_results \
  --score-dir work/readme_fixture/scores \
  --job-name-template "sample_{sample_no}_PD1_binder_candiate_complex_pred"
```

Observed successful fixture metrics:

```text
job_status: success
ranking_score: 0.8
esm_score_status: success
esm_log_likelihood: -1.3837177597473715
esm_perplexity: 3.989706860862709
ipsae_score_status: success
ipSAE_max: 0.624186244364005
sasa_status: success
bsa: 2016.7880859375
design_chain_pi: 8.21
esmfold_plddt_mean: 88.5
```
