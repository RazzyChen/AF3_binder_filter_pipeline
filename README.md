# Aerith

Aerith is a production control plane for protein Binder screening. It does not
implement a new folding model: it validates one immutable job plan, prepares
offline target features, schedules isolated GPU containers, normalizes backend
outputs, analyzes interfaces, and produces an auditable diversity shortlist.

The fixed primary backend is AlphaFold 3. A run may optionally use Protenix-v2
or OpenDDE as a secondary backend for rescue and cross-validation. AF3,
Protenix, OpenDDE, ESMFold, ESM-IF, GPU MMseqs2, and GPU Foldseek are packaged in
one `aerith/fold-runtime` image, but Aerith starts a separate container for each
stage and GPU shard. Biotite geometry and Rosetta InterfaceAnalyzer are
orchestrated on the host.

The canonical repository is
[RazzyChen/aerith](https://github.com/RazzyChen/aerith). `main` is the stable
line and `Dev` is the integration branch. This repository contains orchestration
code, configuration, adapters, tests, and the reproducible runtime recipe; model
weights, databases, and production run data remain external.

## Production quick start

Run commands from the repository root:

```bash
git switch Dev
uv venv --prompt aerith .venv
uv sync --locked
source .venv/bin/activate

aerith config create \
  --output /ssd/aerith_screens/HER3/config.yaml \
  --project-root /ssd/aerith_screens/HER3 \
  --secondary-backend opendde \
  --gpu-ids 0,1,2

aerith config validate --config /ssd/aerith_screens/HER3/config.yaml
aerith config doctor --config /ssd/aerith_screens/HER3/config.yaml
aerith pipeline --config /ssd/aerith_screens/HER3/config.yaml --dry-run
aerith pipeline --config /ssd/aerith_screens/HER3/config.yaml
```

The default CSV is `<project-root>/input/screen.csv` and must contain:

```text
sample_no,run_name,binder_sequence,target_seq
```

Every nonblank row must be complete and valid. Completely blank physical rows
are ignored. All planned rows must share exactly one target sequence; sanitized
job IDs must be unique. `project.limit` is applied once to the immutable job
plan and therefore affects every downstream stage consistently.

Use `--csv PATH` when the CSV is elsewhere and
`--epitope-residues 405,409,436` when a reference target epitope is known:

```bash
aerith config create \
  --output config.yaml \
  --project-root /ssd/aerith_screens/HER3 \
  --csv /ssd/aerith_screens/HER3/input/screen.csv \
  --secondary-backend protenix \
  --epitope-residues 405,409,436 \
  --gpu-ids 0,1,2,3
```

`config create` writes the minimal screen-specific YAML. `config init` writes a
fully expanded host-detected configuration. Both use Hydra Structured Config;
Hydra is composed through the Python API and does not change the working
directory.

## Supported CLI

The production command surface is intentionally small:

```text
aerith config create|init|validate|doctor|show
aerith build-runtime-image
aerith prepare-features
aerith analyze-interface
aerith cluster
aerith pipeline
```

The old target JSON, direct AF3 runner, standalone ESM/ipSAE, aggregation, and
legacy pipeline commands are no longer public CLI APIs. New automation should
only call the commands above.

AF3 is always primary. Select a secondary backend per run:

```bash
# AF3 only
aerith pipeline --config config.yaml --secondary-backend none

# AF3 plus one cross-validation backend
aerith pipeline --config config.yaml --secondary-backend protenix
aerith pipeline --config config.yaml --secondary-backend opendde

# Repeatable Hydra overrides remain available
aerith pipeline --config config.yaml --secondary-backend opendde \
  --override interface.distance=4.5
```

OpenDDE uses the general-purpose `opendde.pt` checkpoint by default. A different
checkpoint must be selected explicitly and becomes part of provenance.

## Pipeline model

The production sequence is:

```text
preflight
  -> offline target MSA/templates
  -> AF3 prediction and interface analysis
  -> optional secondary feature adaptation, prediction, and interface analysis
  -> consensus and effective-backend selection
  -> effective-aware ESM scoring
  -> three-layer clustering and representative selection
```

Run artifacts are organized by the same order:

```text
results/<run_id>/
├── resolved_config.yaml
├── manifest.json
├── all_results.csv
├── candidates.csv
├── final_shortlist.csv
├── backend_review.csv
└── stages/
    ├── 01_preflight/{logs,tables,artifacts}/
    ├── 02_features/{logs,tables,artifacts}/
    ├── 03_primary_prediction/{logs,tables,artifacts}/
    ├── 04_primary_interface/{logs,tables,artifacts}/
    ├── 05_secondary_features/{logs,tables,artifacts}/
    ├── 06_secondary_prediction/{logs,tables,artifacts}/
    ├── 07_secondary_interface/{logs,tables,artifacts}/
    ├── 08_consensus/{logs,tables,artifacts}/
    ├── 09_esm/{logs,tables,artifacts}/
    └── 10_clustering/{logs,tables,artifacts}/
```

Commands, stdout, and stderr live under each stage's `logs/`; detailed tables
under `tables/`; Rosetta inputs, ESM structures, Foldseek files, and other
derived data under `artifacts/`. Raw backend predictions remain in
`outputs/<run_id>/<backend>/` rather than being duplicated into `results/`.

### Orchestration architecture

`src/af3_binder_filter/orchestration/` is the typed production orchestration
package. Its public API is explicit (`create_run_context`, `run_pipeline`, and
standalone stage entry points); CLI code imports the owning modules directly. `PipelineRunner`
keeps the ten-stage transition order explicit, while `PipelineState` is the only
object carrying mutable predictions, rows, manifest state, and clustering
results across stage boundaries. Feature preparation, prediction, interface
analysis, ESM, clustering, cache identity, resume validation, command execution,
and effective selection are separate modules and can be tested independently.

The stage registry owns stable names, order, progress labels, and conditional
secondary/ESM visibility. It deliberately does not dynamically instantiate
stages: cache recovery, failure propagation, and manifest writes remain visible
in the runner. Factories are reserved for components with interchangeable
implementations, currently the configured interface energy engine. The public
`run_pipeline()` function, CLI behavior, Hydra schema, manifest/output schemas,
and directory layout remain compatible.

Target MSA and templates are built locally with GPU MMseqs2 and the configured
AF3 database. The optional secondary backend reuses the target features. A de
novo Binder remains query-only, with no paired MSA and no templates. Prediction
and feature containers run with `--network none`; databases and checkpoints are
mounted read-only. Concurrent runs requesting the same target/database identity
share a process-safe feature-cache lock: the first run performs the GPU search,
while waiters revalidate and consume the published cache instead of launching
duplicate MMseqs2 work.

## Effective backend semantics

`effective_backend` is the one backend whose structure, contacts, and quality
metrics are used for ESM-IF, all three clustering layers, and cluster/cell
representative selection. It is an explicit projection, not a new confidence
score.

A backend is eligible only when its best model is present and parseable and its
Biotite interface status is `success`. If both backends are eligible, Aerith
uses this deterministic lexicographic order:

1. backend pass (`true` first);
2. epitope coverage (higher is better);
3. interface PAE (lower is better);
4. Rosetta normalized dG (lower/more negative is better);
5. Rosetta packstat (higher is better);
6. ipTM (higher is better);
7. secondary backend on an exact tie.

Missing values rank behind valid values. When no reference epitope is
configured, coverage is neutral. `effective_selection_reason` records the first
criterion that selected the backend. A secondary rescue therefore enters
Binder-fold, complex-pose, and target-contact clustering with the secondary
structure rather than an absent or failed AF3 structure.

In dual-backend mode the secondary gate defaults to AF3 `ipTM >= 0.70`. A row is
a candidate only when it entered that gate, the secondary prediction succeeded,
and at least one backend passed the geometry/epitope gate. Secondary rescue and
backend disagreement are retained but marked for manual review; disagreement is
not itself a hard filter.

Default deterministic review triggers are:

- Binder fold TM-score `< 0.50`;
- target-frame Binder RMSD `> 5.0 Å`;
- interface residue-pair Jaccard `< 0.30`;
- target contact/epitope Jaccard `< 0.10` when both interfaces have enough
  contact residues.

The review table also preserves robust cohort anomaly annotations when enabled.

## ESM semantics

Effective selection happens before ESM scoring:

- ESMFold predicts each Binder sequence once.
- `esmfold_effective_binder_tm` compares that fold with the effective Binder.
- ESM-IF scores the Binder backbone selected by `effective_backend` once.
- `backend_review.csv` additionally contains ESMFold-to-primary and
  ESMFold-to-secondary TM comparisons.

If no eligible effective structure exists, ESMFold can still report a sequence
fold, while ESM-IF is unavailable for that row. Missing values remain missing;
Aerith never substitutes numeric zero for an unavailable scientific metric.

## Output schema v3

Schema v3 separates decision-facing results from complete multi-backend audit
data.

The three decision CSVs have the same ordered 55-column schema and report only
effective-backend scientific values:

- `all_results.csv`: every planned input job, including failed jobs;
- `candidates.csv`: rows with `candidate_pass=true`;
- `final_shortlist.csv`: one selected quality representative per three-layer
  diversity cell.

Their columns cover identity/input, candidate and manual-review summaries, 22
`effective_*` prediction/interface/Rosetta fields, effective-aware ESM fields,
and clustering/final-rank fields. Detailed `primary_*` and `secondary_*` metrics
do not appear in these decision tables.

`backend_review.csv` contains every planned job and has 108 ordered columns. It
contains the complete 55-column decision projection plus:

- primary, secondary-gate, and secondary pass state;
- full primary prediction/interface/Rosetta values;
- full secondary prediction/interface/Rosetta values;
- detailed structural consensus values;
- ESMFold comparisons against both backend structures.

Use `backend_review.csv` for joint backend review and the three compact files
for screening decisions. The manifest records output schema version 3 and the
report artifacts. Residue fields always use input-sequence 1-based numbering
with explicit chains, for example `A:405`, `B:15`, and `A:405-B:15`.

See [SHORTLIST_COLUMNS.md](SHORTLIST_COLUMNS.md) for the direction, units,
missing-value rules, and caveats of every decision and review field.

## Interface analysis and failure semantics

Biotite always computes geometry first. Heavy-atom contacts default to `<= 5.0
Å`; the balanced hard gate requires at least five contact pairs and, when a
reference epitope is provided, the configured coverage threshold (default
`0.30`). Coverage is:

```text
number of configured target epitope positions contacted
--------------------------------------------------------
number of configured target epitope positions
```

Purity is not a filter. A non-null legacy `minimum_epitope_purity` value is a
configuration error.

When `energy_engine=rosetta_cli`, every Biotite-successful structure is expected
to complete Rosetta InterfaceAnalyzer. Any Rosetta error, timeout, missing
output, or parse error makes that interface stage `partial`, while valid
Biotite geometry and successful Rosetta rows are preserved. The pipeline writes
safe partial reports and exits nonzero when `project.allow_partial=false`; with
`allow_partial=true`, it records the partial state and returns successfully.
Missing Rosetta binary or database paths are configuration validation errors.

## Three-layer clustering

Candidate diversity is represented by:

1. Binder fold: Foldseek `easy-cluster` on the effective Binder chain;
2. complex pose: Foldseek `easy-multimercluster` on the effective A/B complex;
3. epitope: deterministic greedy clustering of effective target-contact sets.

The balanced defaults are Binder TM-score `0.50`, Binder coverage `0.80`,
multimer TM-score `0.65`, chain TM-score `0.50`, interface lDDT `0.65`, and
epitope Jaccard `0.50`.

A Foldseek singleton is accepted only when Foldseek actually reports the
structure. Missing input, extraction failure, or missing Foldseek output is not
converted into a synthetic singleton: the row receives
`clustering_status=error`, is retained for audit/manual review, and is excluded
from the final shortlist.

## Multi-GPU performance

Aerith remains a single-host, multi-GPU Docker scheduler; Kubernetes is not
required for this execution model.

The host scheduler deliberately uses bounded Python worker pools and the
process-safe executor in this repository. It does not install Ray, Modin, or a
second distributed scheduler. Heavy model runtimes remain isolated inside the
unified Docker image.

### Weighted prediction sharding

Pending prediction and ESM jobs are assigned with deterministic
longest-processing-time balancing. The cost proxy is:

```text
(target_length + binder_length) ** 2
```

Jobs are sorted by descending cost and placed on the currently lightest GPU;
ties use GPU index and job ID. Cache hits do not enter the pending workload.
This avoids assigning several long complexes to one GPU merely because their
row numbers are adjacent.

### Parallel geometry

Biotite interface parsing, contacts, SASA, and Rosetta-input conversion use a
bounded worker pool:

```yaml
runtime:
  geometry_max_workers: 4
```

Results are restored to immutable job-plan order, so worker completion order
does not alter CSV ordering. Rosetta has a separate
`interface.rosetta.max_workers` limit.

### Parallel Foldseek layers

Binder and complex Foldseek remain serial by default. With at least two free
allowed GPUs they can run concurrently:

```yaml
clustering:
  max_workers: 2
```

Aerith assigns the two layers to different physical GPUs. If fewer GPUs are
available, they run sequentially. Epitope clustering remains deterministic on
CPU.

`runtime.gpu_ids` is an allow-list; an empty list means all discovered GPUs.
Devices above `runtime.gpu_busy_threshold_mib` are excluded. Each container
receives one physical device via `docker --gpus device=<host_gpu>` and sees it as
device 0 internally.

### Process lifecycle and timeouts

Prediction, feature-builder, ESMFold, and ESM-IF shards use one shared,
stage-scoped process executor. Every shard writes a shell-quoted command record,
stdout, and stderr under its stage log directory. A timeout, Ctrl-C, or other
controller exception terminates and reaps the complete process group; named
Docker containers receive an additional bounded `docker rm -f` cleanup. Real
negative signal return codes are preserved and a command that never started is
reported without inventing a fake signal.

ESM commands use `scoring.esm.timeout_seconds`; target feature preparation uses
`features.timeout_seconds`. On Ctrl-C the current stage and run manifest are
atomically persisted as `interrupted` before the interrupt is re-raised.

## Fingerprints, recovery, and provenance

Automatic run IDs derive from the normalized scientific configuration and job
plan. Provenance includes sequences/chains, backend models and checkpoints,
feature/database identity, resolved Docker image IDs, interface/Rosetta and
clustering settings, ESM/consensus settings, output schema, and Aerith code
identity. Scheduling-only settings such as GPU IDs and worker counts are
recorded but do not change scientific identity.

Cache reuse requires a matching fingerprint/manifest and parseable artifacts.
If an explicit `project.run_id` already has a different fingerprint, Aerith
refuses it before overwriting resolved configuration or results. A nonempty run
directory without a valid manifest is also refused. Standalone clustering must
consume artifacts from the same run identity and schema.

Interface parsing also publishes a content-addressed derivative bundle
(normalized A/B complex, target, Binder, residue map, and coordinate NPZ). ESM,
consensus, and Foldseek reuse that bundle. If a row declares a successful
derivative and any bound file or checksum changes, downstream stages fail
explicitly rather than silently reparsing another model or manufacturing a
singleton cluster.

CLI progress reports stage status, completed/total counts, and the literal
messages `cache hit!` or `cache missing!`.

## Unified runtime image

The detailed build and release runbook is
[docker/runtime/README.md](docker/runtime/README.md).

The image recipe is `docker/runtime/Dockerfile`. It contains isolated AF3 and
OpenDDE uv environments, Protenix and ESM conda environments, and pinned GPU
MMseqs2/Foldseek releases. Checkpoints and `/data/AF3_database` are mounted at
runtime and are not baked into image layers.

The tracked source authority is `docker/runtime/sources.lock.yaml`. It pins
full commits and canonical Git tree hashes for AF3, Protenix, OpenDDE, ESM, and
OpenFold, plus checksums for all downloaded build tools. The multi-stage recipe
publishes three independent targets:

- `uv-component`: AF3 and OpenDDE;
- `conda-component`: Protenix, ESM, and OpenFold;
- `runtime-base`: GPU MMseqs2, GPU Foldseek, HMMER, Kalign, the dispatcher,
  and feature adapters.

The UV, Conda, and shared targets have independent source, recipe, and lock
identities. UV and Conda can build concurrently without installing into one
another. `docker/runtime/Dockerfile.assemble` combines exact component OCI
digests into the candidate. CUDA compilers remain in builder layers; the final
image retains only required runtime helpers and precompiled extensions.

### Local development build

The Dockerfile requires five named source contexts. A direct BuildKit command
from the repository root is:

```bash
docker build \
  --build-context af3-src=/home/structure/Software/alphafold3-3.0.3 \
  --build-context protenix-src=/home/structure/Software/Protenix-2.0.0 \
  --build-context opendde-src=/home/structure/Software/OpenDDE \
  --build-context esm-src=/home/structure/Software/esm \
  --build-context openfold-src=/home/structure/Software/openfold \
  --file docker/runtime/Dockerfile \
  --tag aerith/fold-runtime:local \
  .
```

This direct command is useful for development diagnosis only: it bypasses the
locked GitHub checkout gate. Use `docker build --check` with the same named
contexts for a fast static recipe check.

For normal local builds, use the Aerith wrapper. It validates the immutable
source lock, prepares the five filtered contexts, and checks all artifact
hashes before Docker runs:

```bash
aerith config validate --config config.yaml
aerith build-runtime-image --config config.yaml --dry-run
aerith build-runtime-image --config config.yaml

docker image inspect aerith/fold-runtime:local
docker run --rm --gpus all --network none \
  aerith/fold-runtime:local doctor
```

The all-in-one local result is a development image. Release candidates require
the separately published UV and Conda component digests described below.

Docker image placement is controlled by the Docker daemon, not by this
Dockerfile. To keep layers on an SSD, configure Docker's data root once and
verify it before a large build:

```bash
docker info --format '{{.DockerRootDir}}'
df -h /ssd
docker system df
```

On a rootless installation the data-root path is configured in the rootless
daemon settings. Moving individual image directories by hand is unsupported;
export with Aerith, change the daemon data root, then reload the verified image
when migration is required.

### Release source and component build

Release builds start from the repository-tracked source lock, not mutable host
source directories:

```bash
uv run python scripts/runtime_sources.py validate
uv run python scripts/runtime_sources.py metadata

BUNDLE=/ssd/aerith-build/runtime-sources
uv run python scripts/runtime_sources.py prepare --output "$BUNDLE"
```

The bundle contains five filtered contexts and an atomic manifest. Every remote
commit and canonical tree hash is verified before checksum-locked patches are
applied. The manifest records the source lock, resulting context hashes, clean
Git state, and bundle identity.

The supported release path is the manual `runtime-build.yml` workflow. A
GitHub-hosted runner prepares the bundle and controls an amd64 remote BuildKit
daemon over mutual TLS. BuildKit publishes independently content-addressed
`uv-component`, `conda-component`, and `runtime-base` images to private
GHCR, reuses registry cache layers, and builds missing targets in parallel.
`Dockerfile.assemble` then combines their exact registry digests into:

```text
ghcr.io/<owner>/aerith-fold-runtime:candidate-<run-id>
ghcr.io/<owner>/aerith-fold-runtime:candidate
```

The run-specific tag is the auditable handoff to GPU smoke; the second tag is a
human convenience alias. The workflow never writes `latest` or a release tag.

Candidate labels contain the full runtime lock, complete recipe, source lock,
source bundle, per-source tree hashes, component build identities, actual UV
and Conda OCI digests, Ubuntu snapshot, and clean-source status. A missing or
dirty value is rejected by `scripts/verify_runtime_image.py`.

The runtime recipe resolves Ubuntu packages from the immutable
`snapshot.ubuntu.com/ubuntu/20260723T000000Z` state. Updating that snapshot,
any pinned source, dependency lock, patch, or relevant Docker stage creates a
new component identity. An isolated UV change does not invalidate the Conda
component identity, and vice versa.

Local dirty-source overrides remain available only for explicit experiments:

```yaml
runtime:
  allow_dirty_source_trees: true
```

They are recorded as dirty and cannot pass release verification or promotion.

### Export and restore

Export uses `docker image save`, preserving layers, labels, and image identity;
it never uses `docker export`:

```bash
uv run python scripts/export_runtime_image.py \
  --image ghcr.io/<owner>/aerith-fold-runtime@sha256:<digest> \
  --output-dir /ssd/aerith_images
```

The command automatically creates an immutable full-image-ID tag and writes:

- `<repository>-sha256-<full-id>.docker.tar.zst`;
- the archive's `.sha256` checksum file;
- `.metadata.json` containing normalized image inspection and restore commands.

Verify and restore using the exact paths printed by the export command:

```bash
cd /ssd/aerith_images
sha256sum --check <archive>.sha256
zstd --decompress --stdout <archive>.docker.tar.zst | docker image load
docker image inspect <immutable-tag-printed-by-export>
```

By default export refuses local all-in-one builds, images with missing
provenance labels, or dirty source provenance. `--allow-unprovenanced` exists
only for explicitly acknowledged historical or non-release experimental images
and must not be described as a reproducible release.

This documentation does not claim that a particular local image has already
been exported or GPU-smoke-tested. Record those facts only after the archive,
checksum, load verification, and one controlled real-data run have completed.

## Validation

After code changes run at least:

```bash
uv run pytest -q
uv run python -m compileall -q src scripts docker
git diff --check
```

Configuration and runtime changes additionally require:

```bash
aerith config validate --config config.yaml
aerith config doctor --config config.yaml
aerith pipeline --config config.yaml --dry-run
docker run --rm --gpus all --network none \
  aerith/fold-runtime:local doctor
```

Real GPU acceptance begins with one controlled Binder. Full production screens
must not run automatically in ordinary pull-request CI.

## CI/CD and runtime delivery

`.github/workflows/ci.yml` is the required CPU gate for every push and pull
request. It creates the locked uv environment, runs Ruff and Deptry, validates
the source lock, compiles all tracked Python, and executes
`pytest -m "not integration"` independently on Python 3.11 and 3.12. Protect
`main` and `Dev` with `Quality` and both `Unit tests` checks.

`docker-contract.yml` is also GitHub-hosted and runs only when runtime-related
files change. It validates source/build identities and executes `docker build
--check` for both Dockerfiles. It never performs a heavy build or pushes an
image, so it remains a suitable pull-request gate.

`runtime-build.yml` is a manual GitHub workflow. The hosted runner is the
authenticated controller; heavy layers are built by a remote BuildKit daemon
with persistent SSD storage. Configure these repository secrets:

```text
AERITH_BUILDKIT_ENDPOINT=tcp://buildkit.example.internal:1234
AERITH_BUILDKIT_CA=<PEM certificate authority>
AERITH_BUILDKIT_CERT=<PEM client certificate>
AERITH_BUILDKIT_KEY=<PEM client private key>
```

The endpoint must require mutual TLS. The workflow logs into private GHCR with
the repository `GITHUB_TOKEN`, uses private registry caches, and supports
`all`, `uv`, or `conda` scope. Missing component identities are built
automatically. UV, Conda, and runtime-base builds run in parallel when needed;
the final candidate uses digest-pinned inputs. The workflow is not scheduled
and is never triggered by a pull request or push.

`gpu-smoke.yml` is manual-only on
`self-hosted, linux, x64, aerith-gpu`. It cannot be triggered by an external
fork. An OS lock prevents concurrent smoke runs, and the job refuses a host
with active GPU compute PIDs. Configure:

```text
AERITH_GPU_SMOKE_CONFIG=/ssd/aerith-ci/golden/config.yaml
AERITH_GPU_SMOKE_CONTRACT=/ssd/aerith-ci/golden/contract.json
AERITH_GPU_SMOKE_ROOT=/ssd/aerith-ci/runs
AERITH_GPU_SMOKE_LOCK=/ssd/aerith-ci/gpu-smoke.lock
```

The job resolves the candidate to an immutable registry digest, pulls that
digest, validates release provenance, runs `doctor` under `--network none`,
then executes AF3+OpenDDE and/or AF3+Protenix serially. If promotion is
requested, a separate GitHub-hosted job waits for approval in the protected
`runtime-release` environment and publishes the exact tested digest as the
requested release tag plus `stable`. No workflow writes `latest`.

The golden contract is external JSON with a fixed `job_id`, required result
columns, exact statuses, and scientifically chosen score ranges per secondary
backend. `scripts/gpu_smoke.py` writes a per-run summary beside the external
results and fails on any missing field, out-of-range metric, or nonzero pipeline
exit. It is intentionally not a replacement for release review.

The resulting topology is:

```text
GitHub-hosted CPU and Docker contract CI
    -> manual GitHub controller -> remote private BuildKit -> private GHCR candidate
        -> manual exclusive GPU runner -> protected digest promotion
```

This preserves GitHub auditability without making the production GPU host a
general-purpose image builder and without introducing Kubernetes.

## Large external assets

The following are intentionally outside Git:

- AF3 databases under `/data/AF3_database`;
- AF3, Protenix, OpenDDE, and ESM checkpoints;
- local source bundles under `/data/aerith/runtime-sources`;
- Docker image archives under `/ssd/aerith_images`;
- screen inputs, work directories, predictions, and results under `/ssd`.

Do not commit model weights, databases, real production outputs, absolute-path
production configurations, or credentials.
