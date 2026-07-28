# Unified fold runtime

Build from the repository root with BuildKit named contexts. These contexts
must contain source trees only; checkpoints, databases, and host environments
are runtime mounts:

    docker build \
      --build-context af3-src=/home/structure/Software/alphafold3-3.0.3 \
      --build-context protenix-src=/home/structure/Software/Protenix-2.0.0 \
      --build-context opendde-src=/home/structure/Software/OpenDDE \
      --build-context esm-src=/home/structure/Software/esm \
      -f docker/runtime/Dockerfile -t aerith/fold-runtime:local .

This direct command is only for development diagnosis. For a static recipe
check without executing build steps, add `--check` with the same named contexts.
Release builds must use a verified source bundle.

The Dockerfile downloads and SHA-256 verifies the official
[MMseqs2 18-8cc5c GPU](https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-gpu.tar.gz)
and [Foldseek 10-941cd33 GPU](https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-gpu.tar.gz)
archives. Runtime commands never mount host MMseqs2 or Foldseek executables.

Ubuntu apt dependencies are resolved from the fixed
`20260723T000000Z` Ubuntu snapshot. Changing that date is an explicit recipe
update and requires a new runtime validation run.

The Dockerfile is multi-stage. The `builder` stage contains CUDA development
packages and compiles Protenix and OpenDDE fused layer-normalization extensions.
The final `runtime` stage copies only execution artifacts: it retains AF3's
`ptxas` helper and Protenix Triton helper, but excludes system `nvcc` and the ESM
OpenFold compiler payload. `doctor` asserts that both fused extensions are
already importable, so inference never silently relies on a host CUDA toolkit.
The default is `LAYERNORM_TYPE=fast_layernorm`.

The pipeline wrapper performs source commit and MMseqs2 SHA-256 validation and
is preferred for reproducible local builds:

    aerith build-runtime-image --config config.yaml --dry-run
    aerith build-runtime-image --config config.yaml

The wrapper rejects dirty Git source trees by default and, for a frozen bundle,
rechecks its OpenDDE and ESM commits against the configured pins. For a deliberate
local experiment only, `runtime.allow_dirty_source_trees=true` records the dirty
state in the bundle manifest. It is not release provenance.
A dirty bundle can only be built while the same override remains explicit; the
result is labelled as dirty and is rejected by the release exporter by default.

A self-hosted builder can select a candidate tag and persistent local cache:

    docker buildx create --name aerith-runtime-ci --driver docker-container --use
    uv run python scripts/build_runtime_image.py \
      --config /ssd/aerith-ci/runtime-build.yaml \
      --source-bundle /data/aerith/runtime-sources/release-YYYYMMDD \
      --image aerith/fold-runtime:ci-example \
      --cache-dir /ssd/aerith-buildkit-cache \
      --builder aerith-runtime-ci

The local cache exporter requires the `docker-container` Buildx driver; Docker's
default `docker` driver cannot export `type=local` cache data.

The CI preflight rejects a same-named builder using any other driver.

When invoked with `--builder`, the Aerith build wrapper also passes `--load`, so
the verified candidate is available to subsequent local `docker run` commands.

Verify the resulting image before inference:

    uv run python scripts/verify_runtime_image.py \
      --image aerith/fold-runtime:local

    docker image inspect aerith/fold-runtime:local
    docker run --rm --gpus all --network none \
      aerith/fold-runtime:local doctor

The image contains four isolated prediction environments plus the offline
feature toolchain. The dispatcher exposes `af3`, `protenix`, `opendde`,
`esmfold`, `esm-if`, and `prepare-features`. The pinned GPU MMseqs2
build (commit `8cc5ce…`), HMMER, Kalign, and ColabFold run inside the same
image. Padded-database search requires a GPU and passes `--gpu 1`; it does
not silently fall back to CPU. Runtime databases, checkpoints, features, inputs,
and outputs are mounted by Aerith and are not baked in.

Aerith launches one container per selected physical GPU with
`--gpus device=<host-index>`. CUDA/JAX/Torch inside that container sees one
device and therefore uses local device 0. AF3 shards complete before eligible
jobs are re-sharded for Protenix or OpenDDE; ESMFold and ESM-IF follow the same
container-isolation policy.
