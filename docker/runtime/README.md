# Unified fold runtime

Aerith uses one runtime image but starts a separate, GPU-isolated container for
each pipeline stage. Model checkpoints, databases, screen inputs, and outputs
remain external read-only or read-write mounts; they are never baked into the
image.

## Locked inputs

docker/runtime/sources.lock.yaml is the authority for all remote build inputs.
It pins the full Git commit and canonical tree hash for AlphaFold 3, Protenix,
OpenDDE, ESM, and OpenFold. It also pins the CUDA base digest, Ubuntu snapshot,
uv, Miniforge, HMMER, GPU MMseqs2, and GPU Foldseek artifacts and checksums.

Validate the lock and show content identities without cloning the repositories:

    uv run python scripts/runtime_sources.py validate
    uv run python scripts/runtime_sources.py metadata

Materialize and verify the five filtered BuildKit contexts:

    BUNDLE=/ssd/aerith-build/runtime-sources
    uv run python scripts/runtime_sources.py prepare --output "$BUNDLE"

The command checks each remote commit and tree, applies only checksum-locked
patches, strips development and test data, and atomically publishes a manifest.
No local source checkout is required for the production build.

## Build graph

docker/runtime/Dockerfile has independently addressable targets:

| Target | Contents | Rebuild identity |
| --- | --- | --- |
| uv-component | AF3 and OpenDDE uv environments | uv sources, uv recipe, uv dependency locks |
| conda-component | Protenix, ESM, and OpenFold conda environments | conda sources, conda recipe, conda dependency locks |
| runtime-base | GPU MMseqs2, GPU Foldseek, HMMER, Kalign, entrypoint, feature adapters | shared tools and shared recipe |
| fold-runtime | local all-in-one development result | complete recipe and source provenance |

The first three outputs are content-only OCI components. UV and Conda builds
can run in parallel and a change isolated to one environment does not invalidate
the other component tag. docker/runtime/Dockerfile.assemble copies exact
digest-pinned components into the release candidate.

Compiler packages exist only in builder stages. The final runtime keeps AF3
ptxas where required, removes system nvcc and the ESM/OpenFold compiler payload,
and runs validate-fold-runtime before the final layer is accepted.

## Local development build

The normal local command prepares the locked source bundle automatically:

    aerith config validate --config config.yaml
    aerith build-runtime-image --config config.yaml --dry-run
    aerith build-runtime-image --config config.yaml

Then run:

    docker run --rm --gpus all --network none \
      aerith/fold-runtime:local doctor

This all-in-one local build is useful for diagnosis and inference. It does not
have the two external component image digests required by the release verifier,
so it is not a promotable release artifact.

A direct development build requires all five named contexts:

    docker build \
      --build-context af3-src=/path/to/alphafold3 \
      --build-context protenix-src=/path/to/Protenix \
      --build-context opendde-src=/path/to/OpenDDE \
      --build-context esm-src=/path/to/esm \
      --build-context openfold-src=/path/to/openfold \
      --file docker/runtime/Dockerfile \
      --tag aerith/fold-runtime:local .

Use docker build --check with the same contexts for a static recipe check.
Direct source-tree builds are development-only because they bypass the locked
GitHub checkout contract.

Docker image placement is controlled by the daemon. Configure the Docker data
root on /ssd before a large build; never move individual layer directories.

## GitHub release flow

The repository intentionally separates fast CI, heavy image construction, and
GPU acceptance:

1. ci.yml runs locked dependencies, Ruff, Deptry, compile checks, source-lock
   validation, and CPU tests on GitHub-hosted runners.
2. docker-contract.yml runs on relevant pushes and pull requests. It performs
   source metadata validation and Dockerfile static checks only.
3. runtime-build.yml is manual-only. A GitHub-hosted runner prepares verified
   contexts, then sends the build to a remote BuildKit daemon over mutual TLS.
4. gpu-smoke.yml is manual-only on the dedicated GPU host. It pulls a candidate
   by registry digest, verifies provenance, runs doctor with network disabled,
   and executes AF3 plus the selected secondary-backend golden contract.
5. An optional promotion job, protected by the runtime-release GitHub
   environment, retags the exact tested digest as a release tag and stable.

The build workflow never publishes latest and never promotes a candidate.
Component images, registry caches, candidates, and stable releases stay in the
private GHCR namespace owned by the repository account.

Configure these repository secrets with the endpoint and PEM contents for a
remote amd64 BuildKit service:

    AERITH_BUILDKIT_ENDPOINT=tcp://buildkit.example.internal:1234
    AERITH_BUILDKIT_CA=<PEM certificate authority>
    AERITH_BUILDKIT_CERT=<PEM client certificate>
    AERITH_BUILDKIT_KEY=<PEM client private key>

The remote builder must have enough persistent SSD space, outbound access to
the pinned package sources, amd64 workers, and registry access supplied through
the GitHub client session. Do not expose an unauthenticated BuildKit TCP socket.

Configure the self-hosted GPU runner with labels self-hosted, linux, x64, and
aerith-gpu. Set these repository variables to external files and directories:

    AERITH_GPU_SMOKE_CONFIG=/ssd/aerith-ci/golden/config.yaml
    AERITH_GPU_SMOKE_CONTRACT=/ssd/aerith-ci/golden/contract.json
    AERITH_GPU_SMOKE_ROOT=/ssd/aerith-ci/runs
    AERITH_GPU_SMOKE_LOCK=/ssd/aerith-ci/gpu-smoke.lock

Create a protected GitHub environment named runtime-release and require a
reviewer before deployment. External fork pull requests must never trigger the
GPU workflow.

Run Manual runtime build with scope uv, conda, or all. Missing component
identities are built automatically; existing immutable component tags are
reused. The final output is:

    ghcr.io/<owner>/aerith-fold-runtime:candidate-<run-id>
    ghcr.io/<owner>/aerith-fold-runtime:candidate

Pass the run-specific candidate to GPU golden smoke when concurrent human
activity could update the mutable candidate alias. Promotion always uses the
digest resolved and tested by the smoke job.

## Provenance and verification

Release candidates record the complete dependency lock, full recipe, source
lock, source bundle, per-source tree hashes, component build identities, actual
UV and Conda image digests, Ubuntu snapshot, and clean-source state.

Verify a pulled candidate and then run the container doctor:

    uv run python scripts/verify_runtime_image.py \
      --image ghcr.io/<owner>/aerith-fold-runtime@sha256:<digest>

    docker run --rm --gpus all --network none \
      ghcr.io/<owner>/aerith-fold-runtime@sha256:<digest> doctor

Missing values remain failures. A dirty source bundle, incomplete hash labels,
or unavailable component digest cannot be exported or promoted as a release.

## Validation

For source changes run:

    uv run pytest -q
    uv run python -m compileall -q src scripts docker
    git diff --check

For runtime changes also run source metadata validation, both Dockerfile
checks, and one controlled real-data GPU smoke. Ordinary pull-request CI must
never start a full production screen.
