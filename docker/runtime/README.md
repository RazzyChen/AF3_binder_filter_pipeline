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

The Dockerfile downloads and SHA-256 verifies the official
[MMseqs2 18-8cc5c GPU](https://github.com/soedinglab/MMseqs2/releases/download/18-8cc5c/mmseqs-linux-gpu.tar.gz)
and [Foldseek 10-941cd33 GPU](https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-gpu.tar.gz)
archives. Runtime commands never mount host MMseqs2 or Foldseek executables.

The pipeline wrapper performs source commit and MMseqs2 SHA-256 validation and
is preferred for reproducible local builds:

    aerith build-runtime-image --config config.yaml --dry-run
    aerith build-runtime-image --config config.yaml

Verify the resulting image before inference:

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
