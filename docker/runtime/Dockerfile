# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

ARG UBUNTU_SNAPSHOT=20260723T000000Z
ARG CUDA_BASE=nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04@sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e

# Shared compiler/toolchain stage. Python environments are deliberately built
# in separate descendants so uv and conda dependency graphs never overlap.
FROM ${CUDA_BASE} AS build-base
ARG UBUNTU_SNAPSHOT
ARG UV_VERSION=0.11.28
ARG UV_SHA256=e490a6464492183c5d4534a5527fb4440f7f2bb2f228162ad7e4afe076dc0224
ARG MINIFORGE_VERSION=26.3.2-3
ARG MINIFORGE_SHA256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    CONDA_DIR=/opt/conda \
    CUTLASS_PATH=/opt/cutlass \
    XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_CLIENT_MEM_FRACTION=0.95 \
    PATH=/usr/local/cuda-12.6/bin:/hmmer/bin:/usr/local/bin:/opt/conda/bin:/usr/bin:/bin

RUN --mount=type=cache,id=aerith-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=aerith-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && sed -ri "s#http://(archive|security).ubuntu.com/ubuntu/#https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#g" \
      /etc/apt/sources.list.d/ubuntu.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' \
      > /etc/apt/apt.conf.d/99aerith-snapshot \
    && grep -F "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
      /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git kalign libglib2.0-0 libgl1 \
      libxrender1 make patch python3-pip python3.12 python3.12-dev wget zlib1g-dev zstd \
      cuda-cccl-12-6=12.6.77-1 \
      cuda-crt-12-6=12.6.85-1 \
      cuda-cudart-dev-12-6=12.6.77-1 \
      cuda-driver-dev-12-6=12.6.77-1 \
      cuda-nvcc-12-6=12.6.85-1 \
      cuda-nvvm-12-6=12.6.85-1

RUN curl --fail --location --retry 5 --retry-all-errors \
      -o /tmp/uv.tar.gz \
      "https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/uv-x86_64-unknown-linux-gnu.tar.gz" \
    && echo "${UV_SHA256}  /tmp/uv.tar.gz" | sha256sum --check - \
    && mkdir /tmp/uv \
    && tar -xzf /tmp/uv.tar.gz -C /tmp/uv --strip-components=1 \
    && install -m 0755 /tmp/uv/uv /usr/local/bin/uv \
    && rm -rf /tmp/uv /tmp/uv.tar.gz \
    && test "$(uv --version)" = "uv ${UV_VERSION} (x86_64-unknown-linux-gnu)"
RUN curl --fail --location --retry 5 --retry-all-errors \
      -o /tmp/miniforge.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" \
    && echo "${MINIFORGE_SHA256}  /tmp/miniforge.sh" | sha256sum --check - \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda config --system --set channel_priority strict


FROM build-base AS tool-builder
ARG HMMER_VERSION=3.4
ARG HMMER_SHA256=ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3
ARG MMSEQS_RELEASE=18-8cc5c
ARG MMSEQS_VERSION=8cc5ce367b5638c4306c2d7cfc652dd099a4643f
ARG MMSEQS_SHA256=83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7
ARG FOLDSEEK_RELEASE=10-941cd33
ARG FOLDSEEK_VERSION=941cd33ff0771cd2e3f144e3293e22a2b87e9fda
ARG FOLDSEEK_SHA256=af7a688ffd8625b356c380380fb5650ec811262a2d18bdb0faeda95cc4894a55

# AF3 relies on the upstream jackhmmer --seq_limit patch.
COPY --from=af3-src /docker/jackhmmer_seq_limit.patch /tmp/jackhmmer_seq_limit.patch
RUN mkdir -p /tmp/hmmer-build /hmmer \
    && wget -q "http://eddylab.org/software/hmmer/hmmer-${HMMER_VERSION}.tar.gz" \
      -O /tmp/hmmer-build/hmmer.tar.gz \
    && echo "${HMMER_SHA256}  /tmp/hmmer-build/hmmer.tar.gz" \
      | sha256sum -c - \
    && tar -xzf /tmp/hmmer-build/hmmer.tar.gz -C /tmp/hmmer-build \
    && cd /tmp/hmmer-build \
    && patch -p0 < /tmp/jackhmmer_seq_limit.patch \
    && cd "hmmer-${HMMER_VERSION}" \
    && ./configure --prefix=/hmmer \
    && make -j"$(nproc)" \
    && make install \
    && cd easel \
    && make install \
    && rm -rf /tmp/hmmer-build /tmp/jackhmmer_seq_limit.patch

# GPU-native offline search and clustering tools belong to the shared runtime.
RUN curl --fail --location --retry 5 --retry-all-errors \
      -o /tmp/mmseqs-linux-gpu.tar.gz \
      "https://github.com/soedinglab/MMseqs2/releases/download/${MMSEQS_RELEASE}/mmseqs-linux-gpu.tar.gz" \
    && echo "${MMSEQS_SHA256}  /tmp/mmseqs-linux-gpu.tar.gz" | sha256sum --check - \
    && tar -xzf /tmp/mmseqs-linux-gpu.tar.gz -C /opt \
    && ln -s /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
    && rm /tmp/mmseqs-linux-gpu.tar.gz \
    && test "$(mmseqs version)" = "${MMSEQS_VERSION}" \
    && curl --fail --location --retry 5 --retry-all-errors \
      -o /tmp/foldseek-linux-gpu.tar.gz \
      "https://github.com/steineggerlab/foldseek/releases/download/${FOLDSEEK_RELEASE}/foldseek-linux-gpu.tar.gz" \
    && echo "${FOLDSEEK_SHA256}  /tmp/foldseek-linux-gpu.tar.gz" | sha256sum --check - \
    && tar -xzf /tmp/foldseek-linux-gpu.tar.gz -C /opt \
    && ln -s /opt/foldseek/bin/foldseek /usr/local/bin/foldseek \
    && rm /tmp/foldseek-linux-gpu.tar.gz \
    && test "$(foldseek version)" = "${FOLDSEEK_VERSION}"

COPY docker/feature-builder/build_local_features.py /opt/aerith/build_local_features.py
COPY docker/feature-builder/mmseqs_wrapper.py /opt/aerith/mmseqs_wrapper.py
COPY docker/feature-builder/convert_af3_templates.py /opt/aerith/convert_af3_templates.py
COPY docker/runtime/esm_if_batch.py /opt/aerith/esm_if_batch.py
COPY docker/runtime/entrypoint.sh /usr/local/bin/fold-runtime
COPY docker/runtime/validate_runtime.sh /usr/local/bin/validate-fold-runtime
RUN chmod 0755 /opt/aerith/mmseqs_wrapper.py /usr/local/bin/fold-runtime \
      /usr/local/bin/validate-fold-runtime


FROM build-base AS uv-builder
COPY --from=af3-src / /opt/apps/alphafold3
COPY --from=opendde-src / /opt/apps/opendde
COPY docker/runtime/locks /opt/locks
COPY docker/feature-builder/build_local_features.py /opt/aerith/build_local_features.py
COPY docker/feature-builder/convert_af3_templates.py /opt/aerith/convert_af3_templates.py

RUN --mount=type=cache,id=aerith-uv-af3,target=/root/.cache/uv \
    uv venv --python python3.12 /opt/envs/af3 \
    && cd /opt/apps/alphafold3 \
    && UV_PROJECT_ENVIRONMENT=/opt/envs/af3 uv sync --frozen --no-editable \
    && /opt/envs/af3/bin/build_data

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN --mount=type=cache,id=aerith-uv-opendde,target=/root/.cache/uv \
    uv python install 3.11.14 \
    && uv venv --python 3.11.14 /opt/envs/opendde \
    && uv pip install --python /opt/envs/opendde/bin/python \
      --no-deps --torch-backend cu126 --requirement /opt/locks/opendde.pip.lock \
    && uv pip install --python /opt/envs/opendde/bin/python \
      --no-deps /opt/apps/opendde

RUN CUDA_HOME=/usr/local/cuda-12.6 \
    CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
    CPATH=/opt/envs/opendde/lib/python3.11/site-packages/nvidia/cublas/include:/opt/envs/opendde/lib/python3.11/site-packages/nvidia/cusparse/include:/opt/envs/opendde/lib/python3.11/site-packages/nvidia/cusolver/include \
    LAYERNORM_TYPE=fast_layernorm \
    MAX_JOBS=8 /opt/envs/opendde/bin/python -c \
      'from opendde.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None'
RUN /opt/envs/af3/bin/python -c 'import alphafold3' \
    && /opt/envs/opendde/bin/opendde --help >/dev/null \
    && /opt/envs/opendde/bin/python /opt/aerith/build_local_features.py --help >/dev/null \
    && /opt/envs/af3/bin/python /opt/aerith/convert_af3_templates.py --help >/dev/null \
    && /opt/envs/opendde/bin/colabfold_search --help >/dev/null \
    && rm -rf /root/.cache /tmp/* /var/tmp/* \
    && find /opt/apps -type d \( -name __pycache__ -o -name .pytest_cache \) \
      -prune -exec rm -rf {} + \
    && find /opt/apps -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

# Content-only OCI component. It is safe to publish independently and can be
# consumed by the assembly Dockerfile without carrying the compiler layers.
FROM scratch AS uv-component
ARG UV_COMPONENT_SHA256=unavailable
LABEL org.aerith.runtime.component.uv.sha256="${UV_COMPONENT_SHA256}"
COPY --from=uv-builder /opt/envs /opt/envs
COPY --from=uv-builder /opt/uv-python /opt/uv-python
COPY --from=uv-builder /opt/apps/alphafold3 /opt/apps/alphafold3
COPY --from=uv-builder /opt/apps/opendde /opt/apps/opendde


FROM build-base AS conda-builder
COPY --from=protenix-src / /opt/apps/protenix
COPY --from=esm-src / /opt/apps/esm
COPY --from=openfold-src / /opt/apps/openfold-2.2.0
COPY docker/runtime/locks /opt/locks

RUN --mount=type=cache,id=aerith-uv-protenix,target=/root/.cache/uv \
    --mount=type=cache,id=aerith-conda-packages,target=/opt/conda/pkgs \
    /opt/conda/bin/conda create -y -n protenix \
      --file /opt/locks/protenix.conda-linux-64.lock \
    && uv pip install --python /opt/conda/envs/protenix/bin/python \
      --no-deps --torch-backend cu126 --requirement /opt/locks/protenix.pip.lock \
    && uv pip install --python /opt/conda/envs/protenix/bin/python \
      --no-deps /opt/apps/protenix
RUN --mount=type=cache,id=aerith-uv-esm,target=/root/.cache/uv \
    --mount=type=cache,id=aerith-conda-packages,target=/opt/conda/pkgs \
    /opt/conda/bin/conda create -y -n esm \
      --file /opt/locks/esm.conda-linux-64.lock \
    && uv pip install --python /opt/conda/envs/esm/bin/python \
      --no-deps --requirement /opt/locks/esm.pip.lock \
    && uv pip install --python /opt/conda/envs/esm/bin/python \
      --no-build-isolation --no-deps /opt/apps/esm
RUN cd /opt/apps/openfold-2.2.0 \
    && CONDA_PREFIX=/opt/conda/envs/esm \
      CUDA_HOME=/opt/conda/envs/esm \
      CC=/opt/conda/envs/esm/bin/x86_64-conda-linux-gnu-gcc \
      CXX=/opt/conda/envs/esm/bin/x86_64-conda-linux-gnu-g++ \
      CUDAHOSTCXX=/opt/conda/envs/esm/bin/x86_64-conda-linux-gnu-g++ \
      MAX_JOBS=8 \
      PATH=/opt/conda/envs/esm/bin:/usr/local/cuda-12.6/bin:/hmmer/bin:/usr/local/bin:/opt/conda/bin:/usr/bin:/bin \
      uv pip install --python /opt/conda/envs/esm/bin/python \
        --no-build-isolation --no-deps /opt/apps/openfold-2.2.0

RUN CUDA_HOME=/usr/local/cuda-12.6 \
    CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
    CPATH=/opt/conda/envs/protenix/lib/python3.12/site-packages/nvidia/cublas/include:/opt/conda/envs/protenix/lib/python3.12/site-packages/nvidia/cusparse/include:/opt/conda/envs/protenix/lib/python3.12/site-packages/nvidia/cusolver/include \
    MAX_JOBS=8 /opt/conda/bin/conda run --no-capture-output -n protenix \
      python -c 'from protenix.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None'
RUN /opt/conda/bin/conda run -n protenix protenix --help >/dev/null \
    && /opt/conda/bin/conda run -n esm python -c 'import esm' \
    && /opt/conda/bin/conda clean --all --yes \
    && rm -rf /root/.cache /tmp/* /var/tmp/* \
    && find /opt/apps -type d \( -name __pycache__ -o -name .pytest_cache \) \
      -prune -exec rm -rf {} + \
    && find /opt/apps -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete

FROM scratch AS conda-component
ARG CONDA_COMPONENT_SHA256=unavailable
LABEL org.aerith.runtime.component.conda.sha256="${CONDA_COMPONENT_SHA256}"
COPY --from=conda-builder /opt/conda /opt/conda
COPY --from=conda-builder /opt/apps/protenix /opt/apps/protenix
COPY --from=conda-builder /opt/apps/esm /opt/apps/esm
COPY --from=conda-builder /opt/apps/openfold-2.2.0 /opt/apps/openfold-2.2.0


FROM ${CUDA_BASE} AS runtime-base
ARG UBUNTU_SNAPSHOT
ARG MMSEQS_VERSION=8cc5ce367b5638c4306c2d7cfc652dd099a4643f
ARG FOLDSEEK_VERSION=941cd33ff0771cd2e3f144e3293e22a2b87e9fda
ARG SHARED_COMPONENT_SHA256=unavailable

LABEL org.aerith.runtime.component.shared.sha256="${SHARED_COMPONENT_SHA256}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    CUTLASS_PATH=/opt/cutlass \
    LAYERNORM_TYPE=fast_layernorm \
    XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_CLIENT_MEM_FRACTION=0.95 \
    PATH=/hmmer/bin:/usr/local/bin:/opt/conda/bin:/usr/bin:/bin

RUN --mount=type=cache,id=aerith-apt-cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,id=aerith-apt-lists,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && sed -ri "s#http://(archive|security).ubuntu.com/ubuntu/#https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/#g" \
      /etc/apt/sources.list.d/ubuntu.sources \
    && printf 'Acquire::Check-Valid-Until "false";\n' \
      > /etc/apt/apt.conf.d/99aerith-snapshot \
    && grep -F "https://snapshot.ubuntu.com/ubuntu/${UBUNTU_SNAPSHOT}/" \
      /etc/apt/sources.list.d/ubuntu.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      ca-certificates gcc g++ kalign libc6-dev libglib2.0-0 libgl1 libgomp1 \
      libstdc++6 libxrender1 make python3.12 zlib1g

COPY --from=tool-builder /hmmer /hmmer
COPY --from=tool-builder /opt/mmseqs /opt/mmseqs
COPY --from=tool-builder /opt/foldseek /opt/foldseek
COPY --from=tool-builder /opt/aerith /opt/aerith
COPY --from=tool-builder /usr/local/bin/fold-runtime /usr/local/bin/fold-runtime
COPY --from=tool-builder /usr/local/bin/validate-fold-runtime /usr/local/bin/validate-fold-runtime
RUN ln -sf /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
    && ln -sf /opt/foldseek/bin/foldseek /usr/local/bin/foldseek \
    && test "$(mmseqs version)" = "${MMSEQS_VERSION}" \
    && test "$(foldseek version)" = "${FOLDSEEK_VERSION}"

ENTRYPOINT ["/usr/local/bin/fold-runtime"]
CMD []


FROM runtime-base AS fold-runtime
ARG MMSEQS_VERSION=8cc5ce367b5638c4306c2d7cfc652dd099a4643f
ARG MMSEQS_SHA256=83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7
ARG FOLDSEEK_VERSION=941cd33ff0771cd2e3f144e3293e22a2b87e9fda
ARG FOLDSEEK_SHA256=af7a688ffd8625b356c380380fb5650ec811262a2d18bdb0faeda95cc4894a55

COPY --from=uv-component /opt/envs /opt/envs
COPY --from=uv-component /opt/uv-python /opt/uv-python
COPY --from=uv-component /opt/apps/alphafold3 /opt/apps/alphafold3
COPY --from=uv-component /opt/apps/opendde /opt/apps/opendde
COPY --from=conda-component /opt/conda /opt/conda
COPY --from=conda-component /opt/apps/protenix /opt/apps/protenix
COPY --from=conda-component /opt/apps/esm /opt/apps/esm
COPY --from=conda-component /opt/apps/openfold-2.2.0 /opt/apps/openfold-2.2.0

# Build-only compilers are absent from the content components. OpenFold's
# conda CUDA payload is removed explicitly while AF3's ptxas remains.
RUN rm -f /opt/conda/envs/esm/bin/nvcc \
      /opt/conda/envs/esm/bin/ptxas \
      /opt/conda/envs/esm/bin/fatbinary \
    && rm -rf /opt/conda/envs/esm/nvvm \
    && rm -f /opt/conda/envs/esm/lib/libnvvm.so* \
    && validate-fold-runtime "${MMSEQS_VERSION}" "${FOLDSEEK_VERSION}"

ARG AF3_VERSION=3.0.3
ARG PROTENIX_VERSION=2.0.0
ARG AF3_COMMIT=7b197fe859790fc3e04d03ea70dd0b9ba48881c9
ARG PROTENIX_COMMIT=2475421477ab414b571149ad4a875c390ff8a35d
ARG OPENDDE_COMMIT=266ce4c49d492ad1077866000d83704999985f46
ARG ESM_COMMIT=2b369911bb5b4b0dda914521b9475cad1656b2ac
ARG OPENFOLD_COMMIT=e938c184a291bf053af3b14c1e3e8bb29aee57e2
ARG RUNTIME_LOCK_SHA256=unavailable
ARG RUNTIME_RECIPE_SHA256=unavailable
ARG RUNTIME_SOURCE_LOCK_SHA256=unavailable
ARG RUNTIME_SOURCE_BUNDLE_SHA256=unavailable
ARG RUNTIME_SOURCE_DIRTY=unavailable
ARG UV_COMPONENT_SHA256=unavailable
ARG CONDA_COMPONENT_SHA256=unavailable
ARG SHARED_COMPONENT_SHA256=unavailable
ARG UV_COMPONENT_IMAGE_DIGEST=unavailable
ARG CONDA_COMPONENT_IMAGE_DIGEST=unavailable
ARG RUNTIME_BASE_IMAGE_DIGEST=unavailable
ARG AF3_SOURCE_SHA256=unavailable
ARG PROTENIX_SOURCE_SHA256=unavailable
ARG OPENDDE_SOURCE_SHA256=unavailable
ARG ESM_SOURCE_SHA256=unavailable
ARG OPENFOLD_SOURCE_SHA256=unavailable
LABEL org.opencontainers.image.title="Aerith unified fold runtime" \
      org.opencontainers.image.base.digest="sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e" \
      org.aerith.runtime.ubuntu-snapshot="${UBUNTU_SNAPSHOT}" \
      org.opencontainers.image.version="af3-${AF3_VERSION}_protenix-${PROTENIX_VERSION}" \
      org.opencontainers.image.revision.af3="${AF3_COMMIT}" \
      org.opencontainers.image.revision.protenix="${PROTENIX_COMMIT}" \
      org.opencontainers.image.revision.opendde="${OPENDDE_COMMIT}" \
      org.opencontainers.image.revision.esm="${ESM_COMMIT}" \
      org.opencontainers.image.revision.openfold="${OPENFOLD_COMMIT}" \
      org.opencontainers.image.revision.mmseqs="${MMSEQS_VERSION}" \
      org.opencontainers.image.mmseqs.sha256="${MMSEQS_SHA256}" \
      org.opencontainers.image.revision.foldseek="${FOLDSEEK_VERSION}" \
      org.opencontainers.image.foldseek.sha256="${FOLDSEEK_SHA256}" \
      org.opencontainers.image.runtime-lock.sha256="${RUNTIME_LOCK_SHA256}" \
      org.aerith.runtime.recipe.sha256="${RUNTIME_RECIPE_SHA256}" \
      org.aerith.runtime.source-lock.sha256="${RUNTIME_SOURCE_LOCK_SHA256}" \
      org.aerith.runtime.source-bundle.sha256="${RUNTIME_SOURCE_BUNDLE_SHA256}" \
      org.aerith.runtime.source.dirty="${RUNTIME_SOURCE_DIRTY}" \
      org.aerith.runtime.component.uv.sha256="${UV_COMPONENT_SHA256}" \
      org.aerith.runtime.component.conda.sha256="${CONDA_COMPONENT_SHA256}" \
      org.aerith.runtime.component.shared.sha256="${SHARED_COMPONENT_SHA256}" \
      org.aerith.runtime.component.uv.image-digest="${UV_COMPONENT_IMAGE_DIGEST}" \
      org.aerith.runtime.component.conda.image-digest="${CONDA_COMPONENT_IMAGE_DIGEST}" \
      org.aerith.runtime.component.runtime-base.image-digest="${RUNTIME_BASE_IMAGE_DIGEST}" \
      org.aerith.runtime.source.af3.sha256="${AF3_SOURCE_SHA256}" \
      org.aerith.runtime.source.protenix.sha256="${PROTENIX_SOURCE_SHA256}" \
      org.aerith.runtime.source.opendde.sha256="${OPENDDE_SOURCE_SHA256}" \
      org.aerith.runtime.source.esm.sha256="${ESM_SOURCE_SHA256}" \
      org.aerith.runtime.source.openfold.sha256="${OPENFOLD_SOURCE_SHA256}"
