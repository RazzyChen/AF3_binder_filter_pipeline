ARG CUDA_BASE=nvcr.io/nvidia/cuda:12.6.3-base-ubuntu24.04@sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e
FROM ${CUDA_BASE} AS builder

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
    PATH=/hmmer/bin:/usr/local/bin:/opt/conda/bin:/usr/bin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl git kalign libglib2.0-0 \
      libgl1 libxrender1 make patch python3-pip python3.12 python3.12-dev wget zlib1g-dev zstd \
    && rm -rf /var/lib/apt/lists/*

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

# Source trees are BuildKit named contexts, not copies from arbitrary host paths.
COPY --from=af3-src / /opt/apps/alphafold3
COPY --from=opendde-src / /opt/apps/opendde
COPY --from=protenix-src / /opt/apps/protenix
COPY --from=esm-src / /opt/apps/esm
COPY docker/runtime/locks /opt/locks

# AF3 relies on jackhmmer's --seq_limit extension, which is not present in the
# Ubuntu HMMER package. Build the same patched HMMER 3.4 used by AF3 upstream.
COPY --from=af3-src /docker/jackhmmer_seq_limit.patch /tmp/jackhmmer_seq_limit.patch
RUN mkdir -p /tmp/hmmer-build /hmmer \
    && wget -q http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -O /tmp/hmmer-build/hmmer.tar.gz \
    && echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  /tmp/hmmer-build/hmmer.tar.gz' | sha256sum -c - \
    && tar -xzf /tmp/hmmer-build/hmmer.tar.gz -C /tmp/hmmer-build \
    && cd /tmp/hmmer-build \
    && patch -p0 < /tmp/jackhmmer_seq_limit.patch \
    && cd hmmer-3.4 \
    && ./configure --prefix=/hmmer \
    && make -j"$(nproc)" \
    && make install \
    && cd easel && make install \
    && rm -rf /tmp/hmmer-build /tmp/jackhmmer_seq_limit.patch

RUN --mount=type=cache,id=aerith-uv-af3,target=/root/.cache/uv \
    uv venv --python python3.12 /opt/envs/af3 \
    && cd /opt/apps/alphafold3 \
    && UV_PROJECT_ENVIRONMENT=/opt/envs/af3 uv sync --frozen --no-editable
RUN /opt/envs/af3/bin/build_data

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
RUN --mount=type=cache,id=aerith-uv-opendde,target=/root/.cache/uv \
    uv python install 3.11.14 \
    && uv venv --python 3.11.14 /opt/envs/opendde \
    && uv pip install --python /opt/envs/opendde/bin/python \
         --no-deps --torch-backend cu126 --requirement /opt/locks/opendde.pip.lock \
    && uv pip install --python /opt/envs/opendde/bin/python \
         --no-deps /opt/apps/opendde

# Protenix and ESM deliberately remain separate conda environments even though
# they live in one image.  This prevents their incompatible torch stacks from
# leaking into AF3/OpenDDE.
RUN --mount=type=cache,id=aerith-uv-opendde,target=/root/.cache/uv \
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
         --no-deps --requirement /opt/locks/esm.pip.lock
RUN --mount=type=cache,id=aerith-uv-esm,target=/root/.cache/uv \
    uv pip install --python /opt/conda/envs/esm/bin/python \
         --no-build-isolation --no-deps /opt/apps/esm
COPY docker/runtime/openfold-cuda-11.6.patch /tmp/openfold-cuda-11.6.patch
RUN cd /opt/apps/esm/openfold-2.2.0 \
    && patch -p1 < /tmp/openfold-cuda-11.6.patch \
    && CONDA_PREFIX=/opt/conda/envs/esm \
       CUDA_HOME=/opt/conda/envs/esm \
       CC=/opt/conda/envs/esm/bin/x86_64-conda-linux-gnu-gcc \
       CXX=/opt/conda/envs/esm/bin/x86_64-conda-linux-gnu-g++ \
       CUDAHOSTCXX=/opt/conda/envs/esm/bin/x86_64-conda-linux-gnu-g++ \
       MAX_JOBS=8 \
       PATH=/opt/conda/envs/esm/bin:/hmmer/bin:/usr/local/bin:/opt/conda/bin:/usr/bin:/bin \
       uv pip install --python /opt/conda/envs/esm/bin/python \
         --no-build-isolation --no-deps /opt/apps/esm/openfold-2.2.0 \
    && rm /tmp/openfold-cuda-11.6.patch

# Both Protenix and OpenDDE use fused CUDA layer-normalization. Compile the
# extensions in the builder so inference never needs nvcc or a host toolkit.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         cuda-cccl-12-6=12.6.77-1 \
         cuda-crt-12-6=12.6.85-1 \
         cuda-cudart-dev-12-6=12.6.77-1 \
         cuda-driver-dev-12-6=12.6.77-1 \
         cuda-nvcc-12-6=12.6.85-1 \
         cuda-nvvm-12-6=12.6.85-1 \
    && rm -rf /var/lib/apt/lists/*
# Torch CUDA headers include cuBLAS, cuSPARSE, and cuSOLVER. Reuse the exactly
# pinned NVIDIA wheel headers already present in each environment.
RUN CUDA_HOME=/usr/local/cuda-12.6 \
    CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
    PATH=/usr/local/cuda-12.6/bin:${PATH} \
    CPATH=/opt/conda/envs/protenix/lib/python3.12/site-packages/nvidia/cublas/include:/opt/conda/envs/protenix/lib/python3.12/site-packages/nvidia/cusparse/include:/opt/conda/envs/protenix/lib/python3.12/site-packages/nvidia/cusolver/include \
    MAX_JOBS=8 /opt/conda/bin/conda run --no-capture-output -n protenix \
      python -c 'from protenix.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None'
RUN CUDA_HOME=/usr/local/cuda-12.6 \
    CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
    PATH=/usr/local/cuda-12.6/bin:${PATH} \
    CPATH=/opt/envs/opendde/lib/python3.11/site-packages/nvidia/cublas/include:/opt/envs/opendde/lib/python3.11/site-packages/nvidia/cusparse/include:/opt/envs/opendde/lib/python3.11/site-packages/nvidia/cusolver/include \
    LAYERNORM_TYPE=fast_layernorm \
    MAX_JOBS=8 /opt/envs/opendde/bin/python -c \
      'from opendde.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None'

# Install the official GPU releases requested for the offline feature and
# clustering stages. Keep their data directories beside each binary because
# both tools resolve bundled matrices relative to the installation tree.
ARG MMSEQS_RELEASE=18-8cc5c
ARG MMSEQS_VERSION=8cc5ce367b5638c4306c2d7cfc652dd099a4643f
ARG MMSEQS_SHA256=83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7
ARG FOLDSEEK_RELEASE=10-941cd33
ARG FOLDSEEK_VERSION=941cd33ff0771cd2e3f144e3293e22a2b87e9fda
ARG FOLDSEEK_SHA256=af7a688ffd8625b356c380380fb5650ec811262a2d18bdb0faeda95cc4894a55
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
RUN chmod 0755 /opt/aerith/mmseqs_wrapper.py \
    && test "$(mmseqs version)" = "${MMSEQS_VERSION}" \
    && /opt/envs/opendde/bin/python /opt/aerith/build_local_features.py --help >/dev/null \
    && /opt/envs/af3/bin/python /opt/aerith/convert_af3_templates.py --help >/dev/null \
    && /opt/envs/opendde/bin/colabfold_search --help >/dev/null

COPY docker/runtime/entrypoint.sh /usr/local/bin/fold-runtime
COPY docker/runtime/esm_if_batch.py /opt/aerith/esm_if_batch.py
RUN chmod 0755 /usr/local/bin/fold-runtime \
    && /opt/envs/af3/bin/python -c 'import alphafold3' \
    && /opt/envs/opendde/bin/opendde --help >/dev/null \
    && /opt/conda/bin/conda run -n protenix protenix --help >/dev/null \
    && /opt/conda/bin/conda run -n esm python -c 'import esm'

# Remove build-only package caches before copying artifacts to the final image.
RUN /opt/conda/bin/conda clean --all --yes \
    && rm -rf /root/.cache /tmp/* /var/tmp/* \
    && find /opt/apps -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} + \
    && find /opt/apps -type f \( -name '*.pyc' -o -name '.DS_Store' \) -delete
ENTRYPOINT ["/usr/local/bin/fold-runtime"]
CMD []

FROM ${CUDA_BASE} AS runtime

# The final image retains only runtime libraries and host-side C/C++ support
# for framework JITs. CUDA compilation remains exclusively in the builder.
ARG MMSEQS_VERSION=8cc5ce367b5638c4306c2d7cfc652dd099a4643f
ARG MMSEQS_SHA256=83969dd5c7d4c32858c2fc9a4d1024c15e8fe5da768ce76e787ab0195ffd64e7
ARG FOLDSEEK_VERSION=941cd33ff0771cd2e3f144e3293e22a2b87e9fda
ARG FOLDSEEK_SHA256=af7a688ffd8625b356c380380fb5650ec811262a2d18bdb0faeda95cc4894a55

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    CUTLASS_PATH=/opt/cutlass \
    LAYERNORM_TYPE=fast_layernorm \
    XLA_FLAGS=--xla_gpu_enable_triton_gemm=false \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_CLIENT_MEM_FRACTION=0.95 \
    PATH=/hmmer/bin:/usr/local/bin:/opt/conda/bin:/usr/bin:/bin

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates gcc g++ kalign libc6-dev libglib2.0-0 libgl1 libgomp1 \
      libstdc++6 libxrender1 make python3.12 zlib1g \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /hmmer /hmmer
COPY --from=builder /opt/conda /opt/conda
COPY --from=builder /opt/uv-python /opt/uv-python
COPY --from=builder /opt/envs /opt/envs
COPY --from=builder /opt/apps /opt/apps
COPY --from=builder /opt/mmseqs /opt/mmseqs
COPY --from=builder /opt/foldseek /opt/foldseek
COPY --from=builder /opt/aerith /opt/aerith
COPY --from=builder /usr/local/bin/fold-runtime /usr/local/bin/fold-runtime

# OpenFold is compiled during the build. Remove its conda CUDA compiler payload
# while retaining AF3's ptxas and Protenix Triton runtime compiler helper.
RUN rm -f /opt/conda/envs/esm/bin/nvcc \
          /opt/conda/envs/esm/bin/ptxas \
          /opt/conda/envs/esm/bin/fatbinary \
    && rm -rf /opt/conda/envs/esm/nvvm \
    && rm -f /opt/conda/envs/esm/lib/libnvvm.so* \
    && ln -sf /opt/mmseqs/bin/mmseqs /usr/local/bin/mmseqs \
    && ln -sf /opt/foldseek/bin/foldseek /usr/local/bin/foldseek \
    && chmod 0755 /usr/local/bin/fold-runtime \
    && test "$(mmseqs version)" = "${MMSEQS_VERSION}" \
    && test "$(foldseek version)" = "${FOLDSEEK_VERSION}" \
    && test ! -e /opt/conda/envs/esm/bin/nvcc \
    && test ! -e /usr/local/cuda-12.6/bin/nvcc \
    && test -n "$(find /opt/envs/af3 -type f -name ptxas -print -quit)" \
    && /opt/envs/af3/bin/python -c 'import alphafold3' \
    && /opt/envs/opendde/bin/python -c 'from opendde.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None; import opendde' \
    && /opt/conda/bin/conda run --no-capture-output -n protenix python -c 'from protenix.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None; import protenix' \
    && /opt/conda/bin/conda run --no-capture-output -n esm python -c 'import esm'

ENTRYPOINT ["/usr/local/bin/fold-runtime"]
CMD []
ARG AF3_VERSION=3.0.3
ARG PROTENIX_VERSION=2.0.0
ARG OPENDDE_COMMIT=266ce4c49d492ad1077866000d83704999985f46
ARG ESM_COMMIT=2b369911bb5b4b0dda914521b9475cad1656b2ac
ARG RUNTIME_LOCK_SHA256=unavailable
ARG RUNTIME_RECIPE_SHA256=unavailable
ARG RUNTIME_SOURCE_BUNDLE_SHA256=unavailable
ARG RUNTIME_SOURCE_DIRTY=unavailable
ARG AF3_SOURCE_SHA256=unavailable
ARG PROTENIX_SOURCE_SHA256=unavailable
ARG OPENDDE_SOURCE_SHA256=unavailable
ARG ESM_SOURCE_SHA256=unavailable
LABEL org.opencontainers.image.title="Aerith unified fold runtime" \
      org.opencontainers.image.base.digest="sha256:c87e78933f4c16e3272123bf2f75537306596d0fbaa395a29696a22786e5ee0e" \
      org.opencontainers.image.version="af3-${AF3_VERSION}_protenix-${PROTENIX_VERSION}" \
      org.opencontainers.image.revision.opendde="${OPENDDE_COMMIT}" \
      org.opencontainers.image.revision.esm="${ESM_COMMIT}" \
      org.opencontainers.image.revision.mmseqs="${MMSEQS_VERSION}" \
      org.opencontainers.image.mmseqs.sha256="${MMSEQS_SHA256}" \
      org.opencontainers.image.revision.foldseek="${FOLDSEEK_VERSION}" \
      org.opencontainers.image.foldseek.sha256="${FOLDSEEK_SHA256}" \
      org.opencontainers.image.runtime-lock.sha256="${RUNTIME_LOCK_SHA256}" \
      org.aerith.runtime.recipe.sha256="${RUNTIME_RECIPE_SHA256}" \
      org.aerith.runtime.source-bundle.sha256="${RUNTIME_SOURCE_BUNDLE_SHA256}" \
      org.aerith.runtime.source.dirty="${RUNTIME_SOURCE_DIRTY}" \
      org.aerith.runtime.source.af3.sha256="${AF3_SOURCE_SHA256}" \
      org.aerith.runtime.source.protenix.sha256="${PROTENIX_SOURCE_SHA256}" \
      org.aerith.runtime.source.opendde.sha256="${OPENDDE_SOURCE_SHA256}" \
      org.aerith.runtime.source.esm.sha256="${ESM_SOURCE_SHA256}"
