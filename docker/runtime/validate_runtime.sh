#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
  echo "usage: validate-fold-runtime MMSEQS_VERSION FOLDSEEK_VERSION" >&2
  exit 2
fi

test "$(mmseqs version)" = "$1"
test "$(foldseek version)" = "$2"
test ! -e /opt/conda/envs/esm/bin/nvcc
test ! -e /usr/local/cuda-12.6/bin/nvcc
test -n "$(find /opt/envs/af3 -type f -name ptxas -print -quit)"
/opt/envs/af3/bin/python -c 'import alphafold3'
/opt/envs/opendde/bin/python -c \
  'from opendde.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None; import opendde'
/opt/conda/bin/conda run --no-capture-output -n protenix python -c \
  'from protenix.model.layer_norm import layer_norm; assert layer_norm.fast_layer_norm_cuda_v2 is not None; import protenix'
/opt/conda/bin/conda run --no-capture-output -n esm python -c 'import esm'
