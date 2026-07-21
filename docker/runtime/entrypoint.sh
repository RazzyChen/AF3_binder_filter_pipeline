#!/usr/bin/env bash
set -euo pipefail

tool="${1:-}"
if [[ -z "${tool}" ]]; then
  echo "usage: fold-runtime <af3|protenix|opendde|esmfold|esm-if|foldseek|prepare-features> [args...]" >&2
  exit 64
fi
shift

case "${tool}" in
  doctor)
    test "$(mmseqs version)" = "8cc5ce367b5638c4306c2d7cfc652dd099a4643f"
    mmseqs gpuserver -h >/dev/null
    test "$(foldseek version)" = "941cd33ff0771cd2e3f144e3293e22a2b87e9fda"
    foldseek easy-cluster -h >/dev/null
    /opt/envs/opendde/bin/python /opt/aerith/build_local_features.py --help >/dev/null
    echo "offline feature tools ok"
    /opt/envs/af3/bin/python -c 'import alphafold3; print("af3 import ok")'
    /opt/envs/opendde/bin/python -c 'import opendde, torch; assert torch.cuda.is_available(); print("opendde cuda ok")'
    /opt/conda/bin/conda run --no-capture-output -n protenix python -c 'import protenix, torch; assert torch.cuda.is_available(); print("protenix cuda ok")'
    /opt/conda/bin/conda run --no-capture-output -n esm python -c 'import esm, torch; assert torch.cuda.is_available(); print("esm cuda ok")'
    ;;
  af3)
    exec /opt/envs/af3/bin/python /opt/apps/alphafold3/run_alphafold.py "$@"
    ;;
  protenix)
    exec /opt/conda/bin/conda run --no-capture-output -n protenix protenix "$@"
    ;;
  opendde)
    exec /opt/envs/opendde/bin/opendde "$@"
    ;;
  esmfold)
    exec /opt/conda/bin/conda run --no-capture-output -n esm esm-fold "$@"
    ;;
  esm-if)
    exec /opt/conda/bin/conda run --no-capture-output -n esm python /opt/aerith/esm_if_batch.py "$@"
    ;;
  foldseek)
    exec /usr/local/bin/foldseek "$@"
    ;;
  prepare-features)
    exec /opt/envs/opendde/bin/python /opt/aerith/build_local_features.py "$@"
    ;;
  *)
    echo "unknown fold-runtime tool: ${tool}" >&2
    exit 64
    ;;
esac
