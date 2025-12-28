#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${ROOT_DIR}"

TRITON_PY="${TRITON_PY:-python3}"
TRITON_SRC="${TRITON_SRC:-${PROJECT_ROOT}/third_party/triton}"
UPMEM_OPT="${UPMEM_OPT:-${PROJECT_ROOT}/third_party/upmem_llvm/llvm-project/build/bin/opt}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${ROOT_DIR}/.triton_cache}"
OUT_LL="${OUT_LL:-${ROOT_DIR}/bin/triton_add.ll}"
KERNEL_NAME="${KERNEL_NAME:-add_kernel}"

mkdir -p "${ROOT_DIR}/bin" "${TRITON_CACHE_DIR}"

export PYTHONPATH="${TRITON_SRC}/python"
export TRITON_BACKENDS_IN_TREE=1
export TRITON_DPU=1
export TRITON_DPU_OPT="${UPMEM_OPT}"
export TRITON_CACHE_DIR
unset TRITON_DPU_SKIP_LEGALIZE
export TRITON_DPU_FORCE_LEGALIZE=1

"${TRITON_PY}" "${TRITON_SRC}/python/triton/backends/dpu/dpu_min_test.py" --out "${OUT_LL}"
"${TRITON_PY}" "${ROOT_DIR}/scripts/generate_triton_wrapper.py" \
  --ir "${OUT_LL}" \
  --out "${ROOT_DIR}/dpu/triton_wrapper.ll" \
  --kernel "${KERNEL_NAME}" \
  --dpu-args-h "${ROOT_DIR}/dpu/triton_args.h" \
  --dpu-args-c "${ROOT_DIR}/dpu/triton_args.c" \
  --host-args-h "${ROOT_DIR}/host/triton_args.h"
echo "Wrote Triton LLVM IR to ${OUT_LL}"
