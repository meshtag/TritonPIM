#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

NR_DPUS="${NR_DPUS:-1}"

dpu-upmem-dpurte-clang -O2 -g0 -x ir -c "${ROOT_DIR}/bin/triton_add.ll" \
  -o "${ROOT_DIR}/bin/triton_add.o"

dpu-upmem-dpurte-clang -O2 -g0 -x ir -c "${ROOT_DIR}/dpu/triton_wrapper.ll" \
  -o "${ROOT_DIR}/bin/triton_wrapper.o"

dpu-upmem-dpurte-clang -O2 -g0 -c "${ROOT_DIR}/dpu/triton_args.c" \
  -o "${ROOT_DIR}/bin/triton_args.o"

/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame \
  "${ROOT_DIR}/bin/triton_add.o"
/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame \
  "${ROOT_DIR}/bin/triton_wrapper.o"
/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame \
  "${ROOT_DIR}/bin/triton_args.o"

dpu-upmem-dpurte-clang \
  "${ROOT_DIR}/bin/triton_add.o" \
  "${ROOT_DIR}/bin/triton_wrapper.o" \
  "${ROOT_DIR}/bin/triton_args.o" \
  -o "${ROOT_DIR}/bin/vec_add_triton"

cc -O2 -std=c11 -Wall -Wextra \
  -o "${ROOT_DIR}/bin/vec_add_triton_host" \
  "${ROOT_DIR}/host/vec_add_triton_host.c" \
  `dpu-pkg-config --cflags --libs dpu` \
  -DNR_DPUS="${NR_DPUS}"

"${ROOT_DIR}/bin/vec_add_triton_host"
