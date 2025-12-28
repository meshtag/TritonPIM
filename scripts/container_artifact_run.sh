#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ARTIFACT_DIR="${ARTIFACT_DIR:-${ROOT_DIR}/artifacts/add}"
NR_DPUS="${NR_DPUS:-1}"

if [[ ! -d "${ARTIFACT_DIR}" ]]; then
  echo "Artifact dir not found: ${ARTIFACT_DIR}" >&2
  exit 1
fi

KERNEL_LL="${ARTIFACT_DIR}/kernel.ll"
WRAPPER_LL="${ARTIFACT_DIR}/wrapper.ll"
DPU_ARGS_C="${ARTIFACT_DIR}/triton_args.c"
HOST_RUNNER_C="${ARTIFACT_DIR}/host_runner.c"

if [[ ! -f "${KERNEL_LL}" || ! -f "${WRAPPER_LL}" || ! -f "${DPU_ARGS_C}" || ! -f "${HOST_RUNNER_C}" ]]; then
  echo "Missing files in artifact dir. Expected kernel.ll, wrapper.ll, triton_args.c, host_runner.c" >&2
  exit 1
fi

mkdir -p "${ARTIFACT_DIR}/bin"

DPU_BIN="${ARTIFACT_DIR}/bin/dpu.elf"
HOST_BIN="${ARTIFACT_DIR}/bin/host_runner"

# Compile DPU objects
/usr/bin/env dpu-upmem-dpurte-clang -O2 -g0 -x ir -c "${KERNEL_LL}" -o "${ARTIFACT_DIR}/bin/kernel.o"
/usr/bin/env dpu-upmem-dpurte-clang -O2 -g0 -x ir -c "${WRAPPER_LL}" -o "${ARTIFACT_DIR}/bin/wrapper.o"
/usr/bin/env dpu-upmem-dpurte-clang -O2 -g0 -I"${ARTIFACT_DIR}" -c "${DPU_ARGS_C}" -o "${ARTIFACT_DIR}/bin/dpu_args.o"

# Strip .eh_frame to keep MRAM clean
/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame "${ARTIFACT_DIR}/bin/kernel.o"
/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame "${ARTIFACT_DIR}/bin/wrapper.o"
/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame "${ARTIFACT_DIR}/bin/dpu_args.o"

# Link DPU binary
/usr/bin/env dpu-upmem-dpurte-clang \
  "${ARTIFACT_DIR}/bin/kernel.o" \
  "${ARTIFACT_DIR}/bin/wrapper.o" \
  "${ARTIFACT_DIR}/bin/dpu_args.o" \
  -o "${DPU_BIN}"

# Build host runner
cc -O2 -std=c11 -Wall -Wextra -I"${ARTIFACT_DIR}" \
  -o "${HOST_BIN}" "${HOST_RUNNER_C}" \
  `dpu-pkg-config --cflags --libs dpu` \
  -DNR_DPUS=${NR_DPUS} \
  -DDPU_BINARY=\"${DPU_BIN}\"

"${HOST_BIN}" "$@"
