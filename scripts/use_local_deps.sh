#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'USAGE'
Usage: use_local_deps.sh [--link] [--triton PATH] [--upmem PATH] [--upmem-opt PATH] [--python PATH] [--print-env]

Modes:
  --link       Create/refresh symlinks under third_party/ using --triton/--upmem.
  --print-env  Print export lines instead of exporting (useful for eval).

Examples:
  # Create symlinks
  scripts/use_local_deps.sh --link --triton /path/to/triton --upmem /path/to/upmem_llvm

  # Export env vars in current shell
  source scripts/use_local_deps.sh --triton /path/to/triton --upmem /path/to/upmem_llvm

  # Print exports (for eval)
  scripts/use_local_deps.sh --print-env --triton /path/to/triton --upmem /path/to/upmem_llvm
USAGE
}

LINK=0
PRINT_ENV=0
TRITON_PATH=""
UPMEM_PATH=""
UPMEM_OPT_PATH=""
PY_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --link)
      LINK=1
      ;;
    --print-env)
      PRINT_ENV=1
      ;;
    --triton)
      TRITON_PATH="${2:-}"
      shift
      ;;
    --upmem)
      UPMEM_PATH="${2:-}"
      shift
      ;;
    --upmem-opt)
      UPMEM_OPT_PATH="${2:-}"
      shift
      ;;
    --python)
      PY_PATH="${2:-}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
  shift
done

if [[ ${LINK} -eq 1 ]]; then
  if [[ -z "${TRITON_PATH}" || -z "${UPMEM_PATH}" ]]; then
    echo "--link requires both --triton and --upmem" >&2
    exit 1
  fi
  mkdir -p "${PROJECT_ROOT}/third_party"
  ln -sfn "${TRITON_PATH}" "${PROJECT_ROOT}/third_party/triton"
  ln -sfn "${UPMEM_PATH}" "${PROJECT_ROOT}/third_party/upmem_llvm"
fi

if [[ -n "${TRITON_PATH}" ]]; then
  TRITON_SRC_VAL="${TRITON_PATH}"
elif [[ -n "${TRITON_SRC:-}" ]]; then
  TRITON_SRC_VAL="${TRITON_SRC}"
elif [[ -e "${PROJECT_ROOT}/third_party/triton" ]]; then
  TRITON_SRC_VAL="${PROJECT_ROOT}/third_party/triton"
else
  TRITON_SRC_VAL=""
fi

if [[ -n "${UPMEM_OPT_PATH}" ]]; then
  UPMEM_OPT_VAL="${UPMEM_OPT_PATH}"
elif [[ -n "${UPMEM_PATH}" ]]; then
  UPMEM_OPT_VAL="${UPMEM_PATH}/llvm-project/build/bin/opt"
elif [[ -n "${UPMEM_OPT:-}" ]]; then
  UPMEM_OPT_VAL="${UPMEM_OPT}"
elif [[ -e "${PROJECT_ROOT}/third_party/upmem_llvm/llvm-project/build/bin/opt" ]]; then
  UPMEM_OPT_VAL="${PROJECT_ROOT}/third_party/upmem_llvm/llvm-project/build/bin/opt"
else
  UPMEM_OPT_VAL=""
fi

if [[ -n "${PY_PATH}" ]]; then
  TRITON_PY_VAL="${PY_PATH}"
elif [[ -n "${TRITON_PY:-}" ]]; then
  TRITON_PY_VAL="${TRITON_PY}"
else
  TRITON_PY_VAL="python3"
fi

TRITON_CACHE_VAL="${TRITON_CACHE_DIR:-${PROJECT_ROOT}/.triton_cache}"

if [[ ${PRINT_ENV} -eq 1 || "${BASH_SOURCE[0]}" == "$0" ]]; then
  [[ -n "${TRITON_SRC_VAL}" ]] && echo "export TRITON_SRC=\"${TRITON_SRC_VAL}\""
  [[ -n "${UPMEM_OPT_VAL}" ]] && echo "export UPMEM_OPT=\"${UPMEM_OPT_VAL}\""
  [[ -n "${TRITON_PY_VAL}" ]] && echo "export TRITON_PY=\"${TRITON_PY_VAL}\""
  [[ -n "${TRITON_CACHE_VAL}" ]] && echo "export TRITON_CACHE_DIR=\"${TRITON_CACHE_VAL}\""
  exit 0
fi

[[ -n "${TRITON_SRC_VAL}" ]] && export TRITON_SRC="${TRITON_SRC_VAL}"
[[ -n "${UPMEM_OPT_VAL}" ]] && export UPMEM_OPT="${UPMEM_OPT_VAL}"
[[ -n "${TRITON_PY_VAL}" ]] && export TRITON_PY="${TRITON_PY_VAL}"
[[ -n "${TRITON_CACHE_VAL}" ]] && export TRITON_CACHE_DIR="${TRITON_CACHE_VAL}"

echo "Environment set for TritonPIM."
