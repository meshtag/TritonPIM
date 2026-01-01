# Examples

This directory contains Triton-based examples that compile to LLVM IR for the
UPMEM DPU flow in this repo. They are successfully run and tested on the UPMEM
simulator for functional execution.

## Required third_party branches (build these first)
These examples depend on DPU-specific patches. Use the branches already vendored
under `third_party/`:

- Triton: `third_party/triton` on branch `prathamesh_triton_pim`
  ```
  cd third_party/triton
  # optional venv
  python -m venv .venv
  source .venv/bin/activate
  pip install -r python/requirements.txt
  TRITON_BUILD_WITH_CLANG_LLD=true pip install -e . --no-build-isolation
  ```
  Then set `TRITON_SRC` and (if using a venv) `TRITON_PY`.

- UPMEM LLVM: `third_party/upmem_llvm/llvm-project` (branch `prathamesh_triton_pim`)
  Build `opt`/`mlir-opt` with the `dpu-legalize` pass:
  ```
  cd third_party/upmem_llvm/llvm-project
  cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host" \
    -B build llvm
  ninja -C build opt mlir-opt mlir-translate
  ```
  Then set `UPMEM_OPT=third_party/upmem_llvm/llvm-project/build/bin/opt`.

## AXPY (Triton -> UPMEM simulator)

`axpy_dpu.py` compiles an integer AXPY kernel:

```
Y = A * X + Y
```

### Host side: auto-generate artifact + host runner

`host_triton_pack.py` generates the DPU wrapper, args structs, and a host runner
for you. You do not need to write any host code.

From the repo root:

```
export TRITON_SRC=/path/to/triton
export UPMEM_OPT=/path/to/upmem_llvm/llvm-project/build/bin/opt
# optional
# export TRITON_PY=/path/to/venv/bin/python

./scripts/host_triton_pack.py \
  --artifact-dir ./artifacts/axpy \
  --triton-script ./examples/axpy_dpu.py \
  --kernel axpy_kernel \
  --out-indices 1
```

Generated files (under `artifacts/axpy`):
- `kernel.ll`
- `wrapper.ll`
- `triton_args.h`
- `triton_args.c`
- `host_args.h`
- `host_runner.c`
- `meta.json`

### Container: build and run on the simulator

```
./container/start_docker.sh
```

Inside the container:

```
cd /mnt/host_cwd
ARTIFACT_DIR=/mnt/host_cwd/artifacts/axpy ./scripts/container_artifact_run.sh \
  --arg 2=3 \
  --arg 3=1024 \
  --len 0=1024 \
  --len 1=1024 \
  --out 1=/tmp/axpy_y.bin
```

Notes:
- Arg indices are expected to be: X=0, Y=1, A=2, N=3. If they differ, check
  `artifacts/axpy/meta.json` and adjust `--arg`/`--len` accordingly.
- The generated host runner accepts integer scalars only, so `A` is an integer.
- DPU count: set `TRITON_DPU_NUM_DPUS` in the Triton script for a default, or
  override at runtime with `--nr-dpus N`.
- Tensor size controls:
  - `--arg 3=<N>` sets the scalar `N` (length) passed to the kernel.
  - `--len <IDX>=<N>` sets the element count for pointer arg `<IDX>` (overrides defaults).
  - The host runner prints the first 10 results per DPU.
