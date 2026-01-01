# TritonPIM: DPU vec_add (from scratch)

This repo contains two complete flows (run from the repo root):

- DPU dialect MLIR -> LLVM IR -> DPU binary -> host runner
- Triton -> LLVM IR -> DPU binary -> host runner

The goal is to be able to run on the UPMEM simulator from a clean checkout.

## Directory layout (what each file is for)
- `dpu/vec_add.mlir`: DPU dialect version of the vector add kernel.
- `dpu/vec_add_args.c`: defines `DPU_INPUT_ARGUMENTS` in the host section.
- `host/vec_add_host.c`: host runner for the DPU dialect flow.
- `dpu_min_test.py`: Triton kernel, now `C = A + B`.
- `dpu/triton_wrapper.ll`: auto-generated DPU entrypoint (`main`) that calls `add_kernel`.
- `dpu/triton_args.h` / `dpu/triton_args.c`: auto-generated DPU args struct + definition.
- `host/triton_args.h`: auto-generated host args struct.
- `host/vec_add_triton_host.c`: host runner for the Triton flow.
- `scripts/host_triton_ir.sh`: host-side script to emit Triton LLVM IR.
- `scripts/container_triton_run.sh`: container-side script to build/run on the simulator.
- `scripts/generate_triton_wrapper.py`: generates `triton_wrapper.ll` + args headers from the Triton IR.
- `scripts/host_triton_pack.py`: host-side packer that creates a self-contained artifact folder.
- `scripts/container_artifact_run.sh`: container-side build/run for artifacts (no per-kernel C edits).
- `scripts/use_local_deps.sh`: helper to link local deps or emit env exports.
- `third_party/`: placeholders for Triton + UPMEM LLVM (can be git submodules).
- `artifacts/`: generated Triton artifact folders (safe to delete/recreate).

## Prerequisites
Host laptop:
- Triton repo (suggested location): `third_party/triton`
- Python venv (example): `.venv/triton_env` or any Python with Triton in `PYTHONPATH`
- UPMEM LLVM build (suggested location): `third_party/upmem_llvm/llvm-project/build`

You can override paths via environment variables:
- `TRITON_PY`, `TRITON_SRC`, `UPMEM_OPT`, `TRITON_CACHE_DIR`
This repo also supports `third_party/` symlinks, but the recommended flow here is to
set the exports directly (no symlinks required).

Container:
- UPMEM SDK installed (provides `dpu-upmem-dpurte-clang`, `dpu-pkg-config`)
- Repo root mounted at `/mnt/host_cwd`

## Required third_party branches (build these first)
These flows rely on DPU-specific patches. Using upstream Triton/LLVM will not
work. Use the provided branches under `third_party/`:

- Triton: `third_party/triton` on branch `prathamesh_triton_pim`
  - Build it locally:
    ```
    cd third_party/triton
    # optional venv
    python -m venv .venv
    source .venv/bin/activate
    pip install -r python/requirements.txt
    TRITON_BUILD_WITH_CLANG_LLD=true pip install -e . --no-build-isolation
    ```
  - Then set `TRITON_SRC` and (if using a venv) `TRITON_PY`.

- UPMEM LLVM: `third_party/upmem_llvm/llvm-project` (branch `main`)
  - Build `opt`/`mlir-opt` with the `dpu-legalize` pass:
    ```
    cd third_party/upmem_llvm/llvm-project
    cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" -DLLVM_TARGETS_TO_BUILD="host" \
      -B build llvm
    ninja -C build opt mlir-opt mlir-translate
    ```
  - Then set `UPMEM_OPT=third_party/upmem_llvm/llvm-project/build/bin/opt`.

## Quickstart (env exports, no symlinks)
From repo root:
```
export TRITON_SRC=/path/to/triton
export UPMEM_OPT=/path/to/upmem_llvm/llvm-project/build/bin/opt
# optional
# export TRITON_PY=/path/to/venv/bin/python

./scripts/host_triton_pack.py --artifact-dir ./artifacts/add --out-indices 2
./container/start_docker.sh
```

Inside the container:
```
cd /mnt/host_cwd
./scripts/container_artifact_run.sh --arg 3=10
```

Artifact flow with multiple DPUs (host + container):
```
# host
./scripts/host_triton_pack.py --artifact-dir ./artifacts/add --out-indices 2
./container/start_docker.sh

# container
ARTIFACT_DIR=/mnt/host_cwd/artifacts/add ./scripts/container_artifact_run.sh \
  --nr-dpus 4 --arg 3=100
```

### Quickstart (from Triton land)
This bypasses the packer and runs the Triton compiler directly, then generates the wrapper/args.
From repo root:
```
export TRITON_SRC=/path/to/triton
export UPMEM_OPT=/path/to/upmem_llvm/llvm-project/build/bin/opt
export TRITON_PY=/path/to/venv/bin/python

$TRITON_PY ./dpu_min_test.py --out bin/triton_add.ll
$TRITON_PY scripts/generate_triton_wrapper.py \
  --ir bin/triton_add.ll \
  --out dpu/triton_wrapper.ll \
  --kernel add_kernel \
  --dpu-args-h dpu/triton_args.h \
  --dpu-args-c dpu/triton_args.c \
  --host-args-h host/triton_args.h

./container/start_docker.sh
```

Inside the container:
```
cd /mnt/host_cwd
./scripts/container_triton_run.sh
```

## Helper script
- `scripts/use_local_deps.sh` can create `third_party` symlinks and/or emit `export` lines for `TRITON_SRC`, `UPMEM_OPT`, `TRITON_PY`, `TRITON_CACHE_DIR`.
  - Example (symlink + export in current shell):
    - `source scripts/use_local_deps.sh --link --triton /path/to/triton --upmem /path/to/upmem_llvm`

## Flow A: DPU dialect MLIR -> DPU binary
This is the original DPU-dialect pipeline.

### A1) Lower MLIR to LLVM IR (host laptop)
Purpose: convert DPU dialect to LLVM IR so the UPMEM toolchain can compile it.
```
third_party/upmem_llvm/llvm-project/build/bin/mlir-opt \
  dpu/vec_add.mlir \
  --convert-dpu-to-llvm \
  -o bin/vec_add_llvm.mlir

third_party/upmem_llvm/llvm-project/build/bin/mlir-translate \
  --mlir-to-llvmir \
  bin/vec_add_llvm.mlir \
  -o bin/vec_add.ll
```

### A2) Build DPU binary (container)
Purpose: compile LLVM IR + host arguments into a DPU executable.
```
dpu-upmem-dpurte-clang -O2 -g0 -x ir -c /mnt/host_cwd/bin/vec_add.ll \
  -o /mnt/host_cwd/bin/vec_add.o

dpu-upmem-dpurte-clang -O2 -g0 -c /mnt/host_cwd/dpu/vec_add_args.c \
  -o /mnt/host_cwd/bin/vec_add_args.o

/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame \
  /mnt/host_cwd/bin/vec_add.o
/upmem-sdk-2023.1.0/bin/llvm-objcopy --remove-section=.eh_frame \
  /mnt/host_cwd/bin/vec_add_args.o

dpu-upmem-dpurte-clang \
  /mnt/host_cwd/bin/vec_add.o \
  /mnt/host_cwd/bin/vec_add_args.o \
  -o /mnt/host_cwd/bin/vec_add
```

### A3) Build and run host (container)
Purpose: move inputs to MRAM, launch the DPU, verify outputs.
```
cc -O2 -std=c11 -Wall -Wextra \
  -o /mnt/host_cwd/bin/vec_add_host \
  /mnt/host_cwd/host/vec_add_host.c \
  `dpu-pkg-config --cflags --libs dpu` \
  -DNR_DPUS=1

/mnt/host_cwd/bin/vec_add_host

# Optional: pass element count at runtime
/mnt/host_cwd/bin/vec_add_host 50
```

Note: this flow was validated up to 512 elements due to alignment/transfer constraints.

## Flow B: Triton -> DPU binary (recommended)
This uses Triton to generate LLVM IR, then runs on the simulator using the same
host/DPU wiring as above.

### B1) Generate Triton LLVM IR (host laptop)
Purpose: compile the Triton kernel to LLVM IR in `bin/triton_add.ll`.
```
./scripts/host_triton_ir.sh
```

The script sets:
- `PYTHONPATH=third_party/triton/python`
- `TRITON_BACKENDS_IN_TREE=1`
- `TRITON_DPU=1`
- `TRITON_DPU_OPT=third_party/upmem_llvm/llvm-project/build/bin/opt`
- `TRITON_CACHE_DIR` (writable cache)

Overrides (optional):
- `TRITON_PY` (python path)
- `TRITON_SRC` (triton repo path)
- `UPMEM_OPT` (opt path)
- `TRITON_CACHE_DIR`
- `OUT_LL` (output .ll path)
- `KERNEL_NAME` (default `add_kernel`)

### B2) Build and run in container
Purpose: compile the Triton IR and run it on the simulator.
```
cd /mnt/host_cwd
./scripts/container_triton_run.sh
```

Overrides (optional):
- `NR_DPUS` (default 1)

## Auto-generated wrapper and args
Triton emits only `add_kernel`. The DPU toolchain expects a `main` entrypoint
and uses `DPU_INPUT_ARGUMENTS` for parameters. The host script now generates:

- `dpu/triton_wrapper.ll` (DPU entrypoint)
- `dpu/triton_args.h` / `dpu/triton_args.c` (DPU args struct + definition)
- `host/triton_args.h` (host-side args struct)

Generation is done by parsing the kernel signature in `bin/triton_add.ll`:

- MRAM pointer args (`addrspace(255)`) are loaded from `DPU_INPUT_ARGUMENTS`
  and bitcast to the exact pointer type.
- Pointer args in `addrspace(1)` are passed as `null` (these are internal Triton
  args and should remain unused).
- Scalar args are loaded from `DPU_INPUT_ARGUMENTS` in order.

This keeps the wrapper and argument structs in sync with the Triton IR without
manual editing.

## Flow C: Triton artifact pack -> container run (no per-kernel C)
This flow makes Triton the only user-facing surface. The host packs a self-contained
artifact folder; the container compiles and runs it with a generic runner.

### C1) Pack Triton artifact (host laptop)
Purpose: generate `kernel.ll`, wrapper/args, and a host runner without manual C edits.
```
./scripts/host_triton_pack.py \
  --artifact-dir ./artifacts/add \
  --out-indices 2
```

Notes:
- `--out-indices` uses kernel arg indices (0-based).
- For the default `add_kernel` in `dpu_min_test.py`, the signature is:
  - args 0/1/2: MRAM pointers (A, B, C)
  - arg 3: scalar `n`
  - args 4/5: internal `addrspace(1)` pointers (ignored)
- You can supply your own Triton script with `--triton-script <file>`. The script
  must accept `--out <path>` and write the LLVM IR there.
- Default Triton script: `./dpu_min_test.py` if present, otherwise
  `$TRITON_SRC/python/triton/backends/dpu/dpu_min_test.py`.
- To set a default DPU count from the Triton script, you can use any of:
  - a kernel launch keyword: `kernel[grid](..., num_dpus=4)`
  - a `triton.Config(..., num_dpus=4)` inside `@triton.autotune`
  - a top-level literal: `TRITON_DPU_NUM_DPUS = 4`
  The generated host runner uses the first found in that order, unless
  `--nr-dpus` is passed at runtime.
  (Example: `dpu_min_test.py` sets `TRITON_DPU_NUM_DPUS = 4`.)
- Environment overrides: `TRITON_PY`, `TRITON_SRC`, `UPMEM_OPT`, `TRITON_CACHE_DIR`.

Artifact contents:
- `kernel.ll`: Triton LLVM IR
- `wrapper.ll`: DPU entrypoint wrapper
- `triton_args.h/.c`: DPU args struct + definition
- `host_args.h`: host args struct
- `host_runner.c`: auto-generated host runner
- `meta.json`: kernel signature + out indices

### C2) Build/run artifact (container)
Purpose: compile the DPU binary and run with the auto-generated host runner.
```
cd /mnt/host_cwd
ARTIFACT_DIR=/mnt/host_cwd/artifacts/add ./scripts/container_artifact_run.sh \
  --arg 3=10
```
`--arg 3=10` sets scalar kernel argument index 3 (the `N` length) to 10.

Optional overrides:
- `--nr-dpus N` to request multiple DPUs (inputs are sharded by default; this
  overrides any `num_dpus` default from the Triton script).
- `--len IDX=N` to set element counts per pointer arg (default uses the single
  scalar arg if present, e.g. arg 3 for `n`).
- `--in IDX=PATH` to provide raw input data for a pointer arg.
- `--out IDX=PATH` to write raw output data for a pointer arg.
Note: the auto-generated host runner supports multiple DPUs and shards inputs by default.

Example (explicit lengths + output capture):
```
ARTIFACT_DIR=/mnt/host_cwd/artifacts/add ./scripts/container_artifact_run.sh \
  --arg 3=10 --len 0=10 --len 1=10 --len 2=10 --out 2=/tmp/c.bin
```

## Legalization and pointer conversion (Triton flow)
The upmem `opt` is LLVM 12, which cannot parse opaque pointers. The DPU backend
now auto-converts `ptr` to typed pointers and retries `opt` when needed.

Useful env flags:
- Skip legalization: `TRITON_DPU_SKIP_LEGALIZE=1`
- Force legalization errors: `TRITON_DPU_FORCE_LEGALIZE=1`

When legalization runs:
```
[dpu_legalize] using opt: third_party/upmem_llvm/llvm-project/build/bin/opt
```

## Quick sanity checks
- Ensure full IR: `tail -n 5 bin/triton_add.ll`
- Ensure `.eh_frame` stripped: `llvm-readelf -l bin/vec_add` (no `.eh_frame` in MRAM)

## Troubleshooting
- `error: expected type` in `opt`: LLVM 12 rejected opaque pointers. Ensure you are using the in-tree Triton backend and let legalization run, or set `TRITON_DPU_SKIP_LEGALIZE=1`.
- `Cast between addresses of different address space is not supported`: you are compiling IR that still contains `addrspacecast`. Regenerate IR with the updated Triton DPU backend and re-run `host_triton_ir.sh`.
- `found end of file when expecting more instructions`: you redirected only the preview output. Use `dpu_min_test.py --out <file>` or `host_triton_ir.sh`.
- `DPU Error (invalid mram access)`: MRAM offsets/lengths must be 8-byte aligned. Keep the host-side align-to-8 logic and pass aligned sizes to transfers.
- `DPU Error (undefined symbol)` or `invalid memory symbol access`: ensure `triton_args.c` (Triton flow) or `vec_add_args.c` (DPU dialect flow) is linked into the DPU binary and `DPU_INPUT_ARGUMENTS` exists.
- `PermissionError` for `.triton/cache`: set `TRITON_CACHE_DIR` to a writable path.
