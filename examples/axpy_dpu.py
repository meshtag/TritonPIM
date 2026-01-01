import argparse
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import PIMTarget
from triton.compiler import ASTSource, make_backend

NUM_WARPS = 1
NUM_CTAS = 1
# host_triton_pack.py uses this to set a default DPU count in the host runner.
TRITON_DPU_NUM_DPUS = 2


@triton.jit
def axpy_kernel(X, Y, A, N, BLOCK: tl.constexpr):
    # One program handles one block of elements.
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask, other=0)
    y = tl.load(Y + offs, mask, other=0)
    out = A * x + y
    tl.store(Y + offs, out, mask)


# Create a binder so tooling can inspect the kernel signature without launching it.
axpy_kernel.create_binder()

# Signature for the DPU backend: pointers in MRAM + scalar args.
signature = {
    "X": "*i32",
    "Y": "*i32",
    "A": "i32",
    "N": "i32",
    "BLOCK": "constexpr",
}
constants = {"BLOCK": 16}
attrs = {}

# Build a compile-only DPU target (UPMEM). Args: backend="dpu", arch="upmem",
# warp_size=1 (UPMEM has no GPU-style warps; this keeps the backend simple).
src = ASTSource(axpy_kernel, signature, constexprs=constants, attrs=attrs)
target = PIMTarget("dpu", "upmem", 1)
backend = make_backend(target)
options = backend.parse_options({"num_warps": NUM_WARPS, "num_ctas": NUM_CTAS})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton DPU AXPY to LLVM IR")
    parser.add_argument("--out", type=str, default=None, help="Write full LLVM IR to this path")
    parser.add_argument("--preview", action="store_true", help="Print only the first 30 lines")
    parser.add_argument("--print", action="store_true", help="Print full LLVM IR to stdout")
    args = parser.parse_args()

    ccinfo = triton.compile(src, target=target, options=options.__dict__)
    ll = ccinfo.asm[backend.binary_ext]
    if isinstance(ll, (bytes, bytearray)):
        ll = ll.decode("utf-8")

    if args.out:
        Path(args.out).write_text(ll, encoding="utf-8")
        print(f"[axpy_dpu] wrote {len(ll)} bytes to {args.out}")

    if args.preview:
        print("\n".join(ll.splitlines()[:30]))
    elif args.print or not args.out:
        print(ll)


if __name__ == "__main__":
    main()
