# dpu_min_test.py
import argparse
from pathlib import Path

import triton
import triton.language as tl
from triton.backends.compiler import PIMTarget
from triton.compiler import ASTSource, make_backend

NUM_WARPS = 1
NUM_CTAS = 1
TRITON_DPU_NUM_DPUS = 2

@triton.jit
def add_kernel(A, B, C, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A + offs, mask, other=0)
    b = tl.load(B + offs, mask, other=0)
    tl.store(C + offs, a + b, mask)

add_kernel.create_binder()

signature = {"A": "*i32", "B": "*i32", "C": "*i32", "N": "i32", "BLOCK": "constexpr"}
constants = {"BLOCK": 16}
attrs = {}

src = ASTSource(add_kernel, signature, constexprs=constants, attrs=attrs)
target = PIMTarget("dpu", "upmem", 1)
backend = make_backend(target)
options = backend.parse_options({"num_warps": NUM_WARPS, "num_ctas": NUM_CTAS})

def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Triton DPU add kernel to LLVM IR")
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
        print(f"[dpu_min_test] wrote {len(ll)} bytes to {args.out}")

    if args.preview:
        print("dpu ll len", len(ll))
        print("\n".join(ll.splitlines()[:30]))
    elif args.print or not args.out:
        print(ll)


if __name__ == "__main__":
    main()
