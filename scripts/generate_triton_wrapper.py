#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


DEFINE_RE = re.compile(r"^define\s+(\S+)\s+@([^(]+)\((.*)\)\s*\{")
ARG_RE = re.compile(r"(.+?)\s+(%[-$._A-Za-z0-9]+)$")
ADDRSPACE_RE = re.compile(r"addrspace\((\d+)\)")


def split_args(arg_str: str) -> list[str]:
    args = []
    cur = []
    depth = 0
    for ch in arg_str:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "," and depth == 0:
            args.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        args.append(tail)
    return args


def parse_kernel_signature(text: str, kernel_name: str | None):
    for line in text.splitlines():
        match = DEFINE_RE.match(line.strip())
        if not match:
            continue
        ret_type, func_name, arg_str = match.groups()
        if kernel_name and func_name != kernel_name:
            continue
        if ret_type != "void":
            raise ValueError(f"unsupported return type: {ret_type}")
        args = []
        for raw_arg in split_args(arg_str):
            if not raw_arg:
                continue
            m = ARG_RE.match(raw_arg)
            if not m:
                raise ValueError(f"cannot parse argument: {raw_arg}")
            ty, arg_name = m.groups()
            ty = ty.strip()
            is_ptr = ty.endswith("*")
            addrspace = None
            if is_ptr:
                as_match = ADDRSPACE_RE.search(ty)
                if as_match:
                    addrspace = int(as_match.group(1))
            args.append((ty, arg_name, is_ptr, addrspace))
        return func_name, args
    raise ValueError("kernel signature not found")


def llvm_scalar_to_c(ty: str, for_dpu: bool) -> str:
    ty = ty.strip()
    if ty == "i1":
        return "uint8_t"
    if ty == "i8":
        return "uint8_t"
    if ty == "i16":
        return "uint16_t"
    if ty == "i32":
        return "uint32_t"
    if ty == "i64":
        return "uint64_t"
    if ty in ("half", "f16"):
        return "uint16_t"
    if ty in ("float", "f32"):
        return "float"
    if ty in ("double", "f64"):
        return "double"
    raise ValueError(f"unsupported scalar type: {ty}")


def generate_wrapper(kernel_name: str, args: list[tuple[str, str, bool, int | None]]):
    struct_fields = []
    call_args = []
    body_lines = []
    field_idx = 0

    body_lines.append("  %args = load %dpu_args, %dpu_args* @DPU_INPUT_ARGUMENTS, align 8")

    for idx, (ty, _name, is_ptr, addrspace) in enumerate(args):
        if is_ptr:
            if addrspace == 255:
                field_name = f"%arg{idx}"
                body_lines.append(
                    f"  {field_name} = extractvalue %dpu_args %args, {field_idx}"
                )
                field_idx += 1
                struct_fields.append("i8 addrspace(255)*")
                if ty == "i8 addrspace(255)*":
                    call_args.append((ty, field_name))
                else:
                    cast_name = f"%arg{idx}_cast"
                    body_lines.append(
                        f"  {cast_name} = bitcast i8 addrspace(255)* {field_name} to {ty}"
                    )
                    call_args.append((ty, cast_name))
            elif addrspace == 1:
                call_args.append((ty, "null"))
            else:
                raise ValueError(f"unsupported pointer address space in arg {idx}: {ty}")
        else:
            field_name = f"%arg{idx}"
            body_lines.append(
                f"  {field_name} = extractvalue %dpu_args %args, {field_idx}"
            )
            field_idx += 1
            struct_fields.append(ty)
            call_args.append((ty, field_name))

    if not struct_fields:
        struct_fields = ["i8"]  # dummy field to avoid empty struct

    struct_ty = ", ".join(struct_fields)
    call_args_str = ", ".join(f"{ty} {val}" for ty, val in call_args)

    wrapper = [
        "; Auto-generated Triton wrapper. Do not edit by hand.",
        'source_filename = "triton_wrapper"',
        "",
        f"%dpu_args = type {{ {struct_ty} }}",
        "",
        "@DPU_INPUT_ARGUMENTS = external global %dpu_args",
        "",
        f"declare void @{kernel_name}({', '.join([a[0] for a in args])})",
        "",
        "define i32 @main() {",
        "entry:",
    ]
    wrapper.extend(body_lines)
    wrapper.append(f"  call void @{kernel_name}({call_args_str})")
    wrapper.append("  ret i32 0")
    wrapper.append("}")
    wrapper.append("")

    return "\n".join(wrapper)


def generate_args_headers(
    args: list[tuple[str, str, bool, int | None]],
    dpu_header: Path | None,
    dpu_source: Path | None,
    host_header: Path | None,
) -> None:
    passed_args = []
    for idx, (ty, _name, is_ptr, addrspace) in enumerate(args):
        if is_ptr and addrspace == 1:
            continue
        passed_args.append((idx, ty, is_ptr, addrspace))

    dpu_fields = []
    host_fields = []
    field_idx = 0
    for _idx, ty, is_ptr, _addrspace in passed_args:
        if is_ptr:
            dpu_fields.append(f"  __mram_ptr uint8_t *arg{field_idx};")
            host_fields.append(f"  uint32_t arg{field_idx}_mram;")
        else:
            c_ty = llvm_scalar_to_c(ty, for_dpu=True)
            dpu_fields.append(f"  {c_ty} arg{field_idx};")
            host_fields.append(f"  {c_ty} arg{field_idx};")
        field_idx += 1

    dpu_struct = "\n".join(dpu_fields) if dpu_fields else "  uint8_t arg0;"
    host_struct = "\n".join(host_fields) if host_fields else "  uint8_t arg0;"

    if dpu_header:
        dpu_header.write_text(
            "\n".join(
                [
                    "#ifndef TRITON_DPU_ARGS_H",
                    "#define TRITON_DPU_ARGS_H",
                    "",
                    "#include <defs.h>",
                    "#include <stdint.h>",
                    "",
                    "typedef struct {",
                    dpu_struct,
                    "} dpu_args_t;",
                    "",
                    "#endif",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    if dpu_source:
        dpu_source.write_text(
            "\n".join(
                [
                    "#include \"triton_args.h\"",
                    "",
                    "__host dpu_args_t DPU_INPUT_ARGUMENTS __attribute__((used, aligned(8)));",
                    "__host uint64_t DPU_INPUT_ARGUMENTS_PAD __attribute__((used));",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    if host_header:
        host_header.write_text(
            "\n".join(
                [
                    "#ifndef TRITON_HOST_ARGS_H",
                    "#define TRITON_HOST_ARGS_H",
                    "",
                    "#include <stdint.h>",
                    "",
                    "typedef struct {",
                    host_struct,
                    "} dpu_args_t;",
                    "",
                    "#endif",
                    "",
                ]
            ),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DPU wrapper for Triton LLVM IR")
    parser.add_argument("--ir", required=True, help="Input LLVM IR (.ll)")
    parser.add_argument("--out", required=True, help="Output wrapper .ll path")
    parser.add_argument("--kernel", default="add_kernel", help="Kernel function name")
    parser.add_argument("--dpu-args-h", default=None, help="Output DPU args header (.h)")
    parser.add_argument("--dpu-args-c", default=None, help="Output DPU args definition (.c)")
    parser.add_argument("--host-args-h", default=None, help="Output host args header (.h)")
    args = parser.parse_args()

    text = Path(args.ir).read_text(encoding="utf-8")
    kernel, sig = parse_kernel_signature(text, args.kernel)
    wrapper = generate_wrapper(kernel, sig)
    Path(args.out).write_text(wrapper, encoding="utf-8")
    generate_args_headers(
        sig,
        Path(args.dpu_args_h) if args.dpu_args_h else None,
        Path(args.dpu_args_c) if args.dpu_args_c else None,
        Path(args.host_args_h) if args.host_args_h else None,
    )
    print(f"[wrapper] wrote {args.out}")


if __name__ == "__main__":
    main()
