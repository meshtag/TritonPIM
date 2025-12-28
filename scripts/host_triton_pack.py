#!/usr/bin/env python3
import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = ROOT_DIR


def infer_elem_type(ptr_ty: str) -> str:
    ty = ptr_ty.strip()
    if ty.startswith("ptr"):
        return "i8"
    if ty.endswith("*"):
        ty = ty[:-1].strip()
    ty = ty.replace("addrspace(255)", "").replace("addrspace(1)", "").strip()
    return ty or "i8"


def elem_bytes(elem_type: str) -> int:
    elem_type = elem_type.strip()
    if elem_type in ("i1", "i8"):
        return 1
    if elem_type == "i16":
        return 2
    if elem_type == "i32":
        return 4
    if elem_type == "i64":
        return 8
    if elem_type in ("half", "f16"):
        return 2
    if elem_type in ("float", "f32"):
        return 4
    if elem_type in ("double", "f64"):
        return 8
    raise ValueError(f"unsupported element type: {elem_type}")


def scalar_c_type(ty: str) -> str:
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


def elem_c_type(elem_type: str) -> str:
    elem_type = elem_type.strip()
    if elem_type == "i1":
        return "uint8_t"
    if elem_type == "i8":
        return "uint8_t"
    if elem_type == "i16":
        return "uint16_t"
    if elem_type == "i32":
        return "int32_t"
    if elem_type == "i64":
        return "int64_t"
    if elem_type in ("half", "f16"):
        return "uint16_t"
    if elem_type in ("float", "f32"):
        return "float"
    if elem_type in ("double", "f64"):
        return "double"
    raise ValueError(f"unsupported element type: {elem_type}")


def parse_out_indices(text: str) -> list[int]:
    if not text:
        return []
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        out.append(int(item))
    return out


def parse_num_dpus_value(text: str, name: str = "TRITON_DPU_NUM_DPUS") -> int:
    clean = text.replace("_", "").strip()
    if not clean.isdigit():
        raise ValueError(f"{name} must be a positive integer")
    val = int(clean, 10)
    if val <= 0:
        raise ValueError(f"{name} must be > 0")
    if val > 0xFFFFFFFF:
        raise ValueError(f"{name} must fit in uint32_t")
    return val


def collect_literal_ints(tree: ast.AST) -> dict[str, int]:
    out: dict[str, int] = {}
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            targets = node.targets
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        if value is None:
            continue
        try:
            literal = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            continue
        if isinstance(literal, bool) or not isinstance(literal, int):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                out[target.id] = literal
    return out


def extract_int_literal(node: ast.AST, consts: dict[str, int], name: str) -> int:
    if isinstance(node, ast.Name) and node.id in consts:
        return consts[node.id]
    try:
        literal = ast.literal_eval(node)
    except (ValueError, SyntaxError):
        raise ValueError(f"{name} must be a literal integer") from None
    if isinstance(literal, bool) or not isinstance(literal, int):
        raise ValueError(f"{name} must be a literal integer")
    return literal


def is_config_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "Config":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "Config":
        return True
    return False


def is_kernel_launch_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Subscript):
        return True
    if isinstance(func, ast.Attribute) and func.attr in ("run", "warmup"):
        return True
    return False


def extract_num_dpus_kw(keywords: list[ast.keyword], consts: dict[str, int]) -> int | None:
    for kw in keywords:
        if kw.arg == "num_dpus":
            literal = extract_int_literal(kw.value, consts, "num_dpus")
            return parse_num_dpus_value(str(literal), name="num_dpus")
    return None


def select_unique(values: list[int], name: str) -> int | None:
    if not values:
        return None
    uniq = sorted(set(values))
    if len(uniq) > 1:
        raise ValueError(f"multiple {name} values found: {', '.join(str(v) for v in uniq)}")
    return uniq[0]


def parse_triton_num_dpus(script_path: Path) -> int | None:
    if not script_path.exists():
        return None
    try:
        text = script_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        tree = ast.parse(text, filename=str(script_path))
    except SyntaxError:
        for line in text.splitlines():
            match = re.match(r"^\s*TRITON_DPU_NUM_DPUS\s*=\s*([0-9_]+)\s*(?:#.*)?$", line)
            if match:
                return parse_num_dpus_value(match.group(1), name="TRITON_DPU_NUM_DPUS")
        return None

    consts = collect_literal_ints(tree)
    launch_vals: list[int] = []
    config_vals: list[int] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            if is_kernel_launch_call(node):
                val = extract_num_dpus_kw(node.keywords, consts)
                if val is not None:
                    launch_vals.append(val)
            elif is_config_call(node):
                val = extract_num_dpus_kw(node.keywords, consts)
                if val is not None:
                    config_vals.append(val)
            self.generic_visit(node)

    Visitor().visit(tree)

    picked = select_unique(launch_vals, "num_dpus launch")
    if picked is not None:
        return picked
    picked = select_unique(config_vals, "num_dpus config")
    if picked is not None:
        return picked

    if "TRITON_DPU_NUM_DPUS" in consts:
        return parse_num_dpus_value(str(consts["TRITON_DPU_NUM_DPUS"]), name="TRITON_DPU_NUM_DPUS")
    return None


def load_kernel_signature(ir_path: Path, kernel_name: str):
    sys.path.append(str(SCRIPT_DIR))
    from generate_triton_wrapper import parse_kernel_signature

    text = ir_path.read_text(encoding="utf-8")
    return parse_kernel_signature(text, kernel_name)


def generate_host_runner(meta: dict, out_path: Path) -> None:
    arg_count = meta["arg_count"]
    ptr_args = [a for a in meta["passed_args"] if a["kind"] == "ptr"]
    scalar_args = [a for a in meta["passed_args"] if a["kind"] == "scalar"]

    default_len_idx = -1
    if scalar_args:
        int_scalars = [a for a in scalar_args if a["c_type"] in ("uint32_t", "uint64_t")]
        if len(int_scalars) == 1 and ptr_args:
            default_len_idx = int_scalars[0]["arg_index"]

    lines = []
    lines.append("// Auto-generated Triton host runner. Do not edit by hand.")
    lines.append("#include <dpu.h>")
    lines.append("#include <errno.h>")
    lines.append("#include <stdint.h>")
    lines.append("#include <stdio.h>")
    lines.append("#include <stdlib.h>")
    lines.append("#include <string.h>")
    lines.append("")
    lines.append("#ifndef DPU_BINARY")
    lines.append("#define DPU_BINARY \"./dpu.elf\"")
    lines.append("#endif")
    lines.append("")
    lines.append("#ifndef NR_DPUS")
    lines.append("#define NR_DPUS 1")
    lines.append("#endif")
    lines.append("")
    lines.append('#include "host_args.h"')
    lines.append("")
    lines.append(f"#define ARG_COUNT {arg_count}")
    lines.append("#define ALIGN8(x) (((x) + 7u) & ~7u)")
    lines.append("")
    lines.append("static inline uint64_t chunk_len(uint64_t total, uint64_t per, uint32_t idx) {")
    lines.append("  if (per == 0) return 0;")
    lines.append("  uint64_t start = (uint64_t)idx * per;")
    lines.append("  if (start >= total) return 0;")
    lines.append("  uint64_t rem = total - start;")
    lines.append("  return rem < per ? rem : per;")
    lines.append("}")
    lines.append("")
    lines.append("static void usage(const char *prog) {")
    lines.append("  fprintf(stderr, \"Usage: %s [--nr-dpus N] [--arg IDX=VAL] [--len IDX=N] [--in IDX=PATH] [--out IDX=PATH]\\n\", prog);")
    lines.append("}")
    lines.append("")
    lines.append("static int parse_kv(const char *arg, uint32_t *idx, const char **val) {")
    lines.append("  const char *sep = strchr(arg, '=');")
    lines.append("  if (!sep) sep = strchr(arg, ':');")
    lines.append("  if (!sep) return 0;")
    lines.append("  char *end = NULL;")
    lines.append("  errno = 0;")
    lines.append("  unsigned long idx_ul = strtoul(arg, &end, 10);")
    lines.append("  if (errno || end != sep) return 0;")
    lines.append("  *idx = (uint32_t)idx_ul;")
    lines.append("  *val = sep + 1;")
    lines.append("  return 1;")
    lines.append("}")
    lines.append("")
    lines.append("static uint64_t parse_u64(const char *s, int *ok) {")
    lines.append("  char *end = NULL;")
    lines.append("  errno = 0;")
    lines.append("  unsigned long long v = strtoull(s, &end, 10);")
    lines.append("  if (errno || end == s || *end != '\\0') {")
    lines.append("    *ok = 0;")
    lines.append("    return 0;")
    lines.append("  }")
    lines.append("  *ok = 1;")
    lines.append("  return (uint64_t)v;")
    lines.append("}")
    lines.append("")
    lines.append("static int read_file(const char *path, void *buf, size_t bytes) {")
    lines.append("  FILE *fp = fopen(path, \"rb\");")
    lines.append("  if (!fp) return 0;")
    lines.append("  size_t got = fread(buf, 1, bytes, fp);")
    lines.append("  fclose(fp);")
    lines.append("  if (got < bytes) {")
    lines.append("    memset((uint8_t *)buf + got, 0, bytes - got);")
    lines.append("  }")
    lines.append("  return 1;")
    lines.append("}")
    lines.append("")
    lines.append("static int write_file(const char *path, const void *buf, size_t bytes) {")
    lines.append("  FILE *fp = fopen(path, \"wb\");")
    lines.append("  if (!fp) return 0;")
    lines.append("  size_t wrote = fwrite(buf, 1, bytes, fp);")
    lines.append("  fclose(fp);")
    lines.append("  return wrote == bytes;")
    lines.append("}")
    lines.append("")

    lines.append("int main(int argc, char **argv) {")
    lines.append("  uint64_t scalar_vals[ARG_COUNT] = {0};")
    lines.append("  uint8_t scalar_set[ARG_COUNT] = {0};")
    lines.append("  uint64_t len_elems[ARG_COUNT] = {0};")
    lines.append("  uint8_t len_set[ARG_COUNT] = {0};")
    lines.append("  const char *in_paths[ARG_COUNT] = {0};")
    lines.append("  const char *out_paths[ARG_COUNT] = {0};")
    default_num_dpus = meta.get("num_dpus")
    if default_num_dpus is None:
        lines.append("  uint32_t nr_requested = NR_DPUS;")
    else:
        lines.append(f"  uint32_t nr_requested = {int(default_num_dpus)}u;")
    lines.append("")
    lines.append("  for (int i = 1; i < argc; i++) {")
    lines.append("    if (strcmp(argv[i], \"--help\") == 0) {")
    lines.append("      usage(argv[0]);")
    lines.append("      return 0;")
    lines.append("    }")
    lines.append("    if (strcmp(argv[i], \"--nr-dpus\") == 0 && i + 1 < argc) {")
    lines.append("      int ok = 0;")
    lines.append("      uint64_t v = parse_u64(argv[++i], &ok);")
    lines.append("      if (!ok) { fprintf(stderr, \"invalid --nr-dpus\\n\"); return 1; }")
    lines.append("      nr_requested = (uint32_t)v;")
    lines.append("      continue;")
    lines.append("    }")
    lines.append("    if ((strcmp(argv[i], \"--arg\") == 0 || strncmp(argv[i], \"--arg=\", 6) == 0) && i + 1 < argc) {")
    lines.append("      const char *kv = argv[i][5] == '=' ? argv[i] + 6 : argv[++i];")
    lines.append("      uint32_t idx = 0; const char *val = NULL;")
    lines.append("      if (!parse_kv(kv, &idx, &val) || idx >= ARG_COUNT) {")
    lines.append("        fprintf(stderr, \"invalid --arg %s\\n\", kv); return 1; }")
    lines.append("      int ok = 0; uint64_t v = parse_u64(val, &ok);")
    lines.append("      if (!ok) { fprintf(stderr, \"invalid --arg value %s\\n\", val); return 1; }")
    lines.append("      scalar_vals[idx] = v; scalar_set[idx] = 1; continue;")
    lines.append("    }")
    lines.append("    if ((strcmp(argv[i], \"--len\") == 0 || strncmp(argv[i], \"--len=\", 6) == 0) && i + 1 < argc) {")
    lines.append("      const char *kv = argv[i][5] == '=' ? argv[i] + 6 : argv[++i];")
    lines.append("      uint32_t idx = 0; const char *val = NULL;")
    lines.append("      if (!parse_kv(kv, &idx, &val) || idx >= ARG_COUNT) {")
    lines.append("        fprintf(stderr, \"invalid --len %s\\n\", kv); return 1; }")
    lines.append("      int ok = 0; uint64_t v = parse_u64(val, &ok);")
    lines.append("      if (!ok) { fprintf(stderr, \"invalid --len value %s\\n\", val); return 1; }")
    lines.append("      len_elems[idx] = v; len_set[idx] = 1; continue;")
    lines.append("    }")
    lines.append("    if ((strcmp(argv[i], \"--in\") == 0 || strncmp(argv[i], \"--in=\", 5) == 0) && i + 1 < argc) {")
    lines.append("      const char *kv = argv[i][4] == '=' ? argv[i] + 5 : argv[++i];")
    lines.append("      uint32_t idx = 0; const char *val = NULL;")
    lines.append("      if (!parse_kv(kv, &idx, &val) || idx >= ARG_COUNT) {")
    lines.append("        fprintf(stderr, \"invalid --in %s\\n\", kv); return 1; }")
    lines.append("      in_paths[idx] = val; continue;")
    lines.append("    }")
    lines.append("    if ((strcmp(argv[i], \"--out\") == 0 || strncmp(argv[i], \"--out=\", 6) == 0) && i + 1 < argc) {")
    lines.append("      const char *kv = argv[i][5] == '=' ? argv[i] + 6 : argv[++i];")
    lines.append("      uint32_t idx = 0; const char *val = NULL;")
    lines.append("      if (!parse_kv(kv, &idx, &val) || idx >= ARG_COUNT) {")
    lines.append("        fprintf(stderr, \"invalid --out %s\\n\", kv); return 1; }")
    lines.append("      out_paths[idx] = val; continue;")
    lines.append("    }")
    lines.append("    fprintf(stderr, \"unknown arg: %s\\n\", argv[i]);")
    lines.append("    usage(argv[0]);")
    lines.append("    return 1;")
    lines.append("  }")

    if default_len_idx >= 0:
        lines.append(f"  const int default_len_idx = {default_len_idx};")
    else:
        lines.append("  const int default_len_idx = -1;")

    lines.append("")
    lines.append("  struct dpu_set_t dpu_set, dpu;")
    lines.append("  uint32_t nr_dpus = 0;")
    lines.append("  DPU_ASSERT(dpu_alloc(nr_requested, NULL, &dpu_set));")
    lines.append("  DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));")
    lines.append("  if (nr_dpus == 0) {")
    lines.append("    fprintf(stderr, \"No DPUs allocated\\n\");")
    lines.append("    return 1;")
    lines.append("  }")
    lines.append("  DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));")
    lines.append("")
    lines.append("  uint64_t heap_base64 = 0;")
    lines.append("  uint32_t idx = 0;")
    lines.append("  DPU_FOREACH(dpu_set, dpu, idx) {")
    lines.append("    DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, &heap_base64, sizeof(heap_base64)));")
    lines.append("    break;")
    lines.append("  }")
    lines.append("  uint32_t heap_base32 = (uint32_t)heap_base64;")
    lines.append("")
    lines.append("  dpu_args_t *args = calloc(nr_dpus, sizeof(dpu_args_t));")
    lines.append("  if (!args) {")
    lines.append("    fprintf(stderr, \"allocation failed for args\\n\");")
    lines.append("    return 1;")
    lines.append("  }")
    lines.append("")

    # Assign scalars
    for arg in scalar_args:
        arg_index = arg["arg_index"]
        field_index = arg["field_index"]
        c_type = arg["c_type"]
        lines.append(f"  if (!scalar_set[{arg_index}]) {{")
        lines.append(f"    fprintf(stderr, \"missing --arg {arg_index}=<value>\\n\");")
        lines.append("    return 1;")
        lines.append("  }")
        lines.append("")

    # Pointer allocations and transfers
    lines.append("  uint64_t cur_off = 0;")
    lines.append("  uint64_t len_default_total = 0;")
    lines.append("  uint64_t len_default_per_dpu = 0;")
    lines.append("  int has_default_len = 0;")

    for arg in ptr_args:
        arg_index = arg["arg_index"]
        field_index = arg["field_index"]
        elem_type = arg["elem_type"]
        elem_c = arg["elem_c_type"]
        elem_sz = arg["elem_bytes"]
        is_output = arg["is_output"]
        base_val = arg.get("base_val", 1)
        lines.append(f"  uint64_t len_total_{arg_index} = 0;")
        lines.append(f"  if (len_set[{arg_index}]) {{")
        lines.append(f"    len_total_{arg_index} = len_elems[{arg_index}];")
        lines.append("  } else if (default_len_idx >= 0 && scalar_set[default_len_idx]) {")
        lines.append(f"    len_total_{arg_index} = scalar_vals[default_len_idx];")
        lines.append("  } else {")
        lines.append(f"    fprintf(stderr, \"missing --len {arg_index}=<n> and no default length available\\n\");")
        lines.append("    return 1;")
        lines.append("  }")
        lines.append(f"  if (len_total_{arg_index} == 0) {{")
        lines.append(f"    fprintf(stderr, \"length for arg {arg_index} must be > 0\\n\");")
        lines.append("    return 1;")
        lines.append("  }")
        lines.append(f"  uint64_t len_per_dpu_{arg_index} = (len_total_{arg_index} + nr_dpus - 1) / nr_dpus;")
        lines.append(f"  uint64_t bytes_per_dpu_{arg_index}_raw = len_per_dpu_{arg_index} * {elem_sz}u;")
        lines.append(f"  uint64_t bytes_per_dpu_{arg_index} = ALIGN8(bytes_per_dpu_{arg_index}_raw);")
        lines.append(f"  uint64_t stride_elems_{arg_index} = bytes_per_dpu_{arg_index} / {elem_sz}u;")
        lines.append(f"  uint32_t off_{arg_index} = (uint32_t)cur_off;")
        lines.append(f"  cur_off = (uint64_t)off_{arg_index} + bytes_per_dpu_{arg_index};")
        lines.append("")
        lines.append(f"  uint8_t *buf_{arg_index} = calloc((size_t)nr_dpus * stride_elems_{arg_index}, {elem_sz}u);")
        lines.append(f"  if (!buf_{arg_index}) {{ fprintf(stderr, \"alloc failed for arg {arg_index}\\n\"); return 1; }}")
        lines.append(f"  for (uint32_t d = 0; d < nr_dpus; d++) {{")
        lines.append(f"    uint64_t len_d = chunk_len(len_total_{arg_index}, len_per_dpu_{arg_index}, d);")
        lines.append(f"    uint8_t *dst = buf_{arg_index} + (uint64_t)d * stride_elems_{arg_index} * {elem_sz}u;")
        lines.append(f"    if (in_paths[{arg_index}]) {{")
        lines.append(f"      if (!read_file(in_paths[{arg_index}], dst, len_d * {elem_sz}u)) {{")
        lines.append(f"        fprintf(stderr, \"failed to read input for arg {arg_index}\\n\"); return 1; }}")
        lines.append("    } else {")
        if not is_output:
            if elem_type in ("i8", "i16", "i32", "i64", "f32", "f64"):
                lines.append(f"      {elem_c} *typed = ({elem_c} *)dst;")
                lines.append(f"      for (uint64_t i = 0; i < len_d; i++) {{")
                if elem_type in ("f32", "f64"):
                    lines.append(f"        typed[i] = ({elem_c})({base_val} + (double)(i + (uint64_t)d * len_per_dpu_{arg_index}));")
                else:
                    lines.append(f"        typed[i] = ({elem_c})({base_val} + (uint64_t)(i + (uint64_t)d * len_per_dpu_{arg_index}));")
                lines.append("      }")
        lines.append("    }")
        lines.append("  }")
        lines.append(f"  for (uint32_t d = 0; d < nr_dpus; d++) {{")
        lines.append(f"    args[d].arg{field_index}_mram = heap_base32 + off_{arg_index};")
        lines.append("  }")
        lines.append(f"  if (!has_default_len && default_len_idx >= 0) {{")
        lines.append(f"    len_default_total = len_total_{arg_index};")
        lines.append(f"    len_default_per_dpu = len_per_dpu_{arg_index};")
        lines.append("    has_default_len = 1;")
        lines.append("  }")
        lines.append("")

    # Populate scalar args per DPU
    for arg in scalar_args:
        arg_index = arg["arg_index"]
        field_index = arg["field_index"]
        c_type = arg["c_type"]
        lines.append(f"  for (uint32_t d = 0; d < nr_dpus; d++) {{")
        if default_len_idx >= 0 and arg_index == default_len_idx:
            lines.append("    if (has_default_len) {")
            lines.append("      uint64_t len_d = chunk_len(len_default_total, len_default_per_dpu, d);")
            lines.append(f"      args[d].arg{field_index} = ({c_type})len_d;")
            lines.append("    } else {")
            lines.append(f"      args[d].arg{field_index} = ({c_type})scalar_vals[{arg_index}];")
            lines.append("    }")
        else:
            lines.append(f"    args[d].arg{field_index} = ({c_type})scalar_vals[{arg_index}];")
        lines.append("  }")
        lines.append("")

    # Push args
    lines.append("  idx = 0;")
    lines.append("  DPU_FOREACH(dpu_set, dpu, idx) {")
    lines.append("    DPU_ASSERT(dpu_prepare_xfer(dpu, &args[idx]));")
    lines.append("  }")
    lines.append("  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, \"DPU_INPUT_ARGUMENTS\", 0, sizeof(dpu_args_t), DPU_XFER_DEFAULT));")
    lines.append("")

    # Copy inputs to MRAM
    for arg in ptr_args:
        arg_index = arg["arg_index"]
        lines.append("  idx = 0;")
        lines.append("  DPU_FOREACH(dpu_set, dpu, idx) {")
        lines.append(f"    uint8_t *src = buf_{arg_index} + (uint64_t)idx * stride_elems_{arg_index} * {arg['elem_bytes']}u;")
        lines.append("    DPU_ASSERT(dpu_prepare_xfer(dpu, src));")
        lines.append("  }")
        lines.append(f"  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, off_{arg_index}, bytes_per_dpu_{arg_index}, DPU_XFER_DEFAULT));")
        lines.append("")

    # Launch
    lines.append("  DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));")
    lines.append("")

    # Fetch outputs
    for arg in ptr_args:
        if not arg["is_output"]:
            continue
        arg_index = arg["arg_index"]
        elem_c = arg["elem_c_type"]
        lines.append("  idx = 0;")
        lines.append("  DPU_FOREACH(dpu_set, dpu, idx) {")
        lines.append(f"    uint8_t *dst = buf_{arg_index} + (uint64_t)idx * stride_elems_{arg_index} * {arg['elem_bytes']}u;")
        lines.append("    DPU_ASSERT(dpu_prepare_xfer(dpu, dst));")
        lines.append("  }")
        lines.append(f"  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, off_{arg_index}, bytes_per_dpu_{arg_index}, DPU_XFER_DEFAULT));")
        lines.append(f"  if (out_paths[{arg_index}]) {{")
        lines.append(f"    uint64_t total_bytes = len_total_{arg_index} * {arg['elem_bytes']}u;")
        lines.append("    uint8_t *out_full = malloc(total_bytes);")
        lines.append("    if (!out_full) { fprintf(stderr, \"alloc failed for output\\n\"); return 1; }")
        lines.append(f"    for (uint32_t d = 0; d < nr_dpus; d++) {{")
        lines.append(f"      uint64_t len_d = chunk_len(len_total_{arg_index}, len_per_dpu_{arg_index}, d);")
        lines.append(f"      uint8_t *src = buf_{arg_index} + (uint64_t)d * stride_elems_{arg_index} * {arg['elem_bytes']}u;")
        lines.append(f"      uint8_t *dst = out_full + (uint64_t)d * len_per_dpu_{arg_index} * {arg['elem_bytes']}u;")
        lines.append(f"      memcpy(dst, src, len_d * {arg['elem_bytes']}u);")
        lines.append("    }")
        lines.append(f"    if (!write_file(out_paths[{arg_index}], out_full, total_bytes)) {{")
        lines.append(f"      fprintf(stderr, \"failed to write output for arg {arg_index}\\n\"); return 1; }}")
        lines.append("    free(out_full);")
        lines.append("  } else {")
        lines.append(f"    for (uint32_t d = 0; d < nr_dpus; d++) {{")
        lines.append(f"      uint64_t len_d = chunk_len(len_total_{arg_index}, len_per_dpu_{arg_index}, d);")
        lines.append(f"      uint64_t to_print = len_d < 8 ? len_d : 8;")
        lines.append(f"      {elem_c} *typed_out = ({elem_c} *)(buf_{arg_index} + (uint64_t)d * stride_elems_{arg_index} * {arg['elem_bytes']}u);")
        lines.append(f"      printf(\"output arg {arg_index} dpu %u:\", d);")
        lines.append(f"      for (uint64_t i = 0; i < to_print; i++) {{")
        if elem_c in ("float", "double"):
            lines.append("        printf(\" %f\", (double)typed_out[i]);")
        else:
            lines.append("        printf(\" %lld\", (long long)typed_out[i]);")
        lines.append("      }")
        lines.append("      printf(\"\\n\");")
        lines.append("    }")
        lines.append("  }")
        lines.append("")

    # Cleanup
    for arg in ptr_args:
        arg_index = arg["arg_index"]
        lines.append(f"  free(buf_{arg_index});")

    lines.append("  free(args);")
    lines.append("  DPU_ASSERT(dpu_free(dpu_set));")
    lines.append("  return 0;")
    lines.append("}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack Triton kernel into a runnable artifact folder")
    parser.add_argument("--artifact-dir", required=True, help="Output artifact directory")
    parser.add_argument("--kernel", default="add_kernel", help="Kernel function name")
    parser.add_argument("--out-indices", required=True, help="Comma-separated kernel arg indices that are outputs")
    parser.add_argument("--triton-script", default=None, help="Python file that emits LLVM IR via --out")
    parser.add_argument("--python", dest="python_bin", default=os.environ.get("TRITON_PY", sys.executable))
    parser.add_argument("--triton-src", default=os.environ.get("TRITON_SRC", str(PROJECT_ROOT / "third_party" / "triton")))
    parser.add_argument("--upmem-opt", default=os.environ.get("UPMEM_OPT", os.environ.get("TRITON_DPU_OPT", str(PROJECT_ROOT / "third_party" / "upmem_llvm" / "llvm-project" / "build" / "bin" / "opt"))))
    parser.add_argument("--cache-dir", default=os.environ.get("TRITON_CACHE_DIR", str(ROOT_DIR / ".triton_cache")))
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.triton_script:
        triton_script = Path(args.triton_script)
    else:
        triton_script = ROOT_DIR / "dpu_min_test.py"
        if not triton_script.exists():
            triton_script = Path(args.triton_src) / "python/triton/backends/dpu/dpu_min_test.py"
    num_dpus = parse_triton_num_dpus(triton_script)
    kernel_ll = artifact_dir / "kernel.ll"
    wrapper_ll = artifact_dir / "wrapper.ll"
    dpu_args_h = artifact_dir / "triton_args.h"
    dpu_args_c = artifact_dir / "triton_args.c"
    host_args_h = artifact_dir / "host_args.h"
    meta_path = artifact_dir / "meta.json"
    host_runner = artifact_dir / "host_runner.c"

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(args.triton_src) / "python")
    env["TRITON_BACKENDS_IN_TREE"] = "1"
    env["TRITON_DPU"] = "1"
    env["TRITON_DPU_OPT"] = args.upmem_opt
    env["TRITON_CACHE_DIR"] = str(cache_dir)
    env.pop("TRITON_DPU_SKIP_LEGALIZE", None)
    env["TRITON_DPU_FORCE_LEGALIZE"] = "1"

    cmd = [args.python_bin, str(triton_script), "--out", str(kernel_ll)]
    subprocess.run(cmd, check=True, env=env)

    wrapper_cmd = [
        args.python_bin,
        str(SCRIPT_DIR / "generate_triton_wrapper.py"),
        "--ir",
        str(kernel_ll),
        "--out",
        str(wrapper_ll),
        "--kernel",
        args.kernel,
        "--dpu-args-h",
        str(dpu_args_h),
        "--dpu-args-c",
        str(dpu_args_c),
        "--host-args-h",
        str(host_args_h),
    ]
    subprocess.run(wrapper_cmd, check=True)

    kernel_name, sig = load_kernel_signature(kernel_ll, args.kernel)
    out_indices = parse_out_indices(args.out_indices)

    kernel_args = []
    passed_args = []
    field_index = 0
    for arg_index, (ty, _name, is_ptr, addrspace) in enumerate(sig):
        arg_info = {
            "arg_index": arg_index,
            "llvm_type": ty,
            "is_ptr": is_ptr,
            "addrspace": addrspace,
        }
        if is_ptr:
            elem_type = infer_elem_type(ty)
            arg_info["elem_type"] = elem_type
            arg_info["elem_bytes"] = elem_bytes(elem_type)
        else:
            arg_info["c_type"] = scalar_c_type(ty)
        passed = not (is_ptr and addrspace == 1)
        arg_info["passed"] = passed
        kernel_args.append(arg_info)

        if not passed:
            continue
        if is_ptr:
            if addrspace != 255:
                raise ValueError(f"unsupported pointer address space in arg {arg_index}: {ty}")
            elem_type = arg_info["elem_type"]
            passed_args.append(
                {
                    "arg_index": arg_index,
                    "field_index": field_index,
                    "kind": "ptr",
                    "elem_type": elem_type,
                    "elem_bytes": arg_info["elem_bytes"],
                    "elem_c_type": elem_c_type(elem_type),
                    "is_output": arg_index in out_indices,
                }
            )
        else:
            passed_args.append(
                {
                    "arg_index": arg_index,
                    "field_index": field_index,
                    "kind": "scalar",
                    "c_type": arg_info["c_type"],
                }
            )
        field_index += 1

    for out_idx in out_indices:
        match = next((a for a in passed_args if a["arg_index"] == out_idx), None)
        if not match or match["kind"] != "ptr":
            raise ValueError(f"out_indices contains non-pointer arg index: {out_idx}")

    meta = {
        "kernel": kernel_name,
        "arg_count": len(sig),
        "args": kernel_args,
        "passed_args": passed_args,
        "out_indices": out_indices,
    }
    if num_dpus is not None:
        meta["num_dpus"] = num_dpus
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    input_ptrs = [a for a in passed_args if a["kind"] == "ptr" and not a["is_output"]]
    for idx, arg in enumerate(input_ptrs):
        arg["base_val"] = 1 if idx == 0 else 100 + (idx - 1) * 100

    generate_host_runner(meta, host_runner)

    print(f"Wrote artifact to {artifact_dir}")


if __name__ == "__main__":
    main()
