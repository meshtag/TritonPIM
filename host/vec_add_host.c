// Host-side runner for DPU vec_add LLVM IR output.
#include <dpu.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DPU_BINARY
#define DPU_BINARY "./bin/vec_add"
#endif

#ifndef NR_DPUS
#define NR_DPUS 1
#endif

#define N_ELEMS 10

typedef struct {
  uint32_t a_mram;
  uint32_t b_mram;
  uint32_t c_mram;
  uint32_t n;
} dpu_args_t;

static void fill_inputs(int32_t *a, int32_t *b, int32_t *ref, uint32_t n,
                        uint32_t stride, uint32_t nr_dpus) {
  for (uint32_t d = 0; d < nr_dpus; d++) {
    for (uint32_t i = 0; i < n; i++) {
      int32_t aval = (int32_t)(i + 1);
      int32_t bval = (int32_t)(100 + i);
      a[d * stride + i] = aval;
      b[d * stride + i] = bval;
      ref[d * stride + i] = aval + bval;
    }
  }
}

int main(int argc, char **argv) {
  struct dpu_set_t dpu_set, dpu;
  uint32_t nr_dpus = 0;

  DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
  DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_dpus));
  struct dpu_program_t *program = NULL;
  DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, &program));

  uint32_t n = N_ELEMS;
  if (argc > 1) {
    n = (uint32_t)strtoul(argv[1], NULL, 10);
  }
  if (n == 0) {
    fprintf(stderr, "n must be > 0\n");
    return 1;
  }

  const uint32_t bytes_raw = n * sizeof(int32_t);
  const uint32_t bytes_aligned = (bytes_raw + 7u) & ~7u;
  const uint32_t stride = bytes_aligned / sizeof(int32_t);
  const uint32_t a_off = 0;
  const uint32_t b_off = a_off + bytes_aligned;
  const uint32_t c_off = b_off + bytes_aligned;

  int32_t *a = calloc(nr_dpus * stride, sizeof(int32_t));
  int32_t *b = calloc(nr_dpus * stride, sizeof(int32_t));
  int32_t *c = calloc(nr_dpus * stride, sizeof(int32_t));
  int32_t *ref = calloc(nr_dpus * stride, sizeof(int32_t));
  dpu_args_t *args = calloc(nr_dpus, sizeof(dpu_args_t));
  if (!a || !b || !c || !ref || !args) {
    fprintf(stderr, "allocation failed\n");
    return 1;
  }

  fill_inputs(a, b, ref, n, stride, nr_dpus);

  uint64_t heap_base64 = 0;
  uint32_t heap_base32 = 0;
  uint32_t idx = 0;
  DPU_FOREACH(dpu_set, dpu, idx) {
    DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, &heap_base64,
                             sizeof(heap_base64)));
    heap_base32 = (uint32_t)heap_base64;
    break;
  }

  for (uint32_t i = 0; i < nr_dpus; i++) {
    // Use absolute MRAM addresses (heap base + offset).
    args[i].a_mram = heap_base32 + a_off;
    args[i].b_mram = heap_base32 + b_off;
    args[i].c_mram = heap_base32 + c_off;
    args[i].n = n;
  }

  idx = 0;
  DPU_FOREACH(dpu_set, dpu, idx) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, &args[idx]));
  }
  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                           sizeof(dpu_args_t), DPU_XFER_DEFAULT));

  if (getenv("DPU_DEBUG")) {
    dpu_args_t args_rb = {0};
    idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
      DPU_ASSERT(dpu_copy_from(dpu, "DPU_INPUT_ARGUMENTS", 0, &args_rb,
                               sizeof(args_rb)));
      break;
    }
    printf("DBG heap_base=0x%08x args: a=0x%08x b=0x%08x c=0x%08x n=%u\n",
           heap_base32, args_rb.a_mram, args_rb.b_mram, args_rb.c_mram,
           args_rb.n);

    int32_t x_dbg[4] = {0};
    int32_t y_dbg[4] = {0};
    idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
      DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, a_off, x_dbg,
                               sizeof(x_dbg)));
      DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, b_off, y_dbg,
                               sizeof(y_dbg)));
      break;
    }
    printf("DBG MRAM X[0..2]=%d %d %d Y[0..2]=%d %d %d\n", x_dbg[0], x_dbg[1],
           x_dbg[2], y_dbg[0], y_dbg[1], y_dbg[2]);
  }

  idx = 0;
  DPU_FOREACH(dpu_set, dpu, idx) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, a + idx * stride));
  }
  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                           DPU_MRAM_HEAP_POINTER_NAME, a_off, bytes_aligned,
                           DPU_XFER_DEFAULT));

  idx = 0;
  DPU_FOREACH(dpu_set, dpu, idx) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, b + idx * stride));
  }
  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                           DPU_MRAM_HEAP_POINTER_NAME, b_off, bytes_aligned,
                           DPU_XFER_DEFAULT));

  DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));

  idx = 0;
  DPU_FOREACH(dpu_set, dpu, idx) {
    DPU_ASSERT(dpu_prepare_xfer(dpu, c + idx * stride));
  }
  DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                           DPU_MRAM_HEAP_POINTER_NAME, c_off, bytes_aligned,
                           DPU_XFER_DEFAULT));

  int ok = 1;
  for (uint32_t d = 0; d < nr_dpus; d++) {
    for (uint32_t i = 0; i < n; i++) {
      uint32_t idx_flat = d * stride + i;
      if (c[idx_flat] != ref[idx_flat]) {
        fprintf(stderr, "mismatch dpu %u idx %u: got %d expected %d\n", d, i,
                c[idx_flat], ref[idx_flat]);
        ok = 0;
        break;
      }
    }
  }

  if (ok) {
    printf("OK: outputs match for %u DPU(s)\n", nr_dpus);
  }

  free(a);
  free(b);
  free(c);
  free(ref);
  free(args);
  DPU_ASSERT(dpu_free(dpu_set));
  return ok ? 0 : 1;
}
