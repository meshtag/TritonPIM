#ifndef TRITON_HOST_ARGS_H
#define TRITON_HOST_ARGS_H

#include <stdint.h>

typedef struct {
  uint32_t arg0_mram;
  uint32_t arg1_mram;
  uint32_t arg2_mram;
  uint32_t arg3;
} dpu_args_t;

#endif
