#ifndef TRITON_DPU_ARGS_H
#define TRITON_DPU_ARGS_H

#include <defs.h>
#include <stdint.h>

typedef struct {
  __mram_ptr uint8_t *arg0;
  __mram_ptr uint8_t *arg1;
  __mram_ptr uint8_t *arg2;
  uint32_t arg3;
} dpu_args_t;

#endif
