// Defines DPU_INPUT_ARGUMENTS in the host-accessible section.
#include <defs.h>
#include <stdint.h>

typedef struct {
  __mram_ptr uint8_t *a;
  __mram_ptr uint8_t *b;
  __mram_ptr uint8_t *c;
  uint32_t n;
} dpu_args_t;

// Force host section size to a multiple of 8 bytes.
__host dpu_args_t DPU_INPUT_ARGUMENTS __attribute__((used, aligned(8)));
__host uint64_t DPU_INPUT_ARGUMENTS_PAD __attribute__((used));
