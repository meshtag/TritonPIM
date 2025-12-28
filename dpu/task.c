/*
 * pointwise with multiple tasklets
 *
 */
#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <stdint.h>
#include <stdio.h>

#include "../support/common.h"
#include "../support/cyclecount.h"

// Input and output arguments.
__host dpu_arguments_t DPU_INPUT_ARGUMENTS;
__host dpu_results_t DPU_RESULTS[NR_TASKLETS];

// Barrier initialization.
BARRIER_INIT(my_barrier, NR_TASKLETS);

extern int main_kernel1(void);
int (*kernels[nr_kernels])(void) = {main_kernel1};
int main(void) {
  // Kernel.
  return kernels[DPU_INPUT_ARGUMENTS.kernel]();
}

// pointwise: Computes pointwise operation for a cached block.
static void pointwise(T *bufferY, T *bufferX, unsigned int arrSize,
                      enum binary_operation op, unsigned int *alpha) {
  // Perform specific pointwise operation based on the supplied value.
  switch (op) {
  case BINARY_OP_ADD:
    for (unsigned int i = 0; i < arrSize; ++i) {
      bufferY[i] = bufferX[i] + bufferY[i];
    }
    break;
  case BINARY_OP_SUB:
    for (unsigned int i = 0; i < arrSize; ++i) {
      bufferY[i] = bufferX[i] - bufferY[i];
    }
    break;
  case BINARY_OP_MUL:
    for (unsigned int i = 0; i < arrSize; ++i) {
      bufferY[i] = bufferX[i] * bufferY[i];
    }
    break;
  case BINARY_OP_DIV:
    for (unsigned int i = 0; i < arrSize; ++i) {
      bufferY[i] = bufferX[i] / bufferY[i];
    }
    break;
  case BINARY_OP_AXPY:
    // AXPY kernel handled separately.
    for (unsigned int i = 0; i < arrSize; ++i) {
      bufferY[i] = (alpha == NULL) ? bufferX[i] : (*alpha) * bufferX[i];
    }
    break;
  default:
    for (unsigned int i = 0; i < arrSize; ++i) {
      bufferY[i] = bufferX[i] + bufferY[i];
    }
    break;
  }
}

// main_kernel1
int main_kernel1() {
  // Get the current tasklet id, this will be used in mapping it to the
  // computation chunk it is supposed to process.
  unsigned int tasklet_id = me();
#if PRINT
  printf("tasklet_id = %u\n", tasklet_id);
#endif
  // Barrier invocation.
  barrier_wait(&my_barrier);
#if defined(CYCLES) || defined(INSTRUCTIONS)
  perfcounter_count count;
  dpu_results_t *result = &DPU_RESULTS[tasklet_id];
  result->count = 0;
  counter_start(&count);
#endif

  // Input size per DPU in bytes.
  uint32_t input_size_dpu_bytes = DPU_INPUT_ARGUMENTS.size;

  // Transferred input size per DPU in bytes
  uint32_t input_size_dpu_bytes_transfer = DPU_INPUT_ARGUMENTS.transfer_size;

  // Binary pointwise operation to perform.
  const enum binary_operation op = DPU_INPUT_ARGUMENTS.operation;

  unsigned int alpha = DPU_INPUT_ARGUMENTS.alpha;

  // Get the address of the current processing chunk for the current tasklet in
  // the MRAM.
  uint32_t baseTasklet = tasklet_id << BLOCK_SIZE_LOG2;

  // Base addresses of the input arrays.
  uint32_t mramBaseAddrX = (uint32_t)DPU_MRAM_HEAP_POINTER;
  uint32_t mramBaseAddrY =
      (uint32_t)(DPU_MRAM_HEAP_POINTER + input_size_dpu_bytes_transfer);

  // Initialize a local cache in WRAM to store the relevant MRAM block.
  T *cacheX = (T *)mem_alloc(BLOCK_SIZE);
  T *cacheT = (T *)mem_alloc(BLOCK_SIZE);

  // Each tasklet is utilizing cyclic mapping to process linear chunks of data.
  // It needs to jump ahead of the work done by all tasklets on the DPU in its
  // next serial iteration, so that it doesn't interfere with any other tasklet.
  // This is similar to cyclic mapping of threads done on GPU to utilize
  // coalesced memory access and reduce bank conflicts in a kernel.
  for (unsigned int byteIndex = baseTasklet; byteIndex < input_size_dpu_bytes;
       byteIndex += BLOCK_SIZE * NR_TASKLETS) {

    // Bound checking.
    // Only record exact number of bytes to be used during the transfer. This
    // will ensure proper handling for the last partial tile of computation.
    uint32_t bytes_to_transfer = BLOCK_SIZE;
    if (byteIndex + bytes_to_transfer > input_size_dpu_bytes_transfer) {
      bytes_to_transfer = input_size_dpu_bytes_transfer - byteIndex;
    }
    if (bytes_to_transfer == 0) {
      continue;
    }

    // Record valid number of elements to be processed during the computation.
    // This is done to implicitly handle the last partial tile of computation.
    uint32_t valid_bytes = bytes_to_transfer;
    if (byteIndex + valid_bytes > input_size_dpu_bytes) {
      valid_bytes = input_size_dpu_bytes - byteIndex;
    }
    if (valid_bytes == 0) {
      continue;
    }
    const unsigned int valid_elems = valid_bytes >> DIV;

    // Load cache with current MRAM block.
    mram_read((__mram_ptr void *)(mramBaseAddrX + byteIndex), cacheX,
              bytes_to_transfer);
    mram_read((__mram_ptr void *)(mramBaseAddrY + byteIndex), cacheT,
              bytes_to_transfer);

    // Computer the pointwise operation.
    pointwise(cacheT, cacheX, valid_elems, op, &alpha);

    // Write cached result to the current MRAM block.
    mram_write(cacheT, (__mram_ptr void *)(mramBaseAddrY + byteIndex),
               bytes_to_transfer);
  }

#if defined(CYCLES) || defined(INSTRUCTIONS)
  result->count += counter_stop(&count);
#endif

  return 0;
}
