/**
 * app.c
 * Host Application Source File
 *
 */
#include <assert.h>
#include <dpu.h>
#include <dpu_log.h>
#include <getopt.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "../support/common.h"
#include "../support/params.h"
#include "../support/timer.h"

// Define the DPU Binary path as DPU_BINARY here.
#ifndef DPU_BINARY
#define DPU_BINARY "./bin/dpu_code"
#endif

// Pointer declaration.
static T *X;
static T *Y;
static T *Y_host;

// Create input arrays.
static void read_input(T *A, T *B, unsigned int numElements) {
  srand(0);
  printf("numElements\t%u\n", numElements);
  for (unsigned int i = 0; i < numElements; i++) {
    A[i] = (T)(rand());
    B[i] = (T)(rand());
  }
}

// Compute output in the host for verification purposes.
static void axpy_host(T *A, T *B, unsigned int numElements,
                      enum binary_operation op) {
  switch (op) {
  case BINARY_OP_ADD:
    for (unsigned int i = 0; i < numElements; i++) {
      B[i] = A[i] + B[i];
    }
    break;
  case BINARY_OP_SUB:
    for (unsigned int i = 0; i < numElements; i++) {
      B[i] = A[i] - B[i];
    }
    break;
  case BINARY_OP_MUL:
    for (unsigned int i = 0; i < numElements; i++) {
      B[i] = A[i] * B[i];
    }
    break;
  case BINARY_OP_DIV:
    for (unsigned int i = 0; i < numElements; i++) {
      B[i] = A[i] / B[i];
    }
    break;
  default:
    for (unsigned int i = 0; i < numElements; i++) {
      B[i] = A[i] + B[i];
    }
    break;
  }
}

int main(int argc, char **argv) {
  // Input parameters.
  struct Params p = input_params(argc, argv);

  // Timer declaration
  Timer timer;
#if defined(CYCLES) || defined(INSTRUCTIONS)
  double cc = 0;
  double cc_min = 0;
#endif

  // Allocate DPUs.
  struct dpu_set_t dpu_set, dpu;
  uint32_t nr_of_dpus;
  DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));

  // Number of DPUs in the DPU set.
  DPU_ASSERT(dpu_get_nr_dpus(dpu_set, &nr_of_dpus));
  printf("Allocated %d DPU(s)\t", nr_of_dpus);
  printf("NR_TASKLETS\t%d\tBLOCK\t%d\n", NR_TASKLETS, BLOCK);

  // Load binary.
  DPU_ASSERT(dpu_load(dpu_set, DPU_BINARY, NULL));

  // Total input elements.
  const unsigned int input_num_elems = p.input_size;

  // Total input elements rounded upto 8.
  const unsigned int input_num_elems_roundup8 =
      ((input_num_elems * sizeof(T)) % 8) != 0 ? roundup(input_num_elems, 8)
                                               : input_num_elems;

  // Number of input elements per dpu.
  const unsigned int input_num_elems_dpu = divceil(input_num_elems, nr_of_dpus);

  // Number of input elements per dpu rounded upto 8.
  const unsigned int input_num_elems_dpu_roundup8 =
      ((input_num_elems_dpu * sizeof(T)) % 8) != 0
          ? roundup(input_num_elems_dpu, 8)
          : input_num_elems_dpu;

  // Input/output allocation in host main memory.
  X = malloc(input_num_elems_dpu_roundup8 * nr_of_dpus * sizeof(T));
  Y = malloc(input_num_elems_dpu_roundup8 * nr_of_dpus * sizeof(T));
  Y_host = malloc(input_num_elems_dpu_roundup8 * nr_of_dpus * sizeof(T));
  T *bufferX = X;
  T *bufferY = Y;
  unsigned int i = 0;

  // Create an input array with arbitrary data.
  read_input(X, Y, input_num_elems);
  memcpy(Y_host, Y, input_num_elems_dpu_roundup8 * nr_of_dpus * sizeof(T));

  // Binary pointwise operation to be performed on the array elements.
  const enum binary_operation operation = p.operation;

  // Loop over main kernel.
  for (int rep = 0; rep < p.n_warmup + p.n_reps; rep++) {
    // Compute output on CPU (verification purposes).
    if (rep >= p.n_warmup)
      start(&timer, 0, rep - p.n_warmup);
    axpy_host(X, Y_host, input_num_elems, operation);
    if (rep >= p.n_warmup)
      stop(&timer, 0);

    printf("Load input data\n");
    // Input arguments.
    unsigned int kernel = 0;
    dpu_arguments_t input_arguments[NR_DPUS];
    for (i = 0; i < nr_of_dpus - 1; i++) {
      input_arguments[i].size = input_num_elems_dpu_roundup8 * sizeof(T);
      input_arguments[i].transfer_size =
          input_num_elems_dpu_roundup8 * sizeof(T);
      input_arguments[i].kernel = kernel;
      input_arguments[i].operation = operation;
      input_arguments[i].alpha = p.alpha;
    }
    input_arguments[nr_of_dpus - 1].size =
        (input_num_elems_roundup8 -
         input_num_elems_dpu_roundup8 * (NR_DPUS - 1)) *
        sizeof(T);
    input_arguments[nr_of_dpus - 1].transfer_size =
        input_num_elems_dpu_roundup8 * sizeof(T);
    input_arguments[nr_of_dpus - 1].kernel = kernel;
    input_arguments[nr_of_dpus - 1].operation = operation;
    input_arguments[nr_of_dpus - 1].alpha = p.alpha;

    if (rep >= p.n_warmup)
      start(&timer, 1, rep - p.n_warmup);
    i = 0;
    const size_t transfer_size_bytes = input_num_elems_dpu_roundup8 * sizeof(T);
    const size_t elements_per_dpu = input_num_elems_dpu_roundup8;
    // Copy input arguments.
    // Parallel transfers.
    DPU_FOREACH(dpu_set, dpu, i) {
      DPU_ASSERT(dpu_prepare_xfer(dpu, &input_arguments[i]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, "DPU_INPUT_ARGUMENTS", 0,
                             sizeof(input_arguments[0]), DPU_XFER_DEFAULT));

    // Copy input arrays.
    // Serial transfers.
#ifdef SERIAL

    DPU_FOREACH(dpu_set, dpu, i) {
      T *src_x = bufferX + (i * elements_per_dpu);
      T *src_y = bufferY + (i * elements_per_dpu);
      DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, src_x,
                             transfer_size_bytes));
      DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                             transfer_size_bytes, src_y, transfer_size_bytes));
    }

    // Parallel transfers.
#else

    DPU_FOREACH(dpu_set, dpu, i) {
      T *src_x = bufferX + (i * elements_per_dpu);
      DPU_ASSERT(dpu_prepare_xfer(dpu, src_x));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME, 0, transfer_size_bytes,
                             DPU_XFER_DEFAULT));
    i = 0;
    DPU_FOREACH(dpu_set, dpu, i) {
      T *src_y = bufferY + (i * elements_per_dpu);
      DPU_ASSERT(dpu_prepare_xfer(dpu, src_y));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME, transfer_size_bytes,
                             transfer_size_bytes, DPU_XFER_DEFAULT));

#endif
    if (rep >= p.n_warmup)
      stop(&timer, 1);

    printf("Run program on DPU(s) \n");
    // Run DPU kernel.
    if (rep >= p.n_warmup) {
      start(&timer, 2, rep - p.n_warmup);
    }
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    if (rep >= p.n_warmup) {
      stop(&timer, 2);
    }

#if PRINT
    {
      unsigned int each_dpu = 0;
      printf("Display DPU Logs\n");
      DPU_FOREACH(dpu_set, dpu) {
        printf("DPU#%d:\n", each_dpu);
        DPU_ASSERT(dpulog_read_for_dpu(dpu.dpu, stdout));
        each_dpu++;
      }
    }
#endif

    printf("Retrieve results\n");
    if (rep >= p.n_warmup)
      start(&timer, 3, rep - p.n_warmup);
    i = 0;
    // Copy output array.
    // Serial transfers.
#ifdef SERIAL

    DPU_FOREACH(dpu_set, dpu, i) {
      T *dst_y = bufferY + (i * elements_per_dpu);
      DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME,
                               transfer_size_bytes, dst_y,
                               transfer_size_bytes));
    }

    // Parallel transfers.
#else

    DPU_FOREACH(dpu_set, dpu, i) {
      T *dst_y = bufferY + (i * elements_per_dpu);
      DPU_ASSERT(dpu_prepare_xfer(dpu, dst_y));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU,
                             DPU_MRAM_HEAP_POINTER_NAME, transfer_size_bytes,
                             transfer_size_bytes, DPU_XFER_DEFAULT));

#endif
    if (rep >= p.n_warmup)
      stop(&timer, 3);

#if defined(CYCLES) || defined(INSTRUCTIONS)
    dpu_results_t results[nr_of_dpus];
    // Parallel transfers.
    dpu_results_t *results_retrieve[nr_of_dpus];
    DPU_FOREACH(dpu_set, dpu, i) {
      results_retrieve[i] =
          (dpu_results_t *)malloc(NR_TASKLETS * sizeof(dpu_results_t));
      DPU_ASSERT(dpu_prepare_xfer(dpu, results_retrieve[i]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, "DPU_RESULTS", 0,
                             NR_TASKLETS * sizeof(dpu_results_t),
                             DPU_XFER_DEFAULT));
    DPU_FOREACH(dpu_set, dpu, i) {
      results[i].count = 0;
      // Retrieve tasklet count.
      for (unsigned int each_tasklet = 0; each_tasklet < NR_TASKLETS;
           each_tasklet++) {
        if (results_retrieve[i][each_tasklet].count > results[i].count)
          results[i].count = results_retrieve[i][each_tasklet].count;
      }
      free(results_retrieve[i]);
    }

    uint64_t max_count = 0;
    uint64_t min_count = 0xFFFFFFFFFFFFFFFF;
    // Print performance results.
    if (rep >= p.n_warmup) {
      i = 0;
      DPU_FOREACH(dpu_set, dpu) {
        if (results[i].count > max_count)
          max_count = results[i].count;
        if (results[i].count < min_count)
          min_count = results[i].count;
        i++;
      }
      cc += (double)max_count;
      cc_min += (double)min_count;
    }
#endif
  }
#ifdef CYCLES
  printf("DPU cycles  = %g\n", cc / p.n_reps);
#elif INSTRUCTIONS
  printf("DPU instructions  = %g\n", cc / p.n_reps);
#endif

  // Print timing results.
  printf("CPU ");
  print(&timer, 0, p.n_reps);
  printf("CPU-DPU ");
  print(&timer, 1, p.n_reps);
  printf("DPU Kernel ");
  print(&timer, 2, p.n_reps);
  printf("DPU-CPU ");
  print(&timer, 3, p.n_reps);

  // Check output.
  bool status = true;
  for (i = 0; i < input_num_elems; i++) {
    if (Y_host[i] != Y[i]) {
      status = false;
      printf("%d: %u -- %u\n", i, Y_host[i], Y[i]);
    }
  }
  if (status) {
    printf("[" ANSI_COLOR_GREEN "OK" ANSI_COLOR_RESET "] Outputs are equal\n");
  } else {
    printf("[" ANSI_COLOR_RED "ERROR" ANSI_COLOR_RESET "] Outputs differ!\n");
  }

  // Deallocation.
  free(X);
  free(Y);
  free(Y_host);
  DPU_ASSERT(dpu_free(dpu_set));

  return status ? 0 : -1;
}
