#ifndef _PARAMS_H_
#define _PARAMS_H_

#include "common.h"
#include <strings.h>

typedef struct Params {
  unsigned int input_size;
  enum binary_operation operation;
  T alpha;
  int n_warmup;
  int n_reps;
} Params;

static void usage();

static enum binary_operation parse_operation(const char *arg) {
  if (arg && *arg) {
    if (strcasecmp(arg, "add") == 0) {
      return BINARY_OP_ADD;
    }
    if (strcasecmp(arg, "sub") == 0 || strcasecmp(arg, "subtract") == 0) {
      return BINARY_OP_SUB;
    }
    if (strcasecmp(arg, "mul") == 0 || strcasecmp(arg, "multiply") == 0) {
      return BINARY_OP_MUL;
    }
    if (strcasecmp(arg, "div") == 0 || strcasecmp(arg, "divide") == 0) {
      return BINARY_OP_DIV;
    }
  }
  fprintf(stderr, "\nInvalid operation '%s'\n", arg ? arg : "(null)");
  usage();
  exit(1);
}

static void usage() {
  fprintf(stderr,
          "\nUsage:  ./program [options]"
          "\n"
          "\nGeneral options:"
          "\n    -h        help"
          "\n    -w <W>    # of untimed warmup iterations (default=0)"
          "\n    -e <E>    # of timed repetition iterations (default=1)"
          "\n"
          "\nWorkload-specific options:"
          "\n    -i <I>    input size (default=2621440 elements)"
          "\n    -a <A>    alpha (default=100)"
          "\n    -o <OP>   binary operation {add, sub, mul, div} (default=add)"
          "\n");
}

struct Params input_params(int argc, char **argv) {
  struct Params p;
  p.input_size = 2621440;
  p.operation = BINARY_OP_ADD;
  p.alpha = 100;
  p.n_warmup = 0;
  p.n_reps = 1;

  int opt;
  while ((opt = getopt(argc, argv, "hi:a:o:w:e:")) >= 0) {
    switch (opt) {
    case 'h':
      usage();
      exit(0);
      break;
    case 'i':
      p.input_size = atoi(optarg);
      break;
    case 'a':
      p.alpha = atoi(optarg);
      break;
    case 'o':
      p.operation = parse_operation(optarg);
      break;
    case 'w':
      p.n_warmup = atoi(optarg);
      break;
    case 'e':
      p.n_reps = atoi(optarg);
      break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  assert(NR_DPUS > 0 && "Invalid # of dpus!");

  return p;
}
#endif
