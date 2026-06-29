#ifndef HOLO_PROGRAM_H
#define HOLO_PROGRAM_H

#include "holo_core.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    HOLO_PROGRAM_OP_LOAD = 0,
    HOLO_PROGRAM_OP_BIND = 1,
    HOLO_PROGRAM_OP_BUNDLE = 2,
    HOLO_PROGRAM_OP_PERMUTE = 3,
    HOLO_PROGRAM_OP_IFMATCH = 6,
    HOLO_PROGRAM_OP_HALT = 9
};

int holo_program_run_basic(holo_engine *engine,
                           const double *program,
                           const double *positions,
                           size_t position_count,
                           const double *op_role,
                           const double *arg_role,
                           const double *op_vectors,
                           const double *op_norms,
                           size_t op_count,
                           const double *data_vectors,
                           const double *data_norms,
                           size_t data_count,
                           const double *init_acc,
                           int has_init_acc,
                           size_t max_steps,
                           double branch_tol,
                           double *out_acc,
                           int *out_has_acc,
                           size_t *out_op_indices,
                           size_t *out_arg_indices,
                           size_t trace_capacity,
                           size_t *out_trace_count);

#ifdef __cplusplus
}
#endif

#endif
