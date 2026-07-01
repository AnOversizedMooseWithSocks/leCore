#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include "holo_program.h"

#include <stdlib.h>
#include <string.h>

#define HOLO_ALIGN 64U

static void *alloc_zeroed(size_t count, size_t size)
{
    void *ptr = NULL;
    const size_t bytes = count * size;
    if (count != 0 && size > ((size_t)-1) / count) {
        return NULL;
    }
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, HOLO_ALIGN);
    if (!ptr) {
        return NULL;
    }
#elif defined(__APPLE__) || defined(__unix__)
    if (posix_memalign(&ptr, HOLO_ALIGN, bytes) != 0) {
        return NULL;
    }
#else
    ptr = malloc(bytes);
    if (!ptr) {
        return NULL;
    }
#endif
    memset(ptr, 0, bytes);
    return ptr;
}

static void free_aligned(void *ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

static int cleanup_index(size_t dim,
                         const double *query,
                         const double *matrix,
                         const double *norms,
                         size_t count,
                         size_t *out_index)
{
    holo_match match;
    int rc;
    if (!out_index || count == 0) {
        return HOLO_EINVAL;
    }
    rc = holo_cleanup_topk_with_norms(dim, query, matrix, norms, NULL, count, 1, &match);
    if (rc != HOLO_OK) {
        return rc;
    }
    *out_index = match.index;
    return HOLO_OK;
}

static int append_trace(size_t op_index,
                        size_t arg_index,
                        size_t *out_op_indices,
                        size_t *out_arg_indices,
                        size_t trace_capacity,
                        size_t *trace_count)
{
    if (!trace_count) {
        return HOLO_EINVAL;
    }
    if (*trace_count >= trace_capacity) {
        return HOLO_EINVAL;
    }
    if (out_op_indices) {
        out_op_indices[*trace_count] = op_index;
    }
    if (out_arg_indices) {
        out_arg_indices[*trace_count] = arg_index;
    }
    *trace_count += 1;
    return HOLO_OK;
}

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
                           size_t *out_trace_count)
{
    const size_t dim = holo_engine_dim(engine);
    double *workspace = NULL;
    double *program_real;
    double *program_imag;
    double *raw;
    double *raw_real;
    double *raw_imag;
    double *op_query;
    double *arg_query;
    double *acc;
    double *pair;
    double *bundle_rows;
    size_t pc = 0;
    size_t trace_count = 0;
    int acc_valid = has_init_acc ? 1 : 0;
    int rc = HOLO_OK;

    if (!engine || !program || !positions || !op_role || !arg_role ||
        !op_vectors || !data_vectors || !out_acc || !out_has_acc ||
        !out_trace_count || dim == 0 || op_count <= HOLO_PROGRAM_OP_HALT ||
        data_count == 0 || max_steps == 0 || trace_capacity == 0) {
        return HOLO_EINVAL;
    }

    if (dim > ((size_t)-1) / 11U) {
        return HOLO_ENOMEM;
    }
    workspace = (double *)alloc_zeroed(11U * dim, sizeof(*workspace));
    if (!workspace) {
        rc = HOLO_ENOMEM;
        goto done;
    }
    program_real = workspace;
    program_imag = program_real + dim;
    raw = program_imag + dim;
    raw_real = raw + dim;
    raw_imag = raw_real + dim;
    op_query = raw_imag + dim;
    arg_query = op_query + dim;
    acc = arg_query + dim;
    pair = acc + dim;
    bundle_rows = pair + dim;

    if (init_acc && acc_valid) {
        memcpy(acc, init_acc, dim * sizeof(acc[0]));
    }

    rc = holo_spectrum_from_real(engine, program, program_real, program_imag);
    if (rc != HOLO_OK) {
        goto done;
    }

    for (size_t step = 0; step < max_steps && pc < position_count; ++step) {
        size_t op_index = 0;
        size_t arg_index = 0;
        const double *arg_vec;

        rc = holo_unbind_spectrum(engine,
                                  program_real,
                                  program_imag,
                                  positions + pc * dim,
                                  raw);
        if (rc != HOLO_OK) {
            goto done;
        }
        rc = holo_spectrum_from_real(engine, raw, raw_real, raw_imag);
        if (rc != HOLO_OK) {
            goto done;
        }
        rc = holo_unbind_spectrum(engine, raw_real, raw_imag, op_role, op_query);
        if (rc != HOLO_OK) {
            goto done;
        }
        rc = cleanup_index(dim, op_query, op_vectors, op_norms, op_count, &op_index);
        if (rc != HOLO_OK) {
            goto done;
        }

        if (op_index == HOLO_PROGRAM_OP_HALT) {
            break;
        }
        if (op_index != HOLO_PROGRAM_OP_LOAD &&
            op_index != HOLO_PROGRAM_OP_BIND &&
            op_index != HOLO_PROGRAM_OP_BUNDLE &&
            op_index != HOLO_PROGRAM_OP_PERMUTE &&
            op_index != HOLO_PROGRAM_OP_IFMATCH) {
            rc = HOLO_EINVAL;
            goto done;
        }

        rc = holo_unbind_spectrum(engine, raw_real, raw_imag, arg_role, arg_query);
        if (rc != HOLO_OK) {
            goto done;
        }
        rc = cleanup_index(dim, arg_query, data_vectors, data_norms, data_count, &arg_index);
        if (rc != HOLO_OK) {
            goto done;
        }
        rc = append_trace(op_index,
                          arg_index,
                          out_op_indices,
                          out_arg_indices,
                          trace_capacity,
                          &trace_count);
        if (rc != HOLO_OK) {
            goto done;
        }

        arg_vec = data_vectors + arg_index * dim;
        if (op_index == HOLO_PROGRAM_OP_IFMATCH) {
            const int matched = acc_valid && holo_cosine(dim, acc, arg_vec) >= branch_tol;
            pc += matched ? 1U : 2U;
            continue;
        }
        if (op_index == HOLO_PROGRAM_OP_LOAD || !acc_valid) {
            memcpy(acc, arg_vec, dim * sizeof(acc[0]));
            acc_valid = 1;
        } else if (op_index == HOLO_PROGRAM_OP_BIND) {
            rc = holo_bind(engine, acc, arg_vec, pair);
            if (rc != HOLO_OK) {
                goto done;
            }
            memcpy(acc, pair, dim * sizeof(acc[0]));
        } else if (op_index == HOLO_PROGRAM_OP_BUNDLE) {
            memcpy(bundle_rows, acc, dim * sizeof(bundle_rows[0]));
            memcpy(bundle_rows + dim, arg_vec, dim * sizeof(bundle_rows[0]));
            rc = holo_bundle(dim, bundle_rows, NULL, 2, acc);
            if (rc != HOLO_OK) {
                goto done;
            }
        } else if (op_index == HOLO_PROGRAM_OP_PERMUTE) {
            rc = holo_permute(dim, acc, 1, pair);
            if (rc != HOLO_OK) {
                goto done;
            }
            memcpy(acc, pair, dim * sizeof(acc[0]));
        }
        pc += 1;
    }

    if (acc_valid) {
        memcpy(out_acc, acc, dim * sizeof(out_acc[0]));
    } else {
        memset(out_acc, 0, dim * sizeof(out_acc[0]));
    }
    *out_has_acc = acc_valid;
    *out_trace_count = trace_count;

done:
    free_aligned(workspace);
    return rc;
}
