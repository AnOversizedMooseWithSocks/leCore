#include "internal/lecore_internal.h"

#include <math.h>
#include <string.h>

static int lecore_hrr_f64_vector_is_finite(
    const double *values,
    uint32_t dimension)
{
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int lecore_hrr_f64_matrix_is_finite(
    const double *rows,
    size_t row_count,
    size_t row_stride,
    uint32_t dimension)
{
    size_t row;

    for (row = 0; row < row_count; ++row) {
        if (!lecore_hrr_f64_vector_is_finite(
                rows + row * row_stride, dimension)) {
            return 0;
        }
    }
    return 1;
}

static void lecore_hrr_bind_f64_raw(
    const double *a,
    const double *b,
    double *output,
    uint32_t dimension)
{
    uint32_t output_index;
    uint32_t left_index;

    if (dimension >= UINT32_C(32)) {
        /*
         * Visit left_index in the same ascending order as the definitional
         * reduction, but update independent outputs contiguously.  This keeps
         * every output's floating-point operation order unchanged while
         * making the inner loops vectorizable.  Public alias checks guarantee
         * that raw output is disjoint from both inputs.
         */
        memset(output, 0, (size_t)dimension * sizeof(*output));
        for (left_index = 0; left_index < dimension; ++left_index) {
            const double left = a[left_index];
            uint32_t right_index = 0;
            const uint32_t first_count = dimension - left_index;

            for (; right_index < first_count; ++right_index) {
                output[left_index + right_index] +=
                    left * b[right_index];
            }
            for (; right_index < dimension; ++right_index) {
                output[right_index - first_count] +=
                    left * b[right_index];
            }
        }
        return;
    }

    for (output_index = 0; output_index < dimension; ++output_index) {
        double sum = 0.0;

        for (left_index = 0; left_index < dimension; ++left_index) {
            uint32_t right_index = output_index >= left_index
                ? output_index - left_index
                : dimension - (left_index - output_index);
            sum += a[left_index] * b[right_index];
        }
        output[output_index] = sum;
    }
}

static void lecore_hrr_unbind_f64_raw(
    const double *composite,
    const double *key,
    double *output,
    uint32_t dimension)
{
    uint32_t output_index;
    uint32_t composite_index;

    if (dimension >= UINT32_C(32)) {
        /* Preserve each output's ascending composite-index reduction order. */
        memset(output, 0, (size_t)dimension * sizeof(*output));
        for (composite_index = 0;
             composite_index < dimension;
             ++composite_index) {
            const double value = composite[composite_index];

            for (output_index = 0;
                 output_index <= composite_index;
                 ++output_index) {
                output[output_index] +=
                    value * key[composite_index - output_index];
            }
            for (; output_index < dimension; ++output_index) {
                output[output_index] += value *
                    key[dimension + composite_index - output_index];
            }
        }
        return;
    }

    /* Equivalent to bind(composite, involution(key)), without a temporary. */
    for (output_index = 0; output_index < dimension; ++output_index) {
        double sum = 0.0;

        for (composite_index = 0; composite_index < dimension;
             ++composite_index) {
            uint32_t key_index = composite_index >= output_index
                ? composite_index - output_index
                : dimension - (output_index - composite_index);
            sum += composite[composite_index] * key[key_index];
        }
        output[output_index] = sum;
    }
}

static void lecore_hrr_bind_f64_selected(
    lecore_context *context,
    const double *a,
    const double *b,
    double *output)
{
#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_bind_f64(context, a, b, output);
        return;
    }
#endif
    lecore_hrr_bind_f64_raw(a, b, output, context->dimension);
}

static void lecore_hrr_unbind_f64_selected(
    lecore_context *context,
    const double *composite,
    const double *key,
    double *output)
{
#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_unbind_f64(
            context, composite, key, output);
        return;
    }
#endif
    lecore_hrr_unbind_f64_raw(
        composite, key, output, context->dimension);
}

static lecore_status lecore_hrr_f64_check_scalar_aliases(
    const lecore_context *context,
    const double *a,
    const double *b,
    const double *output)
{
    lecore_status status;
    const size_t vector_bytes =
        (size_t)context->dimension * sizeof(double);

    status = lecore_internal_check_vector_alias(
        a, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    return lecore_internal_check_vector_alias(
        b, output, vector_bytes, 1);
}

lecore_status LECORE_CALL lecore_hrr_bind_f64(
    lecore_context *context,
    const double *a,
    const double *b,
    double *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    double *target;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (a == NULL || b == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_hrr_f64_check_scalar_aliases(context, a, b, output);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f64_vector_is_finite(a, context->dimension) ||
         !lecore_hrr_f64_vector_is_finite(b, context->dimension))) {
        return LECORE_ENONFINITE;
    }

    vector_bytes = (size_t)context->dimension * sizeof(double);
    target = context->backend == LECORE_BACKEND_DIRECT &&
        (output == a || output == b)
        ? (double *)context->scratch
        : output;
    lecore_hrr_bind_f64_selected(context, a, b, target);
    if (target != output) {
        memcpy(output, target, vector_bytes);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_unbind_f64(
    lecore_context *context,
    const double *composite,
    const double *key,
    double *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    double *target;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (composite == NULL || key == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_hrr_f64_check_scalar_aliases(
        context, composite, key, output);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f64_vector_is_finite(
             composite, context->dimension) ||
         !lecore_hrr_f64_vector_is_finite(key, context->dimension))) {
        return LECORE_ENONFINITE;
    }

    vector_bytes = (size_t)context->dimension * sizeof(double);
    target = context->backend == LECORE_BACKEND_DIRECT &&
        (output == composite || output == key)
        ? (double *)context->scratch
        : output;
    lecore_hrr_unbind_f64_selected(context, composite, key, target);
    if (target != output) {
        memcpy(output, target, vector_bytes);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_bind_batch_f64(
    lecore_context *context,
    const double *a_rows,
    size_t a_stride,
    const double *b_rows,
    size_t b_stride,
    size_t row_count,
    double *out_rows,
    size_t out_stride)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    size_t a_bytes;
    size_t b_bytes;
    size_t out_bytes;
    size_t row;

    if (status != LECORE_OK) {
        return status;
    }
    if (a_rows == NULL || b_rows == NULL || out_rows == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_internal_matrix_span(
        context->dimension, row_count, a_stride, sizeof(double), &a_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension, row_count, b_stride, sizeof(double), &b_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension, row_count, out_stride, sizeof(double), &out_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (lecore_internal_ranges_overlap(a_rows, a_bytes, out_rows, out_bytes) ||
        lecore_internal_ranges_overlap(b_rows, b_bytes, out_rows, out_bytes)) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f64_matrix_is_finite(
             a_rows, row_count, a_stride, context->dimension) ||
         !lecore_hrr_f64_matrix_is_finite(
             b_rows, row_count, b_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }

    for (row = 0; row < row_count; ++row) {
        lecore_hrr_bind_f64_selected(
            context,
            a_rows + row * a_stride,
            b_rows + row * b_stride,
            out_rows + row * out_stride);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_bind_fixed_f64(
    lecore_context *context,
    const double *role,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *out_rows,
    size_t out_stride)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    size_t rows_bytes;
    size_t out_bytes;
    size_t row;
    size_t role_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (role == NULL || rows == NULL || out_rows == NULL) {
        return LECORE_EINVAL;
    }
    role_bytes = (size_t)context->dimension * sizeof(double);
    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        row_stride,
        sizeof(double),
        &rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        out_stride,
        sizeof(double),
        &out_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (lecore_internal_ranges_overlap(
            role, role_bytes, out_rows, out_bytes) ||
        lecore_internal_ranges_overlap(
            rows, rows_bytes, out_rows, out_bytes)) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f64_vector_is_finite(role, context->dimension) ||
         !lecore_hrr_f64_matrix_is_finite(
             rows, row_count, row_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }

#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_bind_fixed_f64(
            context,
            role,
            rows,
            row_count,
            row_stride,
            out_rows,
            out_stride);
        return LECORE_OK;
    }
#endif
    for (row = 0; row < row_count; ++row) {
        lecore_hrr_bind_f64_raw(
            role,
            rows + row * row_stride,
            out_rows + row * out_stride,
            context->dimension);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_unbind_all_f64(
    lecore_context *context,
    const double *trace,
    const double *keys,
    size_t key_count,
    size_t key_stride,
    double *out_rows,
    size_t out_stride)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    size_t keys_bytes;
    size_t out_bytes;
    size_t key;
    size_t trace_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (trace == NULL || keys == NULL || out_rows == NULL) {
        return LECORE_EINVAL;
    }
    trace_bytes = (size_t)context->dimension * sizeof(double);
    status = lecore_internal_matrix_span(
        context->dimension,
        key_count,
        key_stride,
        sizeof(double),
        &keys_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension,
        key_count,
        out_stride,
        sizeof(double),
        &out_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (lecore_internal_ranges_overlap(
            trace, trace_bytes, out_rows, out_bytes) ||
        lecore_internal_ranges_overlap(
            keys, keys_bytes, out_rows, out_bytes)) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f64_vector_is_finite(trace, context->dimension) ||
         !lecore_hrr_f64_matrix_is_finite(
             keys, key_count, key_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }

#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_unbind_all_f64(
            context,
            trace,
            keys,
            key_count,
            key_stride,
            out_rows,
            out_stride);
        return LECORE_OK;
    }
#endif
    for (key = 0; key < key_count; ++key) {
        lecore_hrr_unbind_f64_raw(
            trace,
            keys + key * key_stride,
            out_rows + key * out_stride,
            context->dimension);
    }
    return LECORE_OK;
}
