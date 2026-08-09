#include "internal/lecore_internal.h"

#include <math.h>
#include <string.h>

static int lecore_hrr_f32_vector_is_finite(
    const float *values,
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

static int lecore_hrr_f32_matrix_is_finite(
    const float *rows,
    size_t row_count,
    size_t row_stride,
    uint32_t dimension)
{
    size_t row;

    for (row = 0; row < row_count; ++row) {
        if (!lecore_hrr_f32_vector_is_finite(
                rows + row * row_stride, dimension)) {
            return 0;
        }
    }
    return 1;
}

static void lecore_hrr_bind_f32_raw(
    const float *a,
    const float *b,
    float *output,
    uint32_t dimension)
{
    uint32_t output_index;
    uint32_t left_index;

    if (dimension >= UINT32_C(32)) {
        /*
         * Preserve the definitional ascending reduction order per output,
         * while exposing contiguous independent updates to the compiler.
         */
        memset(output, 0, (size_t)dimension * sizeof(*output));
        for (left_index = 0; left_index < dimension; ++left_index) {
            const float left = a[left_index];
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
        float sum = 0.0f;

        for (left_index = 0; left_index < dimension; ++left_index) {
            uint32_t right_index = output_index >= left_index
                ? output_index - left_index
                : dimension - (left_index - output_index);
            sum += a[left_index] * b[right_index];
        }
        output[output_index] = sum;
    }
}

static void lecore_hrr_unbind_f32_raw(
    const float *composite,
    const float *key,
    float *output,
    uint32_t dimension)
{
    uint32_t output_index;
    uint32_t composite_index;

    if (dimension >= UINT32_C(32)) {
        memset(output, 0, (size_t)dimension * sizeof(*output));
        for (composite_index = 0;
             composite_index < dimension;
             ++composite_index) {
            const float value = composite[composite_index];

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

    for (output_index = 0; output_index < dimension; ++output_index) {
        float sum = 0.0f;

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

static void lecore_hrr_bind_f32_selected(
    lecore_context *context,
    const float *a,
    const float *b,
    float *output)
{
#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_bind_f32(context, a, b, output);
        return;
    }
#endif
    lecore_hrr_bind_f32_raw(a, b, output, context->dimension);
}

static void lecore_hrr_unbind_f32_selected(
    lecore_context *context,
    const float *composite,
    const float *key,
    float *output)
{
#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_unbind_f32(
            context, composite, key, output);
        return;
    }
#endif
    lecore_hrr_unbind_f32_raw(
        composite, key, output, context->dimension);
}

static lecore_status lecore_hrr_f32_check_scalar_aliases(
    const lecore_context *context,
    const float *a,
    const float *b,
    const float *output)
{
    lecore_status status;
    const size_t vector_bytes =
        (size_t)context->dimension * sizeof(float);

    status = lecore_internal_check_vector_alias(
        a, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    return lecore_internal_check_vector_alias(
        b, output, vector_bytes, 1);
}

lecore_status LECORE_CALL lecore_hrr_bind_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    float *target;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (a == NULL || b == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_hrr_f32_check_scalar_aliases(context, a, b, output);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f32_vector_is_finite(a, context->dimension) ||
         !lecore_hrr_f32_vector_is_finite(b, context->dimension))) {
        return LECORE_ENONFINITE;
    }

    vector_bytes = (size_t)context->dimension * sizeof(float);
    target = context->backend == LECORE_BACKEND_DIRECT &&
        (output == a || output == b)
        ? (float *)context->scratch
        : output;
    lecore_hrr_bind_f32_selected(context, a, b, target);
    if (target != output) {
        memcpy(output, target, vector_bytes);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_unbind_f32(
    lecore_context *context,
    const float *composite,
    const float *key,
    float *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    float *target;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (composite == NULL || key == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_hrr_f32_check_scalar_aliases(
        context, composite, key, output);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f32_vector_is_finite(
             composite, context->dimension) ||
         !lecore_hrr_f32_vector_is_finite(key, context->dimension))) {
        return LECORE_ENONFINITE;
    }

    vector_bytes = (size_t)context->dimension * sizeof(float);
    target = context->backend == LECORE_BACKEND_DIRECT &&
        (output == composite || output == key)
        ? (float *)context->scratch
        : output;
    lecore_hrr_unbind_f32_selected(context, composite, key, target);
    if (target != output) {
        memcpy(output, target, vector_bytes);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_bind_batch_f32(
    lecore_context *context,
    const float *a_rows,
    size_t a_stride,
    const float *b_rows,
    size_t b_stride,
    size_t row_count,
    float *out_rows,
    size_t out_stride)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
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
        context->dimension, row_count, a_stride, sizeof(float), &a_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension, row_count, b_stride, sizeof(float), &b_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension, row_count, out_stride, sizeof(float), &out_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (lecore_internal_ranges_overlap(a_rows, a_bytes, out_rows, out_bytes) ||
        lecore_internal_ranges_overlap(b_rows, b_bytes, out_rows, out_bytes)) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_hrr_f32_matrix_is_finite(
             a_rows, row_count, a_stride, context->dimension) ||
         !lecore_hrr_f32_matrix_is_finite(
             b_rows, row_count, b_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }

    for (row = 0; row < row_count; ++row) {
        lecore_hrr_bind_f32_selected(
            context,
            a_rows + row * a_stride,
            b_rows + row * b_stride,
            out_rows + row * out_stride);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_bind_fixed_f32(
    lecore_context *context,
    const float *role,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *out_rows,
    size_t out_stride)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    size_t role_bytes;
    size_t rows_bytes;
    size_t out_bytes;
    size_t row;

    if (status != LECORE_OK) {
        return status;
    }
    if (role == NULL || rows == NULL || out_rows == NULL) {
        return LECORE_EINVAL;
    }
    role_bytes = (size_t)context->dimension * sizeof(float);
    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        row_stride,
        sizeof(float),
        &rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        out_stride,
        sizeof(float),
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
        (!lecore_hrr_f32_vector_is_finite(role, context->dimension) ||
         !lecore_hrr_f32_matrix_is_finite(
             rows, row_count, row_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }

#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_bind_fixed_f32(
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
        lecore_hrr_bind_f32_raw(
            role,
            rows + row * row_stride,
            out_rows + row * out_stride,
            context->dimension);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_hrr_unbind_all_f32(
    lecore_context *context,
    const float *trace,
    const float *keys,
    size_t key_count,
    size_t key_stride,
    float *out_rows,
    size_t out_stride)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    size_t trace_bytes;
    size_t keys_bytes;
    size_t out_bytes;
    size_t key;

    if (status != LECORE_OK) {
        return status;
    }
    if (trace == NULL || keys == NULL || out_rows == NULL) {
        return LECORE_EINVAL;
    }
    trace_bytes = (size_t)context->dimension * sizeof(float);
    status = lecore_internal_matrix_span(
        context->dimension,
        key_count,
        key_stride,
        sizeof(float),
        &keys_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_internal_matrix_span(
        context->dimension,
        key_count,
        out_stride,
        sizeof(float),
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
        (!lecore_hrr_f32_vector_is_finite(trace, context->dimension) ||
         !lecore_hrr_f32_matrix_is_finite(
             keys, key_count, key_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }

#if LECORE_ENABLE_RADIX2
    if (context->backend == LECORE_BACKEND_RADIX2) {
        lecore_internal_hrr_radix2_unbind_all_f32(
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
        lecore_hrr_unbind_f32_raw(
            trace,
            keys + key * key_stride,
            out_rows + key * out_stride,
            context->dimension);
    }
    return LECORE_OK;
}
