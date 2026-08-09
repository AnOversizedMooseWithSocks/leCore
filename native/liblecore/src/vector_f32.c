#include "internal/lecore_internal.h"

#include <math.h>
#include <string.h>

static int lecore_f32_vector_is_finite(const float *values, uint32_t dimension)
{
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int lecore_f32_matrix_is_finite(
    const float *rows,
    size_t row_count,
    size_t row_stride,
    uint32_t dimension)
{
    size_t row;

    for (row = 0; row < row_count; ++row) {
        if (!lecore_f32_vector_is_finite(
                rows + row * row_stride, dimension)) {
            return 0;
        }
    }
    return 1;
}

static float lecore_f32_dot_raw(
    const float *a,
    const float *b,
    uint32_t dimension)
{
    float result = 0.0f;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        result += a[index] * b[index];
    }
    return result;
}

static float lecore_f32_cosine_raw(
    const float *a,
    const float *b,
    uint32_t dimension)
{
    float dot;
    float norm_a_squared = 0.0f;
    float norm_b_squared = 0.0f;
    float norm_a;
    float norm_b;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        norm_a_squared += a[index] * a[index];
        norm_b_squared += b[index] * b[index];
    }
    if (norm_a_squared == 0.0f || norm_b_squared == 0.0f) {
        return 0.0f;
    }
    norm_a = sqrtf(norm_a_squared);
    norm_b = sqrtf(norm_b_squared);
    dot = lecore_f32_dot_raw(a, b, dimension);
    return dot / (norm_a * norm_b);
}

static float lecore_f32_cosine_with_query_norm_squared(
    const float *query,
    float query_norm_squared,
    float query_norm,
    const float *row,
    uint32_t dimension)
{
    float row_norm_squared = 0.0f;
    float row_norm;
    float dot;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        row_norm_squared += row[index] * row[index];
    }
    if (query_norm_squared == 0.0f || row_norm_squared == 0.0f) {
        return 0.0f;
    }
    row_norm = sqrtf(row_norm_squared);
    dot = lecore_f32_dot_raw(query, row, dimension);
    return dot / (query_norm * row_norm);
}

static lecore_status lecore_f32_prepare_matrix_output(
    const lecore_context *context,
    const float *query,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    const float *out_values,
    size_t *out_rows_bytes)
{
    lecore_status status;
    size_t output_bytes;
    const size_t vector_bytes = (size_t)context->dimension * sizeof(float);

    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        row_stride,
        sizeof(float),
        out_rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (row_count > SIZE_MAX / sizeof(float)) {
        return LECORE_EOVERFLOW;
    }
    output_bytes = row_count * sizeof(float);
    if (lecore_internal_ranges_overlap(
            query, vector_bytes, out_values, output_bytes) ||
        lecore_internal_ranges_overlap(
            rows, *out_rows_bytes, out_values, output_bytes)) {
        return LECORE_EINVAL;
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_validate_f32(
    const float *values,
    size_t count)
{
    size_t index;

    if (count == 0) {
        return LECORE_OK;
    }
    if (values == NULL) {
        return LECORE_EINVAL;
    }
    for (index = 0; index < count; ++index) {
        if (!isfinite(values[index])) {
            return LECORE_ENONFINITE;
        }
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_normalize_f32(
    lecore_context *context,
    const float *input,
    float *output)
{
    lecore_status status;
    float norm_squared = 0.0f;
    float norm;
    uint32_t index;
    size_t vector_bytes;

    status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    if (status != LECORE_OK) {
        return status;
    }
    if (input == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    vector_bytes = (size_t)context->dimension * sizeof(float);
    status = lecore_internal_check_vector_alias(
        input, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f32_vector_is_finite(input, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    for (index = 0; index < context->dimension; ++index) {
        norm_squared += input[index] * input[index];
    }
    norm = sqrtf(norm_squared);
    if (norm > 0.0f) {
        for (index = 0; index < context->dimension; ++index) {
            output[index] = input[index] / norm;
        }
    } else if (output != input) {
        memcpy(output, input, vector_bytes);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_dot_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *out_dot)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);

    if (status != LECORE_OK) {
        return status;
    }
    if (a == NULL || b == NULL || out_dot == NULL) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_f32_vector_is_finite(a, context->dimension) ||
         !lecore_f32_vector_is_finite(b, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    *out_dot = lecore_f32_dot_raw(a, b, context->dimension);
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_dot_many_f32(
    lecore_context *context,
    const float *query,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *out_scores)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    size_t rows_bytes;
    size_t row;

    if (status != LECORE_OK) {
        return status;
    }
    if (query == NULL || rows == NULL || out_scores == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_f32_prepare_matrix_output(
        context,
        query,
        rows,
        row_count,
        row_stride,
        out_scores,
        &rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_f32_vector_is_finite(query, context->dimension) ||
         !lecore_f32_matrix_is_finite(
             rows, row_count, row_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    (void)rows_bytes;
    for (row = 0; row < row_count; ++row) {
        out_scores[row] = lecore_f32_dot_raw(
            query, rows + row * row_stride, context->dimension);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_cosine_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *out_cosine)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);

    if (status != LECORE_OK) {
        return status;
    }
    if (a == NULL || b == NULL || out_cosine == NULL) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_f32_vector_is_finite(a, context->dimension) ||
         !lecore_f32_vector_is_finite(b, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    *out_cosine = lecore_f32_cosine_raw(a, b, context->dimension);
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_cosine_many_f32(
    lecore_context *context,
    const float *query,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *out_scores)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    float query_norm;
    float query_norm_squared = 0.0f;
    size_t rows_bytes;
    size_t row;
    uint32_t index;

    if (status != LECORE_OK) {
        return status;
    }
    if (query == NULL || rows == NULL || out_scores == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_f32_prepare_matrix_output(
        context,
        query,
        rows,
        row_count,
        row_stride,
        out_scores,
        &rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_f32_vector_is_finite(query, context->dimension) ||
         !lecore_f32_matrix_is_finite(
             rows, row_count, row_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    (void)rows_bytes;
    for (index = 0; index < context->dimension; ++index) {
        query_norm_squared += query[index] * query[index];
    }
    query_norm = sqrtf(query_norm_squared);
    for (row = 0; row < row_count; ++row) {
        out_scores[row] = lecore_f32_cosine_with_query_norm_squared(
            query,
            query_norm_squared,
            query_norm,
            rows + row * row_stride,
            context->dimension);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_involution_f32(
    lecore_context *context,
    const float *input,
    float *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    uint32_t index;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (input == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    vector_bytes = (size_t)context->dimension * sizeof(float);
    status = lecore_internal_check_vector_alias(
        input, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f32_vector_is_finite(input, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    if (input == output) {
        for (index = 1; index < context->dimension - index; ++index) {
            float temporary = output[index];
            output[index] = output[context->dimension - index];
            output[context->dimension - index] = temporary;
        }
    } else {
        output[0] = input[0];
        for (index = 1; index < context->dimension; ++index) {
            output[index] = input[context->dimension - index];
        }
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_permute_f32(
    lecore_context *context,
    const float *input,
    int64_t shift,
    float *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    float *target;
    int64_t reduced_shift;
    uint32_t normalized_shift;
    uint32_t index;
    uint32_t source;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (input == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    vector_bytes = (size_t)context->dimension * sizeof(float);
    status = lecore_internal_check_vector_alias(
        input, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f32_vector_is_finite(input, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    reduced_shift = shift % (int64_t)context->dimension;
    if (reduced_shift < 0) {
        reduced_shift += (int64_t)context->dimension;
    }
    normalized_shift = (uint32_t)reduced_shift;
    target = input == output ? (float *)context->scratch : output;

    for (index = 0; index < context->dimension; ++index) {
        source = index >= normalized_shift
            ? index - normalized_shift
            : context->dimension - (normalized_shift - index);
        target[index] = input[source];
    }
    if (target != output) {
        memcpy(output, target, vector_bytes);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_bundle_f32(
    lecore_context *context,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    size_t rows_bytes;
    size_t row;
    uint32_t index;
    float norm_squared = 0.0f;
    float norm;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (rows == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    vector_bytes = (size_t)context->dimension * sizeof(float);
    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        row_stride,
        sizeof(float),
        &rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (lecore_internal_ranges_overlap(
            rows, rows_bytes, output, vector_bytes)) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f32_matrix_is_finite(
            rows, row_count, row_stride, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    for (index = 0; index < context->dimension; ++index) {
        output[index] = 0.0f;
    }
    for (row = 0; row < row_count; ++row) {
        const float *current = rows + row * row_stride;
        for (index = 0; index < context->dimension; ++index) {
            output[index] += current[index];
        }
    }
    for (index = 0; index < context->dimension; ++index) {
        norm_squared += output[index] * output[index];
    }
    norm = sqrtf(norm_squared);
    if (norm > 0.0f) {
        for (index = 0; index < context->dimension; ++index) {
            output[index] /= norm;
        }
    }
    return LECORE_OK;
}
