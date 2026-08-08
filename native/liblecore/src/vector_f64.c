#include "internal/lecore_internal.h"

#include <math.h>
#include <string.h>

static int lecore_f64_vector_is_finite(const double *values, uint32_t dimension)
{
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        if (!isfinite(values[index])) {
            return 0;
        }
    }
    return 1;
}

static int lecore_f64_matrix_is_finite(
    const double *rows,
    size_t row_count,
    size_t row_stride,
    uint32_t dimension)
{
    size_t row;

    for (row = 0; row < row_count; ++row) {
        if (!lecore_f64_vector_is_finite(
                rows + row * row_stride, dimension)) {
            return 0;
        }
    }
    return 1;
}

static double lecore_f64_dot_raw(
    const double *a,
    const double *b,
    uint32_t dimension)
{
    double result = 0.0;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        result += a[index] * b[index];
    }
    return result;
}

static double lecore_f64_cosine_raw(
    const double *a,
    const double *b,
    uint32_t dimension)
{
    double dot;
    double norm_a_squared = 0.0;
    double norm_b_squared = 0.0;
    double norm_a;
    double norm_b;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        norm_a_squared += a[index] * a[index];
        norm_b_squared += b[index] * b[index];
    }

    /* The exact zero branch precedes NaN/Inf propagation by contract. */
    if (norm_a_squared == 0.0 || norm_b_squared == 0.0) {
        return 0.0;
    }
    norm_a = sqrt(norm_a_squared);
    norm_b = sqrt(norm_b_squared);
    dot = lecore_f64_dot_raw(a, b, dimension);
    return dot / (norm_a * norm_b);
}

static lecore_status lecore_f64_prepare_matrix_output(
    const lecore_context *context,
    const double *query,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    const double *out_values,
    size_t *out_rows_bytes)
{
    lecore_status status;
    size_t output_bytes;
    const size_t vector_bytes = (size_t)context->dimension * sizeof(double);

    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        row_stride,
        sizeof(double),
        out_rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (row_count > SIZE_MAX / sizeof(double)) {
        return LECORE_EOVERFLOW;
    }
    output_bytes = row_count * sizeof(double);
    if (lecore_internal_ranges_overlap(
            query, vector_bytes, out_values, output_bytes) ||
        lecore_internal_ranges_overlap(
            rows, *out_rows_bytes, out_values, output_bytes)) {
        return LECORE_EINVAL;
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_validate_f64(
    const double *values,
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

lecore_status LECORE_CALL lecore_normalize_f64(
    lecore_context *context,
    const double *input,
    double *output)
{
    lecore_status status;
    double norm_squared = 0.0;
    double norm;
    uint32_t index;
    size_t vector_bytes;

    status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    if (status != LECORE_OK) {
        return status;
    }
    if (input == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    vector_bytes = (size_t)context->dimension * sizeof(double);
    status = lecore_internal_check_vector_alias(
        input, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f64_vector_is_finite(input, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    for (index = 0; index < context->dimension; ++index) {
        norm_squared += input[index] * input[index];
    }
    norm = sqrt(norm_squared);
    if (norm > 0.0) {
        for (index = 0; index < context->dimension; ++index) {
            output[index] = input[index] / norm;
        }
    } else if (output != input) {
        memcpy(output, input, vector_bytes);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_dot_f64(
    lecore_context *context,
    const double *a,
    const double *b,
    double *out_dot)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);

    if (status != LECORE_OK) {
        return status;
    }
    if (a == NULL || b == NULL || out_dot == NULL) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_f64_vector_is_finite(a, context->dimension) ||
         !lecore_f64_vector_is_finite(b, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    *out_dot = lecore_f64_dot_raw(a, b, context->dimension);
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_dot_many_f64(
    lecore_context *context,
    const double *query,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *out_scores)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    size_t rows_bytes;
    size_t row;

    if (status != LECORE_OK) {
        return status;
    }
    if (query == NULL || rows == NULL || out_scores == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_f64_prepare_matrix_output(
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
        (!lecore_f64_vector_is_finite(query, context->dimension) ||
         !lecore_f64_matrix_is_finite(
             rows, row_count, row_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    (void)rows_bytes;
    for (row = 0; row < row_count; ++row) {
        out_scores[row] = lecore_f64_dot_raw(
            query, rows + row * row_stride, context->dimension);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_cosine_f64(
    lecore_context *context,
    const double *a,
    const double *b,
    double *out_cosine)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);

    if (status != LECORE_OK) {
        return status;
    }
    if (a == NULL || b == NULL || out_cosine == NULL) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        (!lecore_f64_vector_is_finite(a, context->dimension) ||
         !lecore_f64_vector_is_finite(b, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    *out_cosine = lecore_f64_cosine_raw(a, b, context->dimension);
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_cosine_many_f64(
    lecore_context *context,
    const double *query,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *out_scores)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    size_t rows_bytes;
    size_t row;

    if (status != LECORE_OK) {
        return status;
    }
    if (query == NULL || rows == NULL || out_scores == NULL) {
        return LECORE_EINVAL;
    }
    status = lecore_f64_prepare_matrix_output(
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
        (!lecore_f64_vector_is_finite(query, context->dimension) ||
         !lecore_f64_matrix_is_finite(
             rows, row_count, row_stride, context->dimension))) {
        return LECORE_ENONFINITE;
    }
    (void)rows_bytes;
    for (row = 0; row < row_count; ++row) {
        out_scores[row] = lecore_f64_cosine_raw(
            query, rows + row * row_stride, context->dimension);
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_involution_f64(
    lecore_context *context,
    const double *input,
    double *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    uint32_t index;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (input == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    vector_bytes = (size_t)context->dimension * sizeof(double);
    status = lecore_internal_check_vector_alias(
        input, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f64_vector_is_finite(input, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    if (input == output) {
        for (index = 1; index < context->dimension - index; ++index) {
            double temporary = output[index];
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

lecore_status LECORE_CALL lecore_permute_f64(
    lecore_context *context,
    const double *input,
    int64_t shift,
    double *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    double *target;
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
    vector_bytes = (size_t)context->dimension * sizeof(double);
    status = lecore_internal_check_vector_alias(
        input, output, vector_bytes, 1);
    if (status != LECORE_OK) {
        return status;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f64_vector_is_finite(input, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    reduced_shift = shift % (int64_t)context->dimension;
    if (reduced_shift < 0) {
        reduced_shift += (int64_t)context->dimension;
    }
    normalized_shift = (uint32_t)reduced_shift;
    target = input == output ? (double *)context->scratch : output;

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

lecore_status LECORE_CALL lecore_bundle_f64(
    lecore_context *context,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *output)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    size_t rows_bytes;
    size_t row;
    uint32_t index;
    double norm_squared = 0.0;
    double norm;
    size_t vector_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (rows == NULL || output == NULL) {
        return LECORE_EINVAL;
    }
    vector_bytes = (size_t)context->dimension * sizeof(double);
    status = lecore_internal_matrix_span(
        context->dimension,
        row_count,
        row_stride,
        sizeof(double),
        &rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (lecore_internal_ranges_overlap(
            rows, rows_bytes, output, vector_bytes)) {
        return LECORE_EINVAL;
    }
    if (context->validation == LECORE_VALIDATION_FINITE &&
        !lecore_f64_matrix_is_finite(
            rows, row_count, row_stride, context->dimension)) {
        return LECORE_ENONFINITE;
    }

    for (index = 0; index < context->dimension; ++index) {
        output[index] = 0.0;
    }
    for (row = 0; row < row_count; ++row) {
        const double *current = rows + row * row_stride;
        for (index = 0; index < context->dimension; ++index) {
            output[index] += current[index];
        }
    }
    for (index = 0; index < context->dimension; ++index) {
        norm_squared += output[index] * output[index];
    }
    norm = sqrt(norm_squared);
    if (norm > 0.0) {
        for (index = 0; index < context->dimension; ++index) {
            output[index] /= norm;
        }
    }
    return LECORE_OK;
}

lecore_status LECORE_CALL lecore_cosine_many_f64_f32(
    lecore_context *f32_context,
    const double *query,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    double *out_scores)
{
    lecore_status status = lecore_internal_check_context(
        f32_context, LECORE_PROFILE_HRR_F32_V1);
    size_t rows_bytes;
    size_t output_bytes;
    size_t row;
    uint32_t index;
    double query_norm_squared = 0.0;
    size_t query_bytes;

    if (status != LECORE_OK) {
        return status;
    }
    if (query == NULL || rows == NULL || out_scores == NULL) {
        return LECORE_EINVAL;
    }
    if ((uintmax_t)f32_context->dimension >
        (uintmax_t)SIZE_MAX / (uintmax_t)sizeof(double)) {
        return LECORE_EOVERFLOW;
    }
    query_bytes = (size_t)f32_context->dimension * sizeof(double);
    status = lecore_internal_matrix_span(
        f32_context->dimension,
        row_count,
        row_stride,
        sizeof(float),
        &rows_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    if (row_count > SIZE_MAX / sizeof(double)) {
        return LECORE_EOVERFLOW;
    }
    output_bytes = row_count * sizeof(double);
    if (lecore_internal_ranges_overlap(
            query, query_bytes, out_scores, output_bytes) ||
        lecore_internal_ranges_overlap(
            rows, rows_bytes, out_scores, output_bytes)) {
        return LECORE_EINVAL;
    }
    if (f32_context->validation == LECORE_VALIDATION_FINITE) {
        if (!lecore_f64_vector_is_finite(query, f32_context->dimension)) {
            return LECORE_ENONFINITE;
        }
        for (row = 0; row < row_count; ++row) {
            const float *current = rows + row * row_stride;
            for (index = 0; index < f32_context->dimension; ++index) {
                if (!isfinite(current[index])) {
                    return LECORE_ENONFINITE;
                }
            }
        }
    }

    for (index = 0; index < f32_context->dimension; ++index) {
        query_norm_squared += query[index] * query[index];
    }
    for (row = 0; row < row_count; ++row) {
        const float *current = rows + row * row_stride;
        double dot;
        double row_norm_squared = 0.0;

        for (index = 0; index < f32_context->dimension; ++index) {
            const double component = (double)current[index];
            row_norm_squared += component * component;
        }
        if (query_norm_squared == 0.0 || row_norm_squared == 0.0) {
            out_scores[row] = 0.0;
        } else {
            dot = 0.0;
            for (index = 0; index < f32_context->dimension; ++index) {
                dot += query[index] * (double)current[index];
            }
            out_scores[row] = dot /
                (sqrt(query_norm_squared) * sqrt(row_norm_squared));
        }
    }
    return LECORE_OK;
}
