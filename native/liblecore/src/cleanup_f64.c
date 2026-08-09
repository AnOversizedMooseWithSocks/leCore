#include "internal/lecore_internal.h"

#include <math.h>

static int lecore_cleanup_f64_vector_is_finite(
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

static double lecore_cleanup_f64_cosine(
    const double *query,
    double query_norm_squared,
    double query_norm,
    const double *candidate,
    uint32_t dimension)
{
    double candidate_norm_squared = 0.0;
    double candidate_norm;
    double dot = 0.0;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        candidate_norm_squared += candidate[index] * candidate[index];
    }
    if (query_norm_squared == 0.0 || candidate_norm_squared == 0.0) {
        return 0.0;
    }
    for (index = 0; index < dimension; ++index) {
        dot += query[index] * candidate[index];
    }
    candidate_norm = sqrt(candidate_norm_squared);
    return dot / (query_norm * candidate_norm);
}

lecore_status LECORE_CALL lecore_cleanup_f64(
    lecore_context *context,
    const double *query,
    const double *candidates,
    size_t candidate_count,
    size_t candidate_stride,
    size_t *out_index,
    double *out_score)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F64_V1);
    size_t candidates_bytes;
    size_t candidate;
    size_t best_index;
    double best_score;
    double query_norm;
    double query_norm_squared = 0.0;
    uint32_t index;

    if (status != LECORE_OK) {
        return status;
    }
    if (query == NULL || candidates == NULL ||
        out_index == NULL || out_score == NULL) {
        return LECORE_EINVAL;
    }
    if (lecore_internal_ranges_overlap(
            out_index, sizeof(*out_index), out_score, sizeof(*out_score))) {
        return LECORE_EINVAL;
    }
    status = lecore_internal_matrix_span(
        context->dimension,
        candidate_count,
        candidate_stride,
        sizeof(double),
        &candidates_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    (void)candidates_bytes;

    if (context->validation == LECORE_VALIDATION_FINITE) {
        if (!lecore_cleanup_f64_vector_is_finite(
                query, context->dimension)) {
            return LECORE_ENONFINITE;
        }
        for (candidate = 0; candidate < candidate_count; ++candidate) {
            if (!lecore_cleanup_f64_vector_is_finite(
                    candidates + candidate * candidate_stride,
                    context->dimension)) {
                return LECORE_ENONFINITE;
            }
        }
    }

    for (index = 0; index < context->dimension; ++index) {
        query_norm_squared += query[index] * query[index];
    }
    query_norm = sqrt(query_norm_squared);
    best_index = 0;
    best_score = lecore_cleanup_f64_cosine(
        query,
        query_norm_squared,
        query_norm,
        candidates,
        context->dimension);

    /* NumPy argmax treats the first NaN as the winning stable index. */
    if (!isnan(best_score)) {
        for (candidate = 1; candidate < candidate_count; ++candidate) {
            double score = lecore_cleanup_f64_cosine(
                query,
                query_norm_squared,
                query_norm,
                candidates + candidate * candidate_stride,
                context->dimension);
            if (isnan(score)) {
                best_index = candidate;
                best_score = score;
                break;
            }
            if (score > best_score) {
                best_index = candidate;
                best_score = score;
            }
        }
    }

    *out_index = best_index;
    *out_score = best_score;
    return LECORE_OK;
}
