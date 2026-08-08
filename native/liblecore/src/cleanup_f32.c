#include "internal/lecore_internal.h"

#include <math.h>

static int lecore_cleanup_f32_vector_is_finite(
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

static float lecore_cleanup_f32_cosine(
    const float *query,
    const float *candidate,
    uint32_t dimension)
{
    float query_norm_squared = 0.0f;
    float candidate_norm_squared = 0.0f;
    float dot = 0.0f;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        query_norm_squared += query[index] * query[index];
        candidate_norm_squared += candidate[index] * candidate[index];
    }
    if (query_norm_squared == 0.0f || candidate_norm_squared == 0.0f) {
        return 0.0f;
    }
    for (index = 0; index < dimension; ++index) {
        dot += query[index] * candidate[index];
    }
    return dot /
        (sqrtf(query_norm_squared) * sqrtf(candidate_norm_squared));
}

lecore_status LECORE_CALL lecore_cleanup_f32(
    lecore_context *context,
    const float *query,
    const float *candidates,
    size_t candidate_count,
    size_t candidate_stride,
    size_t *out_index,
    float *out_score)
{
    lecore_status status = lecore_internal_check_context(
        context, LECORE_PROFILE_HRR_F32_V1);
    size_t candidates_bytes;
    size_t candidate;
    size_t best_index;
    float best_score;

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
        sizeof(float),
        &candidates_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    (void)candidates_bytes;

    if (context->validation == LECORE_VALIDATION_FINITE) {
        if (!lecore_cleanup_f32_vector_is_finite(
                query, context->dimension)) {
            return LECORE_ENONFINITE;
        }
        for (candidate = 0; candidate < candidate_count; ++candidate) {
            if (!lecore_cleanup_f32_vector_is_finite(
                    candidates + candidate * candidate_stride,
                    context->dimension)) {
                return LECORE_ENONFINITE;
            }
        }
    }

    best_index = 0;
    best_score = lecore_cleanup_f32_cosine(
        query, candidates, context->dimension);
    if (!isnan(best_score)) {
        for (candidate = 1; candidate < candidate_count; ++candidate) {
            float score = lecore_cleanup_f32_cosine(
                query,
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
