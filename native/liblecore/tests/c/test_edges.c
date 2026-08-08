#include "test_common.h"

#include <stdint.h>
#include <string.h>

static int test_validation_modes(void)
{
    lecore_context *shape_context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_SHAPE);
    lecore_context *finite_context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_FINITE);
    const double finite[] = {1.0, 0.0, 0.0, 0.0};
    const double zero[] = {0.0, 0.0, 0.0, 0.0};
    const double with_nan[] = {NAN, 1.0, 0.0, 0.0};
    const double with_inf[] = {INFINITY, 1.0, 0.0, 0.0};
    const double candidates[] = {
        1.0, 0.0, 0.0, 0.0,
        NAN, 1.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0
    };
    double output[] = {42.0, 42.0, 42.0, 42.0};
    double score = 42.0;
    size_t index = SIZE_MAX;

    CHECK(shape_context != NULL && finite_context != NULL);
    CHECK_STATUS(lecore_validate_f64(NULL, 0), LECORE_OK);
    CHECK_STATUS(lecore_validate_f32(NULL, 0), LECORE_OK);
    CHECK_STATUS(lecore_validate_f64(NULL, 1), LECORE_EINVAL);
    CHECK_STATUS(lecore_validate_f64(finite, 4), LECORE_OK);
    CHECK_STATUS(lecore_validate_f64(with_nan, 4), LECORE_ENONFINITE);
    CHECK_STATUS(lecore_validate_f64(with_inf, 4), LECORE_ENONFINITE);

    CHECK_STATUS(lecore_hrr_bind_f64(
        finite_context, finite, with_nan, output), LECORE_ENONFINITE);
    CHECK(output[0] == 42.0 && output[3] == 42.0);
    CHECK_STATUS(lecore_hrr_bind_f64(
        shape_context, finite, with_nan, output), LECORE_OK);
    CHECK(isnan(output[0]) && isnan(output[1]) && isnan(output[2]) && isnan(output[3]));

    CHECK_STATUS(lecore_cosine_f64(shape_context, zero, with_nan, &score), LECORE_OK);
    CHECK(score == 0.0 && !signbit(score));
    score = 42.0;
    CHECK_STATUS(lecore_cosine_f64(finite_context, zero, with_nan, &score), LECORE_ENONFINITE);
    CHECK(score == 42.0);

    CHECK_STATUS(lecore_cleanup_f64(
        shape_context, finite, candidates, 3, 4, &index, &score), LECORE_OK);
    CHECK(index == 1);
    CHECK(isnan(score));

    lecore_context_destroy(finite_context);
    lecore_context_destroy(shape_context);
    return EXIT_SUCCESS;
}

static int test_errors_aliases_and_overflow(void)
{
    lecore_context *f64_context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_SHAPE);
    lecore_context *f32_context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F32_V1, LECORE_VALIDATION_SHAPE);
    const double vector[] = {1.0, 2.0, 3.0, 4.0};
    const double rows[] = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0
    };
    double overlap[9] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    double output[8] = {0.0};
    double score = 17.0;
    float f32_output[4];
    size_t index = 17;

    CHECK(f64_context != NULL && f32_context != NULL);
    CHECK_STATUS(lecore_normalize_f64(NULL, vector, output), LECORE_EINVAL);
    CHECK_STATUS(lecore_normalize_f64(f64_context, NULL, output), LECORE_EINVAL);
    CHECK_STATUS(lecore_normalize_f32(f64_context, (const float *)vector, f32_output), LECORE_EPROFILE);
    CHECK_STATUS(lecore_normalize_f64(f32_context, vector, output), LECORE_EPROFILE);
    CHECK_STATUS(lecore_normalize_f64(f64_context, overlap, overlap + 1), LECORE_EINVAL);
    CHECK_STATUS(lecore_hrr_bind_f64(f64_context, overlap, vector, overlap + 1), LECORE_EINVAL);

    CHECK_STATUS(lecore_dot_many_f64(f64_context, vector, rows, 0, 4, output), LECORE_EINVAL);
    CHECK_STATUS(lecore_dot_many_f64(f64_context, vector, rows, 2, 3, output), LECORE_EDIM);
    CHECK_STATUS(lecore_dot_many_f64(
        f64_context, vector, rows, 2, SIZE_MAX, output), LECORE_EOVERFLOW);
    CHECK_STATUS(lecore_bundle_f64(f64_context, rows, 0, 4, output), LECORE_EINVAL);
    CHECK_STATUS(lecore_bundle_f64(f64_context, rows, 2, 4, (double *)rows), LECORE_EINVAL);
    CHECK_STATUS(lecore_cleanup_f64(
        f64_context, vector, rows, 0, 4, &index, &score), LECORE_EINVAL);
    CHECK(index == 17 && score == 17.0);

    CHECK_STATUS(lecore_hrr_bind_batch_f64(
        f64_context, rows, 4, rows, 4, 2, output, 4), LECORE_OK);
    CHECK_STATUS(lecore_hrr_bind_batch_f64(
        f64_context, rows, 4, rows, 4, 2, (double *)rows, 4), LECORE_EINVAL);
    CHECK_STATUS(lecore_hrr_bind_batch_f64(
        f64_context, rows, 3, rows, 4, 2, output, 4), LECORE_EDIM);
    CHECK_STATUS(lecore_hrr_bind_batch_f64(
        f64_context, rows, 4, rows, 4, 0, output, 4), LECORE_EINVAL);

    lecore_context_destroy(f32_context);
    lecore_context_destroy(f64_context);
    return EXIT_SUCCESS;
}

int main(void)
{
    CHECK(test_validation_modes() == EXIT_SUCCESS);
    CHECK(test_errors_aliases_and_overflow() == EXIT_SUCCESS);
    return EXIT_SUCCESS;
}
