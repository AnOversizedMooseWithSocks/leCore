#include "test_common.h"

#include <string.h>

static int check_vector_f64(
    const double *actual,
    const double *expected,
    size_t count,
    double tolerance)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        if (!lecore_test_close_f64(actual[index], expected[index], tolerance)) {
            fprintf(stderr, "f64[%zu]: %.17g != %.17g\n",
                    index, actual[index], expected[index]);
            return 0;
        }
    }
    return 1;
}

static int check_vector_f32(
    const float *actual,
    const float *expected,
    size_t count,
    float tolerance)
{
    size_t index;
    for (index = 0; index < count; ++index) {
        if (!lecore_test_close_f32(actual[index], expected[index], tolerance)) {
            fprintf(stderr, "f32[%zu]: %.9g != %.9g\n",
                    index, (double)actual[index], (double)expected[index]);
            return 0;
        }
    }
    return 1;
}

static void reference_bind_f64(
    const double *a,
    const double *b,
    double *output,
    uint32_t dimension)
{
    uint32_t output_index;

    for (output_index = 0; output_index < dimension; ++output_index) {
        double sum = 0.0;
        uint32_t left_index;

        for (left_index = 0; left_index < dimension; ++left_index) {
            const uint32_t right_index = output_index >= left_index
                ? output_index - left_index
                : dimension - (left_index - output_index);
            const volatile double product =
                a[left_index] * b[right_index];
            sum += product;
        }
        output[output_index] = sum;
    }
}

static void reference_unbind_f64(
    const double *composite,
    const double *key,
    double *output,
    uint32_t dimension)
{
    uint32_t output_index;

    for (output_index = 0; output_index < dimension; ++output_index) {
        double sum = 0.0;
        uint32_t composite_index;

        for (composite_index = 0;
             composite_index < dimension;
             ++composite_index) {
            const uint32_t key_index = composite_index >= output_index
                ? composite_index - output_index
                : dimension - (output_index - composite_index);
            const volatile double product =
                composite[composite_index] * key[key_index];
            sum += product;
        }
        output[output_index] = sum;
    }
}

static void reference_bind_f32(
    const float *a,
    const float *b,
    float *output,
    uint32_t dimension)
{
    uint32_t output_index;

    for (output_index = 0; output_index < dimension; ++output_index) {
        float sum = 0.0f;
        uint32_t left_index;

        for (left_index = 0; left_index < dimension; ++left_index) {
            const uint32_t right_index = output_index >= left_index
                ? output_index - left_index
                : dimension - (left_index - output_index);
            const volatile float product = a[left_index] * b[right_index];
            sum += product;
        }
        output[output_index] = sum;
    }
}

static void reference_unbind_f32(
    const float *composite,
    const float *key,
    float *output,
    uint32_t dimension)
{
    uint32_t output_index;

    for (output_index = 0; output_index < dimension; ++output_index) {
        float sum = 0.0f;
        uint32_t composite_index;

        for (composite_index = 0;
             composite_index < dimension;
             ++composite_index) {
            const uint32_t key_index = composite_index >= output_index
                ? composite_index - output_index
                : dimension - (output_index - composite_index);
            const volatile float product =
                composite[composite_index] * key[key_index];
            sum += product;
        }
        output[output_index] = sum;
    }
}

static int test_direct_outer_product_order(void)
{
    enum { dimension = 32 };
    lecore_context *f64_context = lecore_test_context(
        dimension, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_SHAPE);
    lecore_context *f32_context = lecore_test_context(
        dimension, LECORE_PROFILE_HRR_F32_V1, LECORE_VALIDATION_SHAPE);
    double a_f64[dimension];
    double b_f64[dimension];
    double expected_f64[dimension];
    double actual_f64[dimension];
    float a_f32[dimension];
    float b_f32[dimension];
    float expected_f32[dimension];
    float actual_f32[dimension];
    uint32_t index;

    CHECK(f64_context != NULL && f32_context != NULL);
    for (index = 0; index < dimension; ++index) {
        const int a_integer = (int)((index * 7U + 3U) % 23U) - 11;
        const int b_integer = (int)((index * 11U + 5U) % 29U) - 14;

        a_f64[index] = (double)a_integer / 13.0;
        b_f64[index] = (double)b_integer / 17.0;
        a_f32[index] = (float)a_f64[index];
        b_f32[index] = (float)b_f64[index];
    }

    reference_bind_f64(a_f64, b_f64, expected_f64, dimension);
    CHECK_STATUS(lecore_hrr_bind_f64(
        f64_context, a_f64, b_f64, actual_f64), LECORE_OK);
    CHECK(memcmp(actual_f64, expected_f64, sizeof(actual_f64)) == 0);
    memcpy(actual_f64, a_f64, sizeof(actual_f64));
    CHECK_STATUS(lecore_hrr_bind_f64(
        f64_context, actual_f64, b_f64, actual_f64), LECORE_OK);
    CHECK(memcmp(actual_f64, expected_f64, sizeof(actual_f64)) == 0);

    reference_unbind_f64(a_f64, b_f64, expected_f64, dimension);
    CHECK_STATUS(lecore_hrr_unbind_f64(
        f64_context, a_f64, b_f64, actual_f64), LECORE_OK);
    CHECK(memcmp(actual_f64, expected_f64, sizeof(actual_f64)) == 0);

    reference_bind_f32(a_f32, b_f32, expected_f32, dimension);
    CHECK_STATUS(lecore_hrr_bind_f32(
        f32_context, a_f32, b_f32, actual_f32), LECORE_OK);
    CHECK(memcmp(actual_f32, expected_f32, sizeof(actual_f32)) == 0);
    reference_unbind_f32(a_f32, b_f32, expected_f32, dimension);
    CHECK_STATUS(lecore_hrr_unbind_f32(
        f32_context, a_f32, b_f32, actual_f32), LECORE_OK);
    CHECK(memcmp(actual_f32, expected_f32, sizeof(actual_f32)) == 0);

    lecore_context_destroy(f32_context);
    lecore_context_destroy(f64_context);
    return EXIT_SUCCESS;
}

static int test_f64_dense(void)
{
    lecore_context *context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_SHAPE);
    const double vector[] = {3.0, 4.0, 0.0, 0.0};
    const double expected_normalized[] = {0.6, 0.8, 0.0, 0.0};
    const double zero[] = {0.0, 0.0, 0.0, 0.0};
    const double cancellation[] = {1e16, 1.0, -1e16, 0.0};
    const double ones[] = {1.0, 1.0, 1.0, 1.0};
    const double rows[] = {
        1.0, 0.0, 0.0, 0.0, 99.0,
        0.0, 2.0, 0.0, 0.0, 99.0,
        1.0, 0.0, 0.0, 0.0, 99.0
    };
    const double expected_dot_many[] = {3.0, 8.0, 3.0};
    const double expected_cosine_many[] = {0.6, 0.8, 0.6};
    const double bundle_rows[] = {
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0
    };
    const double cancel_rows[] = {
        1.0, -2.0, 3.0, -4.0,
        -1.0, 2.0, -3.0, 4.0
    };
    const double expected_bundle[] = {
        0.7071067811865475244, 0.7071067811865475244, 0.0, 0.0
    };
    double output[4];
    double scores[3];
    double scalar = -1.0;
    size_t cleanup_index = SIZE_MAX;

    CHECK(context != NULL);
    CHECK_STATUS(lecore_normalize_f64(context, vector, output), LECORE_OK);
    CHECK(check_vector_f64(output, expected_normalized, 4, 1e-15));
    memcpy(output, vector, sizeof(output));
    CHECK_STATUS(lecore_normalize_f64(context, output, output), LECORE_OK);
    CHECK(check_vector_f64(output, expected_normalized, 4, 1e-15));
    CHECK_STATUS(lecore_normalize_f64(context, zero, output), LECORE_OK);
    CHECK(memcmp(output, zero, sizeof(output)) == 0);

    CHECK_STATUS(lecore_dot_f64(context, cancellation, ones, &scalar), LECORE_OK);
    CHECK(scalar == 0.0);
    CHECK_STATUS(lecore_dot_many_f64(context, vector, rows, 3, 5, scores), LECORE_OK);
    CHECK(check_vector_f64(scores, expected_dot_many, 3, 0.0));
    CHECK_STATUS(lecore_cosine_f64(context, vector, zero, &scalar), LECORE_OK);
    CHECK(scalar == 0.0 && !signbit(scalar));
    CHECK_STATUS(lecore_cosine_many_f64(context, vector, rows, 3, 5, scores), LECORE_OK);
    CHECK(check_vector_f64(scores, expected_cosine_many, 3, 1e-15));
    {
        size_t row;
        for (row = 0; row < 3; ++row) {
            CHECK_STATUS(lecore_cosine_f64(
                context, vector, rows + row * 5, &scalar), LECORE_OK);
            CHECK(memcmp(&scores[row], &scalar, sizeof(scalar)) == 0);
        }
    }

    CHECK_STATUS(lecore_bundle_f64(context, bundle_rows, 2, 4, output), LECORE_OK);
    CHECK(check_vector_f64(output, expected_bundle, 4, 1e-15));
    CHECK_STATUS(lecore_bundle_f64(context, cancel_rows, 2, 4, output), LECORE_OK);
    CHECK(memcmp(output, zero, sizeof(output)) == 0);

    CHECK_STATUS(lecore_cleanup_f64(
        context, rows, rows, 3, 5, &cleanup_index, &scalar), LECORE_OK);
    CHECK(cleanup_index == 0);
    CHECK(scalar == 1.0);

    lecore_context_destroy(context);
    return EXIT_SUCCESS;
}

static int test_f64_hrr_and_batches(void)
{
    lecore_context *context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_SHAPE);
    const double a[] = {1.0, 2.0, 0.0, -1.0};
    const double b[] = {2.0, 0.0, 1.0, 0.0};
    const double expected_bind[] = {2.0, 3.0, 1.0, 0.0};
    const double expected_involution[] = {1.0, -1.0, 0.0, 2.0};
    const double expected_unbind[] = {5.0, 6.0, 4.0, 3.0};
    const double expected_shift[] = {-1.0, 1.0, 2.0, 0.0};
    const double expected_negative_shift[] = {2.0, 0.0, -1.0, 1.0};
    const double a_rows[] = {
        1.0, 2.0, 0.0, -1.0, 91.0,
        1.0, 0.0, 0.0, 0.0, 92.0
    };
    const double b_rows[] = {
        2.0, 0.0, 1.0, 0.0, 81.0,
        0.0, 1.0, 0.0, 0.0, 82.0
    };
    double batch_output[12] = {0.0};
    double fixed_output[12] = {0.0};
    double unbind_output[12] = {0.0};
    double output[4];
    double expected[4];
    double alias[4];
    size_t row;

    CHECK(context != NULL);
    CHECK_STATUS(lecore_hrr_bind_f64(context, a, b, output), LECORE_OK);
    CHECK(check_vector_f64(output, expected_bind, 4, 0.0));
    memcpy(alias, a, sizeof(alias));
    CHECK_STATUS(lecore_hrr_bind_f64(context, alias, b, alias), LECORE_OK);
    CHECK(check_vector_f64(alias, expected_bind, 4, 0.0));

    CHECK_STATUS(lecore_involution_f64(context, a, output), LECORE_OK);
    CHECK(memcmp(output, expected_involution, sizeof(output)) == 0);
    memcpy(alias, a, sizeof(alias));
    CHECK_STATUS(lecore_involution_f64(context, alias, alias), LECORE_OK);
    CHECK(memcmp(alias, expected_involution, sizeof(alias)) == 0);

    CHECK_STATUS(lecore_hrr_unbind_f64(context, expected_bind, b, output), LECORE_OK);
    CHECK(check_vector_f64(output, expected_unbind, 4, 0.0));
    memcpy(alias, expected_bind, sizeof(alias));
    CHECK_STATUS(lecore_hrr_unbind_f64(context, alias, b, alias), LECORE_OK);
    CHECK(check_vector_f64(alias, expected_unbind, 4, 0.0));

    CHECK_STATUS(lecore_permute_f64(context, a, 1, output), LECORE_OK);
    CHECK(memcmp(output, expected_shift, sizeof(output)) == 0);
    memcpy(alias, a, sizeof(alias));
    CHECK_STATUS(lecore_permute_f64(context, alias, -1, alias), LECORE_OK);
    CHECK(memcmp(alias, expected_negative_shift, sizeof(alias)) == 0);

    CHECK_STATUS(lecore_hrr_bind_batch_f64(
        context, a_rows, 5, b_rows, 5, 2, batch_output, 6), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_bind_f64(
            context, a_rows + row * 5, b_rows + row * 5, expected), LECORE_OK);
        CHECK(check_vector_f64(batch_output + row * 6, expected, 4, 0.0));
    }

    CHECK_STATUS(lecore_hrr_bind_fixed_f64(
        context, a, b_rows, 2, 5, fixed_output, 6), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_bind_f64(context, a, b_rows + row * 5, expected), LECORE_OK);
        CHECK(check_vector_f64(fixed_output + row * 6, expected, 4, 0.0));
    }

    CHECK_STATUS(lecore_hrr_unbind_all_f64(
        context, expected_bind, b_rows, 2, 5, unbind_output, 6), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_unbind_f64(
            context, expected_bind, b_rows + row * 5, expected), LECORE_OK);
        CHECK(check_vector_f64(unbind_output + row * 6, expected, 4, 0.0));
    }

    lecore_context_destroy(context);
    return EXIT_SUCCESS;
}

static int test_f32_and_mixed(void)
{
    lecore_context *context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F32_V1, LECORE_VALIDATION_SHAPE);
    const float a[] = {1.0f, 2.0f, 0.0f, -1.0f};
    const float b[] = {2.0f, 0.0f, 1.0f, 0.0f};
    const float expected_bind[] = {2.0f, 3.0f, 1.0f, 0.0f};
    const float rows[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f
    };
    const double query[] = {1.0, 2.0, 0.0, 0.0};
    const double expected_mixed[] = {
        0.4472135954999579393, 0.8944271909999158786, 0.0
    };
    float output[4];
    float scores[3];
    float score = -1.0f;
    double mixed_scores[3];
    size_t index = SIZE_MAX;

    CHECK(context != NULL);
    CHECK(lecore_context_scalar_type(context) == LECORE_SCALAR_F32);
    CHECK(lecore_context_scratch_bytes(context) == 4 * sizeof(float));
    CHECK_STATUS(lecore_hrr_bind_f32(context, a, b, output), LECORE_OK);
    CHECK(check_vector_f32(output, expected_bind, 4, 0.0f));
    CHECK_STATUS(lecore_cosine_f32(context, a, a, &score), LECORE_OK);
    CHECK(lecore_test_close_f32(score, 1.0f, 1e-6f));
    CHECK_STATUS(lecore_cosine_many_f32(context, a, rows, 3, 4, scores), LECORE_OK);
    {
        size_t row;
        for (row = 0; row < 3; ++row) {
            CHECK_STATUS(lecore_cosine_f32(
                context, a, rows + row * 4, &score), LECORE_OK);
            CHECK(memcmp(&scores[row], &score, sizeof(score)) == 0);
        }
    }
    CHECK_STATUS(lecore_cleanup_f32(context, a, rows, 3, 4, &index, &score), LECORE_OK);
    CHECK(index == 1);
    CHECK(lecore_test_close_f32(score, 2.0f / sqrtf(6.0f), 1e-6f));
    CHECK_STATUS(lecore_cosine_many_f64_f32(
        context, query, rows, 3, 4, mixed_scores), LECORE_OK);
    CHECK(check_vector_f64(mixed_scores, expected_mixed, 3, 1e-15));

    lecore_context_destroy(context);
    return EXIT_SUCCESS;
}

#if LECORE_ENABLE_RADIX2
static lecore_context *test_radix2_context(
    uint32_t dimension,
    lecore_profile profile)
{
    lecore_config_v0 config;
    lecore_context *context = NULL;

    lecore_config_init_v0(&config);
    config.dimension = dimension;
    config.profile = profile;
    config.backend = LECORE_BACKEND_RADIX2;
    if (lecore_context_create(&config, &context) != LECORE_OK) {
        return NULL;
    }
    return context;
}

static int test_forced_radix2(void)
{
    const double a_f64[] = {1.0, 2.0, 0.0, -1.0};
    const double b_f64[] = {2.0, 0.0, 1.0, 0.0};
    const double expected_bind_f64[] = {2.0, 3.0, 1.0, 0.0};
    const double expected_unbind_f64[] = {5.0, 6.0, 4.0, 3.0};
    const float a_f32[] = {1.0f, 2.0f, 0.0f, -1.0f};
    const float b_f32[] = {2.0f, 0.0f, 1.0f, 0.0f};
    const float expected_bind_f32[] = {2.0f, 3.0f, 1.0f, 0.0f};
    const float a_rows_f32[] = {
        1.0f, 2.0f, 0.0f, -1.0f,
        1.0f, 0.0f, 0.0f, 0.0f
    };
    const float b_rows_f32[] = {
        2.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f
    };
    const double a_rows[] = {
        1.0, 2.0, 0.0, -1.0,
        1.0, 0.0, 0.0, 0.0
    };
    const double b_rows[] = {
        2.0, 0.0, 1.0, 0.0,
        0.0, 1.0, 0.0, 0.0
    };
    lecore_config_v0 config;
    lecore_context *context = NULL;
    lecore_context *direct_context;
    double output_f64[4];
    double alias_f64[4];
    double batch_output[8];
    double expected_row[4];
    float output_f32[4];
    float batch_output_f32[8];
    float expected_row_f32[4];
    size_t row;

    lecore_config_init_v0(&config);
    config.dimension = 3;
    config.backend = LECORE_BACKEND_RADIX2;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EUNSUPPORTED);
    CHECK(context == NULL);

    config.dimension = 4;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_OK);
    CHECK(lecore_context_backend(context) == LECORE_BACKEND_RADIX2);
    CHECK(lecore_context_scratch_bytes(context) == 4 * 4 * sizeof(double));
    CHECK_STATUS(lecore_hrr_bind_f64(context, a_f64, b_f64, output_f64), LECORE_OK);
    CHECK(check_vector_f64(output_f64, expected_bind_f64, 4, 1e-9));
    memcpy(alias_f64, a_f64, sizeof(alias_f64));
    CHECK_STATUS(lecore_hrr_bind_f64(context, alias_f64, b_f64, alias_f64), LECORE_OK);
    CHECK(check_vector_f64(alias_f64, expected_bind_f64, 4, 1e-9));
    CHECK_STATUS(lecore_hrr_unbind_f64(
        context, expected_bind_f64, b_f64, output_f64), LECORE_OK);
    CHECK(check_vector_f64(output_f64, expected_unbind_f64, 4, 1e-9));
    memcpy(alias_f64, expected_bind_f64, sizeof(alias_f64));
    CHECK_STATUS(lecore_hrr_unbind_f64(context, alias_f64, b_f64, alias_f64), LECORE_OK);
    CHECK(check_vector_f64(alias_f64, expected_unbind_f64, 4, 1e-9));

    direct_context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_SHAPE);
    CHECK(direct_context != NULL);
    CHECK(lecore_context_backend(direct_context) == LECORE_BACKEND_DIRECT);
    CHECK_STATUS(lecore_hrr_bind_batch_f64(
        context, a_rows, 4, b_rows, 4, 2, batch_output, 4), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_bind_f64(
            direct_context, a_rows + row * 4, b_rows + row * 4, expected_row), LECORE_OK);
        CHECK(check_vector_f64(batch_output + row * 4, expected_row, 4, 1e-9));
    }
    CHECK_STATUS(lecore_hrr_bind_fixed_f64(
        context, a_f64, b_rows, 2, 4, batch_output, 4), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_bind_f64(
            context, a_f64, b_rows + row * 4, expected_row), LECORE_OK);
        CHECK(memcmp(
            batch_output + row * 4, expected_row, sizeof(expected_row)) == 0);
        CHECK_STATUS(lecore_hrr_bind_f64(
            direct_context, a_f64, b_rows + row * 4, expected_row), LECORE_OK);
        CHECK(check_vector_f64(batch_output + row * 4, expected_row, 4, 1e-9));
    }
    CHECK_STATUS(lecore_hrr_unbind_all_f64(
        context, expected_bind_f64, b_rows, 2, 4, batch_output, 4), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_unbind_f64(
            context, expected_bind_f64, b_rows + row * 4, expected_row), LECORE_OK);
        CHECK(memcmp(
            batch_output + row * 4, expected_row, sizeof(expected_row)) == 0);
        CHECK_STATUS(lecore_hrr_unbind_f64(
            direct_context, expected_bind_f64, b_rows + row * 4, expected_row), LECORE_OK);
        CHECK(check_vector_f64(batch_output + row * 4, expected_row, 4, 1e-9));
    }
    lecore_context_destroy(direct_context);
    lecore_context_destroy(context);
    context = NULL;

    context = test_radix2_context(1, LECORE_PROFILE_HRR_F64_V1);
    CHECK(context != NULL);
    {
        const double left[] = {2.0};
        const double right[] = {3.0};
        double scalar_output[1];
        CHECK_STATUS(lecore_hrr_bind_f64(context, left, right, scalar_output), LECORE_OK);
        CHECK(lecore_test_close_f64(scalar_output[0], 6.0, 1e-12));
        CHECK_STATUS(lecore_hrr_unbind_f64(
            context, scalar_output, right, scalar_output), LECORE_OK);
        CHECK(lecore_test_close_f64(scalar_output[0], 18.0, 1e-12));
    }
    lecore_context_destroy(context);
    context = NULL;

    lecore_config_init_v0(&config);
    config.dimension = 4;
    config.profile = LECORE_PROFILE_HRR_F32_V1;
    config.backend = LECORE_BACKEND_RADIX2;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_OK);
    CHECK(lecore_context_scratch_bytes(context) == 4 * 4 * sizeof(float));
    CHECK_STATUS(lecore_hrr_bind_f32(context, a_f32, b_f32, output_f32), LECORE_OK);
    CHECK(check_vector_f32(output_f32, expected_bind_f32, 4, 1e-5f));
    direct_context = lecore_test_context(
        4, LECORE_PROFILE_HRR_F32_V1, LECORE_VALIDATION_SHAPE);
    CHECK(direct_context != NULL);
    CHECK_STATUS(lecore_hrr_bind_batch_f32(
        context, a_rows_f32, 4, b_rows_f32, 4, 2, batch_output_f32, 4), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_bind_f32(
            direct_context,
            a_rows_f32 + row * 4,
            b_rows_f32 + row * 4,
            expected_row_f32), LECORE_OK);
        CHECK(check_vector_f32(
            batch_output_f32 + row * 4, expected_row_f32, 4, 1e-5f));
    }
    CHECK_STATUS(lecore_hrr_bind_fixed_f32(
        context, a_f32, b_rows_f32, 2, 4, batch_output_f32, 4), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_bind_f32(
            context, a_f32, b_rows_f32 + row * 4, expected_row_f32), LECORE_OK);
        CHECK(memcmp(
            batch_output_f32 + row * 4,
            expected_row_f32,
            sizeof(expected_row_f32)) == 0);
        CHECK_STATUS(lecore_hrr_bind_f32(
            direct_context, a_f32, b_rows_f32 + row * 4, expected_row_f32), LECORE_OK);
        CHECK(check_vector_f32(
            batch_output_f32 + row * 4, expected_row_f32, 4, 1e-5f));
    }
    CHECK_STATUS(lecore_hrr_unbind_all_f32(
        context, expected_bind_f32, b_rows_f32, 2, 4, batch_output_f32, 4), LECORE_OK);
    for (row = 0; row < 2; ++row) {
        CHECK_STATUS(lecore_hrr_unbind_f32(
            context,
            expected_bind_f32,
            b_rows_f32 + row * 4,
            expected_row_f32), LECORE_OK);
        CHECK(memcmp(
            batch_output_f32 + row * 4,
            expected_row_f32,
            sizeof(expected_row_f32)) == 0);
        CHECK_STATUS(lecore_hrr_unbind_f32(
            direct_context,
            expected_bind_f32,
            b_rows_f32 + row * 4,
            expected_row_f32), LECORE_OK);
        CHECK(check_vector_f32(
            batch_output_f32 + row * 4, expected_row_f32, 4, 1e-5f));
    }
    lecore_context_destroy(direct_context);
    lecore_context_destroy(context);
    return EXIT_SUCCESS;
}

static int test_radix2_matches_direct_across_dimensions(void)
{
    static const uint32_t dimensions[] = {2, 4, 8, 16};
    double a_f64[16];
    double b_f64[16];
    double direct_f64[16];
    double radix_f64[16];
    float a_f32[16];
    float b_f32[16];
    float direct_f32[16];
    float radix_f32[16];
    size_t case_index;

    for (case_index = 0;
         case_index < sizeof(dimensions) / sizeof(dimensions[0]);
         ++case_index) {
        const uint32_t dimension = dimensions[case_index];
        lecore_context *direct_context;
        lecore_context *radix_context;
        uint32_t index;

        for (index = 0; index < dimension; ++index) {
            const int a_integer = (int)((index * 7U + 3U) % 11U) - 5;
            const int b_integer = (int)((index * 5U + 1U) % 13U) - 6;
            a_f64[index] = (double)a_integer / 7.0;
            b_f64[index] = (double)b_integer / 9.0;
            a_f32[index] = (float)a_f64[index];
            b_f32[index] = (float)b_f64[index];
        }

        direct_context = lecore_test_context(
            dimension, LECORE_PROFILE_HRR_F64_V1, LECORE_VALIDATION_SHAPE);
        radix_context = test_radix2_context(
            dimension, LECORE_PROFILE_HRR_F64_V1);
        CHECK(direct_context != NULL && radix_context != NULL);
        CHECK_STATUS(lecore_hrr_bind_f64(
            direct_context, a_f64, b_f64, direct_f64), LECORE_OK);
        CHECK_STATUS(lecore_hrr_bind_f64(
            radix_context, a_f64, b_f64, radix_f64), LECORE_OK);
        CHECK(check_vector_f64(radix_f64, direct_f64, dimension, 1e-9));
        CHECK_STATUS(lecore_hrr_unbind_f64(
            direct_context, direct_f64, b_f64, direct_f64), LECORE_OK);
        CHECK_STATUS(lecore_hrr_unbind_f64(
            radix_context, radix_f64, b_f64, radix_f64), LECORE_OK);
        CHECK(check_vector_f64(radix_f64, direct_f64, dimension, 1e-9));
        lecore_context_destroy(radix_context);
        lecore_context_destroy(direct_context);

        direct_context = lecore_test_context(
            dimension, LECORE_PROFILE_HRR_F32_V1, LECORE_VALIDATION_SHAPE);
        radix_context = test_radix2_context(
            dimension, LECORE_PROFILE_HRR_F32_V1);
        CHECK(direct_context != NULL && radix_context != NULL);
        CHECK_STATUS(lecore_hrr_bind_f32(
            direct_context, a_f32, b_f32, direct_f32), LECORE_OK);
        CHECK_STATUS(lecore_hrr_bind_f32(
            radix_context, a_f32, b_f32, radix_f32), LECORE_OK);
        CHECK(check_vector_f32(radix_f32, direct_f32, dimension, 1e-5f));
        CHECK_STATUS(lecore_hrr_unbind_f32(
            direct_context, direct_f32, b_f32, direct_f32), LECORE_OK);
        CHECK_STATUS(lecore_hrr_unbind_f32(
            radix_context, radix_f32, b_f32, radix_f32), LECORE_OK);
        CHECK(check_vector_f32(radix_f32, direct_f32, dimension, 1e-5f));
        lecore_context_destroy(radix_context);
        lecore_context_destroy(direct_context);
    }
    return EXIT_SUCCESS;
}
#endif

int main(void)
{
    CHECK(test_direct_outer_product_order() == EXIT_SUCCESS);
    CHECK(test_f64_dense() == EXIT_SUCCESS);
    CHECK(test_f64_hrr_and_batches() == EXIT_SUCCESS);
    CHECK(test_f32_and_mixed() == EXIT_SUCCESS);
#if LECORE_ENABLE_RADIX2
    CHECK(test_forced_radix2() == EXIT_SUCCESS);
    CHECK(test_radix2_matches_direct_across_dimensions() == EXIT_SUCCESS);
#endif
    return EXIT_SUCCESS;
}
