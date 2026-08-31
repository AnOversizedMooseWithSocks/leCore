#include <lecore/lecore.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define FUZZ_MAX_DIMENSION ((size_t)64)
#define FUZZ_MAX_STRIDE ((size_t)72)
#define FUZZ_ROWS ((size_t)3)

static uint8_t fuzz_byte(const uint8_t *data, size_t size, size_t index)
{
    return size == 0 ? UINT8_C(0) : data[index % size];
}

static double fuzz_f64(const uint8_t *data, size_t size, size_t index)
{
    uint8_t byte = fuzz_byte(data, size, index);

    if ((byte & UINT8_C(31)) == UINT8_C(31)) {
        return NAN;
    }
    return ((double)(int)byte - 127.0) / 64.0;
}

static float fuzz_f32(const uint8_t *data, size_t size, size_t index)
{
    uint8_t byte = fuzz_byte(data, size, index);

    if ((byte & UINT8_C(31)) == UINT8_C(30)) {
        return INFINITY;
    }
    return ((float)(int)byte - 127.0F) / 64.0F;
}

static void exercise_f64(
    lecore_context *context,
    const uint8_t *data,
    size_t size,
    size_t dimension)
{
    double a[FUZZ_MAX_DIMENSION + 1];
    double b[FUZZ_MAX_DIMENSION + 1];
    double output[FUZZ_MAX_DIMENSION + 1];
    double rows[FUZZ_ROWS * FUZZ_MAX_STRIDE];
    double out_rows[FUZZ_ROWS * FUZZ_MAX_STRIDE];
    double scores[FUZZ_ROWS];
    double scalar = 0.0;
    size_t decision = 0;
    size_t stride = dimension +
        (size_t)(fuzz_byte(data, size, 7) % UINT8_C(9));
    size_t index;

    for (index = 0; index < FUZZ_MAX_DIMENSION + 1; ++index) {
        a[index] = fuzz_f64(data, size, index);
        b[index] = fuzz_f64(data, size, index + FUZZ_MAX_DIMENSION);
        output[index] = 0.0;
    }
    for (index = 0; index < FUZZ_ROWS * FUZZ_MAX_STRIDE; ++index) {
        rows[index] = fuzz_f64(data, size, index + 2 * FUZZ_MAX_DIMENSION);
        out_rows[index] = 0.0;
    }

    (void)lecore_validate_f64(a, dimension);
    (void)lecore_normalize_f64(context, a, output);
    (void)lecore_normalize_f64(context, a, a);
    (void)lecore_dot_f64(context, a, b, &scalar);
    (void)lecore_dot_many_f64(context, a, rows, FUZZ_ROWS, stride, scores);
    (void)lecore_cosine_f64(context, a, b, &scalar);
    (void)lecore_cosine_many_f64(context, a, rows, FUZZ_ROWS, stride, scores);
    (void)lecore_hrr_bind_f64(context, a, b, output);
    (void)lecore_hrr_bind_f64(context, a, b, a);
    (void)lecore_hrr_unbind_f64(context, a, b, output);
    (void)lecore_hrr_unbind_f64(context, a, b, b);
    (void)lecore_involution_f64(context, a, output);
    (void)lecore_involution_f64(context, a, a);
    (void)lecore_permute_f64(
        context, a, (int64_t)(int8_t)fuzz_byte(data, size, 11), output);
    (void)lecore_permute_f64(
        context, a, (int64_t)(int8_t)fuzz_byte(data, size, 12), a);
    (void)lecore_bundle_f64(context, rows, FUZZ_ROWS, stride, output);
    (void)lecore_cleanup_f64(
        context, a, rows, FUZZ_ROWS, stride, &decision, &scalar);
    (void)lecore_hrr_bind_batch_f64(
        context, rows, stride, rows + FUZZ_MAX_STRIDE, stride,
        (size_t)2, out_rows, FUZZ_MAX_STRIDE);
    (void)lecore_hrr_bind_fixed_f64(
        context, a, rows, FUZZ_ROWS, stride, out_rows, FUZZ_MAX_STRIDE);
    (void)lecore_hrr_unbind_all_f64(
        context, a, rows, FUZZ_ROWS, stride, out_rows, FUZZ_MAX_STRIDE);

    /* Invalid pointers, partial aliases, and size arithmetic must fail before
     * dereference or writes. Buffers include one padding element so a broken
     * partial-alias check remains visible to sanitizers without harness UB. */
    (void)lecore_normalize_f64(context, NULL, output);
    (void)lecore_normalize_f64(context, a, a + 1);
    (void)lecore_dot_many_f64(
        context, a, rows, SIZE_MAX, dimension, scores);
    (void)lecore_cosine_many_f64(context, a, rows, 0, stride, scores);
    (void)lecore_bundle_f64(context, rows, FUZZ_ROWS, dimension - 1, output);
    (void)lecore_hrr_bind_batch_f64(
        context, rows, stride, rows, stride, FUZZ_ROWS,
        rows + 1, stride);
}

static void exercise_f32(
    lecore_context *context,
    const uint8_t *data,
    size_t size,
    size_t dimension)
{
    float a[FUZZ_MAX_DIMENSION + 1];
    float b[FUZZ_MAX_DIMENSION + 1];
    float output[FUZZ_MAX_DIMENSION + 1];
    float rows[FUZZ_ROWS * FUZZ_MAX_STRIDE];
    float out_rows[FUZZ_ROWS * FUZZ_MAX_STRIDE];
    double query[FUZZ_MAX_DIMENSION + 1];
    double mixed_scores[FUZZ_ROWS];
    float scores[FUZZ_ROWS];
    float scalar = 0.0F;
    size_t decision = 0;
    size_t stride = dimension +
        (size_t)(fuzz_byte(data, size, 9) % UINT8_C(9));
    size_t index;

    for (index = 0; index < FUZZ_MAX_DIMENSION + 1; ++index) {
        a[index] = fuzz_f32(data, size, index);
        b[index] = fuzz_f32(data, size, index + FUZZ_MAX_DIMENSION);
        output[index] = 0.0F;
        query[index] = fuzz_f64(data, size, index + 3 * FUZZ_MAX_DIMENSION);
    }
    for (index = 0; index < FUZZ_ROWS * FUZZ_MAX_STRIDE; ++index) {
        rows[index] = fuzz_f32(data, size, index + 2 * FUZZ_MAX_DIMENSION);
        out_rows[index] = 0.0F;
    }

    (void)lecore_validate_f32(a, dimension);
    (void)lecore_normalize_f32(context, a, output);
    (void)lecore_normalize_f32(context, a, a);
    (void)lecore_dot_f32(context, a, b, &scalar);
    (void)lecore_dot_many_f32(context, a, rows, FUZZ_ROWS, stride, scores);
    (void)lecore_cosine_f32(context, a, b, &scalar);
    (void)lecore_cosine_many_f32(context, a, rows, FUZZ_ROWS, stride, scores);
    (void)lecore_hrr_bind_f32(context, a, b, output);
    (void)lecore_hrr_bind_f32(context, a, b, a);
    (void)lecore_hrr_unbind_f32(context, a, b, output);
    (void)lecore_hrr_unbind_f32(context, a, b, b);
    (void)lecore_involution_f32(context, a, output);
    (void)lecore_involution_f32(context, a, a);
    (void)lecore_permute_f32(
        context, a, (int64_t)(int8_t)fuzz_byte(data, size, 13), output);
    (void)lecore_permute_f32(
        context, a, (int64_t)(int8_t)fuzz_byte(data, size, 14), a);
    (void)lecore_bundle_f32(context, rows, FUZZ_ROWS, stride, output);
    (void)lecore_cleanup_f32(
        context, a, rows, FUZZ_ROWS, stride, &decision, &scalar);
    (void)lecore_hrr_bind_batch_f32(
        context, rows, stride, rows + FUZZ_MAX_STRIDE, stride,
        (size_t)2, out_rows, FUZZ_MAX_STRIDE);
    (void)lecore_hrr_bind_fixed_f32(
        context, a, rows, FUZZ_ROWS, stride, out_rows, FUZZ_MAX_STRIDE);
    (void)lecore_hrr_unbind_all_f32(
        context, a, rows, FUZZ_ROWS, stride, out_rows, FUZZ_MAX_STRIDE);
    (void)lecore_cosine_many_f64_f32(
        context, query, rows, FUZZ_ROWS, stride, mixed_scores);

    (void)lecore_normalize_f32(context, NULL, output);
    (void)lecore_normalize_f32(context, a, a + 1);
    (void)lecore_dot_many_f32(
        context, a, rows, SIZE_MAX, dimension, scores);
    (void)lecore_cosine_many_f32(context, a, rows, 0, stride, scores);
    (void)lecore_bundle_f32(context, rows, FUZZ_ROWS, dimension - 1, output);
    (void)lecore_hrr_bind_batch_f32(
        context, rows, stride, rows, stride, FUZZ_ROWS,
        rows + 1, stride);
    (void)lecore_cosine_many_f64_f32(
        context, query, rows, SIZE_MAX, dimension, mixed_scores);
}

static void exercise_invalid_configurations(
    const uint8_t *data,
    size_t size,
    size_t dimension)
{
    lecore_config_v0 config;
    lecore_context *context = NULL;

    lecore_config_init_v0(&config);
    config.dimension = (uint32_t)dimension;
    switch (fuzz_byte(data, size, 5) % UINT8_C(9)) {
    case 0:
        config.struct_size -= UINT32_C(1);
        break;
    case 1:
        config.abi_version = UINT32_MAX;
        break;
    case 2:
        config.profile = UINT32_MAX;
        break;
    case 3:
        config.backend = UINT32_MAX;
        break;
    case 4:
        config.validation = UINT32_MAX;
        break;
    case 5:
        config.flags = UINT32_C(1);
        break;
    case 6:
        config.dimension = UINT32_C(0);
        break;
    case 7:
        config.allocator.struct_size = UINT32_C(0);
        break;
    default:
        config.reserved[0] = UINT64_C(1);
        break;
    }
    if (lecore_context_create(&config, &context) == LECORE_OK ||
        context != NULL) {
        lecore_context_destroy(context);
        abort();
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    lecore_config_v0 config;
    lecore_context *context = NULL;
    size_t dimension;

    dimension = (size_t)(fuzz_byte(data, size, 0) %
        (uint8_t)FUZZ_MAX_DIMENSION) + (size_t)1;
    exercise_invalid_configurations(data, size, dimension);
    lecore_config_init_v0(&config);
    config.dimension = (uint32_t)dimension;
    config.profile = (fuzz_byte(data, size, 1) & UINT8_C(1)) != 0
        ? LECORE_PROFILE_HRR_F32_V1
        : LECORE_PROFILE_HRR_F64_V1;
    config.backend = (fuzz_byte(data, size, 2) & UINT8_C(1)) != 0
        ? LECORE_BACKEND_RADIX2
        : LECORE_BACKEND_DIRECT;
    config.validation = (fuzz_byte(data, size, 3) & UINT8_C(1)) != 0
        ? LECORE_VALIDATION_FINITE
        : LECORE_VALIDATION_SHAPE;

    if (lecore_context_create(&config, &context) == LECORE_OK) {
        (void)lecore_abi_version();
        (void)lecore_isa_version();
        (void)lecore_version_string();
        (void)lecore_capabilities();
        (void)lecore_status_string((lecore_status)fuzz_byte(data, size, 4));
        (void)lecore_context_dimension(context);
        (void)lecore_context_profile(context);
        (void)lecore_context_backend(context);
        (void)lecore_context_validation(context);
        (void)lecore_context_scalar_type(context);
        (void)lecore_context_scratch_bytes(context);
        if (config.profile == LECORE_PROFILE_HRR_F64_V1) {
            exercise_f64(context, data, size, dimension);
        } else {
            exercise_f32(context, data, size, dimension);
        }
    }
    lecore_context_destroy(context);
    return 0;
}
