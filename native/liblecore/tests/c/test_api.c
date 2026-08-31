#include "test_common.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct test_allocation {
    void *pointer;
    size_t bytes;
    size_t alignment;
    int active;
} test_allocation;

typedef struct test_allocator_state {
    _Alignas(max_align_t) unsigned char storage[65536];
    size_t used;
    size_t allocation_count;
    size_t deallocation_count;
    size_t fail_at;
    int return_misaligned;
    int use_misalign_at;
    size_t misalign_at;
    int mismatch;
    test_allocation allocations[16];
} test_allocator_state;

static void *LECORE_CALL test_allocate(
    void *user,
    size_t bytes,
    size_t alignment)
{
    test_allocator_state *state = (test_allocator_state *)user;
    uintptr_t begin;
    uintptr_t aligned;
    size_t padding;
    void *pointer;

    if (bytes == 0 || alignment < _Alignof(max_align_t) ||
        (alignment & (alignment - 1)) != 0) {
        state->mismatch = 1;
        return NULL;
    }
    if (state->allocation_count == state->fail_at) {
        ++state->allocation_count;
        return NULL;
    }
    if (state->allocation_count >= 16) {
        state->mismatch = 1;
        return NULL;
    }

    begin = (uintptr_t)(state->storage + state->used);
    aligned = (begin + alignment - 1) & ~(uintptr_t)(alignment - 1);
    padding = (size_t)(aligned - begin);
    if (padding > sizeof(state->storage) - state->used ||
        bytes > sizeof(state->storage) - state->used - padding - 1) {
        return NULL;
    }
    {
        const int misalign = state->return_misaligned ||
            (state->use_misalign_at &&
             state->allocation_count == state->misalign_at);
        pointer = (void *)(aligned + (misalign ? 1U : 0U));
        state->used += padding + bytes + (misalign ? 1U : 0U);
    }
    state->allocations[state->allocation_count].pointer = pointer;
    state->allocations[state->allocation_count].bytes = bytes;
    state->allocations[state->allocation_count].alignment = alignment;
    state->allocations[state->allocation_count].active = 1;
    ++state->allocation_count;
    return pointer;
}

static void LECORE_CALL test_deallocate(
    void *user,
    void *pointer,
    size_t bytes,
    size_t alignment)
{
    test_allocator_state *state = (test_allocator_state *)user;
    size_t index;

    for (index = 0; index < state->allocation_count && index < 16; ++index) {
        test_allocation *allocation = &state->allocations[index];
        if (allocation->pointer == pointer && allocation->active) {
            if (allocation->bytes != bytes || allocation->alignment != alignment) {
                state->mismatch = 1;
            }
            allocation->active = 0;
            ++state->deallocation_count;
            return;
        }
    }
    state->mismatch = 1;
}

static void configure_allocator(
    lecore_config_v0 *config,
    test_allocator_state *state)
{
    config->allocator.user = state;
    config->allocator.allocate = test_allocate;
    config->allocator.deallocate = test_deallocate;
}

static int check_representative_f64_hot_path_allocations(
    lecore_context *context,
    test_allocator_state *state)
{
    const double a[8] = {1.0, 0.5, -0.25, 0.0, 0.75, -0.5, 0.25, 1.0};
    const double b[8] = {0.0, 1.0, 0.5, -0.5, 0.25, 0.0, -0.25, 0.75};
    const double rows[16] = {
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    };
    double output[16] = {0.0};
    double scores[2] = {0.0, 0.0};
    double score = 0.0;
    size_t index = 0;
    const size_t before = state->allocation_count;

    CHECK_STATUS(lecore_normalize_f64(context, a, output), LECORE_OK);
    CHECK_STATUS(lecore_dot_f64(context, a, b, &score), LECORE_OK);
    CHECK_STATUS(lecore_dot_many_f64(context, a, rows, 2, 8, scores), LECORE_OK);
    CHECK_STATUS(lecore_cosine_f64(context, a, b, &score), LECORE_OK);
    CHECK_STATUS(lecore_cosine_many_f64(context, a, rows, 2, 8, scores), LECORE_OK);
    CHECK_STATUS(lecore_hrr_bind_f64(context, a, b, output), LECORE_OK);
    CHECK_STATUS(lecore_hrr_unbind_f64(context, a, b, output), LECORE_OK);
    CHECK_STATUS(lecore_involution_f64(context, a, output), LECORE_OK);
    CHECK_STATUS(lecore_permute_f64(context, a, INT64_C(-3), output), LECORE_OK);
    CHECK_STATUS(lecore_bundle_f64(context, rows, 2, 8, output), LECORE_OK);
    CHECK_STATUS(lecore_cleanup_f64(
        context, a, rows, 2, 8, &index, &score), LECORE_OK);
    CHECK_STATUS(lecore_hrr_bind_batch_f64(
        context, rows, 8, rows, 8, 2, output, 8), LECORE_OK);
    CHECK_STATUS(lecore_hrr_bind_fixed_f64(
        context, a, rows, 2, 8, output, 8), LECORE_OK);
    CHECK_STATUS(lecore_hrr_unbind_all_f64(
        context, a, rows, 2, 8, output, 8), LECORE_OK);
    CHECK(state->allocation_count == before);
    CHECK(state->deallocation_count == 0);
    return EXIT_SUCCESS;
}

static int check_representative_f32_hot_path_allocations(
    lecore_context *context,
    test_allocator_state *state)
{
    const float a[8] = {1.0F, 0.5F, -0.25F, 0.0F, 0.75F, -0.5F, 0.25F, 1.0F};
    const float b[8] = {0.0F, 1.0F, 0.5F, -0.5F, 0.25F, 0.0F, -0.25F, 0.75F};
    const float rows[16] = {
        1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F
    };
    const double query[8] = {1.0, 0.5, -0.25, 0.0, 0.75, -0.5, 0.25, 1.0};
    float output[16] = {0.0F};
    float scores[2] = {0.0F, 0.0F};
    double mixed_scores[2] = {0.0, 0.0};
    float score = 0.0F;
    size_t index = 0;
    const size_t before = state->allocation_count;

    CHECK_STATUS(lecore_normalize_f32(context, a, output), LECORE_OK);
    CHECK_STATUS(lecore_dot_f32(context, a, b, &score), LECORE_OK);
    CHECK_STATUS(lecore_dot_many_f32(context, a, rows, 2, 8, scores), LECORE_OK);
    CHECK_STATUS(lecore_cosine_f32(context, a, b, &score), LECORE_OK);
    CHECK_STATUS(lecore_cosine_many_f32(context, a, rows, 2, 8, scores), LECORE_OK);
    CHECK_STATUS(lecore_cosine_many_f64_f32(
        context, query, rows, 2, 8, mixed_scores), LECORE_OK);
    CHECK_STATUS(lecore_hrr_bind_f32(context, a, b, output), LECORE_OK);
    CHECK_STATUS(lecore_hrr_unbind_f32(context, a, b, output), LECORE_OK);
    CHECK_STATUS(lecore_involution_f32(context, a, output), LECORE_OK);
    CHECK_STATUS(lecore_permute_f32(context, a, INT64_C(-3), output), LECORE_OK);
    CHECK_STATUS(lecore_bundle_f32(context, rows, 2, 8, output), LECORE_OK);
    CHECK_STATUS(lecore_cleanup_f32(
        context, a, rows, 2, 8, &index, &score), LECORE_OK);
    CHECK_STATUS(lecore_hrr_bind_batch_f32(
        context, rows, 8, rows, 8, 2, output, 8), LECORE_OK);
    CHECK_STATUS(lecore_hrr_bind_fixed_f32(
        context, a, rows, 2, 8, output, 8), LECORE_OK);
    CHECK_STATUS(lecore_hrr_unbind_all_f32(
        context, a, rows, 2, 8, output, 8), LECORE_OK);
    CHECK(state->allocation_count == before);
    CHECK(state->deallocation_count == 0);
    return EXIT_SUCCESS;
}

static int test_introspection_and_defaults(void)
{
    lecore_config_v0 config;
    uint64_t expected_caps = LECORE_CAP_HRR_F64 | LECORE_CAP_HRR_F32 |
        LECORE_CAP_DIRECT | LECORE_CAP_BATCH | LECORE_CAP_MIXED_F64_F32 |
        LECORE_CAP_FINITE_VALIDATION;
    size_t index;

    CHECK(lecore_abi_version() == UINT32_C(0));
    CHECK(lecore_abi_version() == LECORE_ABI_VERSION);
    CHECK(lecore_isa_version() == (uint32_t)LECORE_TEST_ISA_VERSION);
    CHECK(lecore_version_string() != NULL);
    CHECK(strcmp(lecore_version_string(), LECORE_TEST_VERSION) == 0);
#if LECORE_ENABLE_FORMAT
    expected_caps |= LECORE_CAP_FORMAT;
#endif
#if LECORE_ENABLE_RADIX2
    expected_caps |= LECORE_CAP_RADIX2;
#endif
    CHECK((lecore_capabilities() & expected_caps) == expected_caps);
#if !LECORE_ENABLE_RADIX2
    CHECK((lecore_capabilities() & LECORE_CAP_RADIX2) == 0);
#endif

    for (index = LECORE_OK; index <= LECORE_ECHECKSUM; ++index) {
        CHECK(lecore_status_string((lecore_status)index) != NULL);
        CHECK(lecore_status_string((lecore_status)index)[0] != '\0');
    }
    CHECK(strcmp(lecore_status_string(UINT32_C(999)), "unknown status") == 0);

    memset(&config, 0xa5, sizeof(config));
    lecore_config_init_v0(&config);
    CHECK(config.struct_size == sizeof(config));
    CHECK(config.abi_version == LECORE_ABI_VERSION);
    CHECK(config.profile == LECORE_PROFILE_HRR_F64_V1);
    CHECK(config.backend == LECORE_BACKEND_AUTO);
    CHECK(config.validation == LECORE_VALIDATION_SHAPE);
    CHECK(config.dimension == 0);
    CHECK(config.flags == 0);
    CHECK(config.allocator.struct_size == sizeof(config.allocator));
    CHECK(config.allocator.user == NULL);
    CHECK(config.allocator.allocate == NULL);
    CHECK(config.allocator.deallocate == NULL);
    for (index = 0; index < sizeof(config.reserved) / sizeof(config.reserved[0]); ++index) {
        CHECK(config.reserved[index] == 0);
    }
    lecore_config_init_v0(NULL);
    return EXIT_SUCCESS;
}

static int test_context_validation(void)
{
    lecore_config_v0 config;
    lecore_context *context = (lecore_context *)(uintptr_t)1;

    CHECK_STATUS(lecore_context_create(NULL, &context), LECORE_EINVAL);
    CHECK(context == NULL);
    CHECK_STATUS(lecore_context_create(NULL, NULL), LECORE_EINVAL);

    lecore_config_init_v0(&config);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EDIM);
    CHECK(context == NULL);

    config.dimension = 4;
    config.struct_size = 0;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
    config.struct_size = (uint32_t)sizeof(config);
    config.abi_version = UINT32_C(999);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
    config.abi_version = LECORE_ABI_VERSION;
    config.profile = UINT32_C(999);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EPROFILE);
    config.profile = LECORE_PROFILE_HRR_F64_V1;
    config.backend = LECORE_BACKEND_RADIX2;
#if LECORE_ENABLE_RADIX2
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_OK);
    CHECK(context != NULL);
    CHECK(lecore_context_backend(context) == LECORE_BACKEND_RADIX2);
    lecore_context_destroy(context);
    context = NULL;
#else
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EUNSUPPORTED);
#endif
    config.backend = UINT32_C(999);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EBACKEND);
    config.backend = LECORE_BACKEND_AUTO;
    config.validation = UINT32_C(999);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
    config.validation = LECORE_VALIDATION_SHAPE;
    config.flags = 1;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
    config.flags = 0;
    config.reserved[0] = 1;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
    config.reserved[0] = 0;
    config.allocator.allocate = test_allocate;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
    CHECK(context == NULL);
    return EXIT_SUCCESS;
}

static int test_context_and_allocator(void)
{
    lecore_config_v0 config;
    lecore_context *context = NULL;
    test_allocator_state state;

    memset(&state, 0, sizeof(state));
    state.fail_at = SIZE_MAX;
    lecore_config_init_v0(&config);
    config.dimension = 8;
    config.validation = LECORE_VALIDATION_FINITE;
    configure_allocator(&config, &state);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_OK);
    CHECK(context != NULL);
    CHECK(lecore_context_dimension(context) == 8);
    CHECK(lecore_context_profile(context) == LECORE_PROFILE_HRR_F64_V1);
    CHECK(lecore_context_backend(context) == LECORE_BACKEND_DIRECT);
    CHECK(lecore_context_validation(context) == LECORE_VALIDATION_FINITE);
    CHECK(lecore_context_scalar_type(context) == LECORE_SCALAR_F64);
    CHECK(lecore_context_scratch_bytes(context) == 8 * sizeof(double));
    CHECK(state.allocation_count > 0);
    CHECK(state.deallocation_count == 0);
    CHECK(check_representative_f64_hot_path_allocations(context, &state) == EXIT_SUCCESS);
    lecore_context_destroy(context);
    CHECK(state.deallocation_count == state.allocation_count);
    CHECK(state.mismatch == 0);

    memset(&state, 0, sizeof(state));
    state.fail_at = SIZE_MAX;
    lecore_config_init_v0(&config);
    config.dimension = 8;
    config.profile = LECORE_PROFILE_HRR_F32_V1;
    config.validation = LECORE_VALIDATION_FINITE;
    configure_allocator(&config, &state);
    context = NULL;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_OK);
    CHECK(context != NULL);
    CHECK(lecore_context_scalar_type(context) == LECORE_SCALAR_F32);
    CHECK(lecore_context_scratch_bytes(context) == 8 * sizeof(float));
    CHECK(check_representative_f32_hot_path_allocations(context, &state) == EXIT_SUCCESS);
    lecore_context_destroy(context);
    CHECK(state.deallocation_count == state.allocation_count);
    CHECK(state.mismatch == 0);

#if LECORE_ENABLE_RADIX2
    {
        size_t failure_index;

        for (failure_index = 0; failure_index < 4; ++failure_index) {
            memset(&state, 0, sizeof(state));
            state.fail_at = failure_index;
            lecore_config_init_v0(&config);
            config.dimension = 8;
            config.backend = LECORE_BACKEND_RADIX2;
            configure_allocator(&config, &state);
            context = (lecore_context *)(uintptr_t)1;
            CHECK_STATUS(lecore_context_create(&config, &context), LECORE_ENOMEM);
            CHECK(context == NULL);
            CHECK(state.deallocation_count == failure_index);
            CHECK(state.mismatch == 0);
        }

        for (failure_index = 0; failure_index < 4; ++failure_index) {
            memset(&state, 0, sizeof(state));
            state.fail_at = SIZE_MAX;
            state.use_misalign_at = 1;
            state.misalign_at = failure_index;
            lecore_config_init_v0(&config);
            config.dimension = 8;
            config.backend = LECORE_BACKEND_RADIX2;
            configure_allocator(&config, &state);
            context = (lecore_context *)(uintptr_t)1;
            CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
            CHECK(context == NULL);
            CHECK(state.deallocation_count == failure_index + 1);
            CHECK(state.mismatch == 0);
        }
    }

    memset(&state, 0, sizeof(state));
    state.fail_at = SIZE_MAX;
    lecore_config_init_v0(&config);
    config.dimension = 8;
    config.backend = LECORE_BACKEND_RADIX2;
    configure_allocator(&config, &state);
    context = NULL;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_OK);
    CHECK(context != NULL);
    CHECK(lecore_context_backend(context) == LECORE_BACKEND_RADIX2);
    CHECK(lecore_context_scratch_bytes(context) == 4 * 8 * sizeof(double));
    CHECK(state.allocation_count > 2);
    CHECK(check_representative_f64_hot_path_allocations(context, &state) == EXIT_SUCCESS);
    lecore_context_destroy(context);
    CHECK(state.deallocation_count == state.allocation_count);
    CHECK(state.mismatch == 0);

    memset(&state, 0, sizeof(state));
    state.fail_at = SIZE_MAX;
    lecore_config_init_v0(&config);
    config.dimension = 8;
    config.profile = LECORE_PROFILE_HRR_F32_V1;
    config.backend = LECORE_BACKEND_RADIX2;
    configure_allocator(&config, &state);
    context = NULL;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_OK);
    CHECK(context != NULL);
    CHECK(lecore_context_scratch_bytes(context) == 4 * 8 * sizeof(float));
    CHECK(check_representative_f32_hot_path_allocations(context, &state) == EXIT_SUCCESS);
    lecore_context_destroy(context);
    CHECK(state.deallocation_count == state.allocation_count);
    CHECK(state.mismatch == 0);
#endif

    CHECK(lecore_context_dimension(NULL) == 0);
    CHECK(lecore_context_profile(NULL) == 0);
    CHECK(lecore_context_backend(NULL) == 0);
    CHECK(lecore_context_validation(NULL) == 0);
    CHECK(lecore_context_scalar_type(NULL) == 0);
    CHECK(lecore_context_scratch_bytes(NULL) == 0);
    lecore_context_destroy(NULL);

    memset(&state, 0, sizeof(state));
    state.fail_at = 0;
    lecore_config_init_v0(&config);
    config.dimension = 4;
    configure_allocator(&config, &state);
    context = (lecore_context *)(uintptr_t)1;
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_ENOMEM);
    CHECK(context == NULL);

    memset(&state, 0, sizeof(state));
    state.fail_at = 1;
    lecore_config_init_v0(&config);
    config.dimension = 4;
    configure_allocator(&config, &state);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_ENOMEM);
    CHECK(context == NULL);
    CHECK(state.deallocation_count == 1);
    CHECK(state.mismatch == 0);

    memset(&state, 0, sizeof(state));
    state.fail_at = SIZE_MAX;
    state.return_misaligned = 1;
    lecore_config_init_v0(&config);
    config.dimension = 4;
    configure_allocator(&config, &state);
    CHECK_STATUS(lecore_context_create(&config, &context), LECORE_EINVAL);
    CHECK(context == NULL);
    CHECK(state.deallocation_count == 1);
    CHECK(state.mismatch == 0);
    return EXIT_SUCCESS;
}

int main(void)
{
    CHECK(test_introspection_and_defaults() == EXIT_SUCCESS);
    CHECK(test_context_validation() == EXIT_SUCCESS);
    CHECK(test_context_and_allocator() == EXIT_SUCCESS);
    return EXIT_SUCCESS;
}
