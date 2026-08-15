#include "internal/lecore_internal.h"

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifndef LECORE_ENABLE_FORMAT
#  define LECORE_ENABLE_FORMAT 0
#endif

#define LECORE_INTERNAL_TAU 6.2831853071795864769252867665590057683943387987502

static void *LECORE_CALL lecore_default_allocate(
    void *user,
    size_t bytes,
    size_t alignment)
{
    (void)user;
    (void)alignment;
    return malloc(bytes);
}

static void LECORE_CALL lecore_default_deallocate(
    void *user,
    void *pointer,
    size_t bytes,
    size_t alignment)
{
    (void)user;
    (void)bytes;
    (void)alignment;
    free(pointer);
}

static int lecore_is_power_of_two(uintmax_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

static lecore_status lecore_allocate_block(
    const lecore_allocator_v0 *allocator,
    size_t bytes,
    size_t alignment,
    void **out_pointer)
{
    void *pointer;

    if (bytes == 0 || out_pointer == NULL) {
        return LECORE_EINVAL;
    }
    *out_pointer = NULL;
    pointer = allocator->allocate(allocator->user, bytes, alignment);
    if (pointer == NULL) {
        return LECORE_ENOMEM;
    }
    if (((uintptr_t)pointer % alignment) != 0) {
        allocator->deallocate(
            allocator->user, pointer, bytes, alignment);
        return LECORE_EINVAL;
    }
    *out_pointer = pointer;
    return LECORE_OK;
}

#if LECORE_ENABLE_RADIX2
static void lecore_initialize_radix2_plan(lecore_context *context)
{
    uint32_t bit_count = 0;
    uint32_t value = context->dimension;
    uint32_t index;

    while (value > 1) {
        ++bit_count;
        value >>= 1;
    }

    for (index = 0; index < context->dimension; ++index) {
        uint32_t source = index;
        uint32_t reversed = 0;
        uint32_t bit;

        for (bit = 0; bit < bit_count; ++bit) {
            reversed = (reversed << 1) | (source & UINT32_C(1));
            source >>= 1;
        }
        context->radix2_bit_reversal[index] = reversed;
    }

    if (context->dimension == 1) {
        return;
    }
    if (context->profile == LECORE_PROFILE_HRR_F64_V1) {
        double *twiddles = (double *)context->radix2_twiddles;
        uint32_t twiddle;

        for (twiddle = 0; twiddle < context->dimension / 2; ++twiddle) {
            double angle = -LECORE_INTERNAL_TAU * (double)twiddle /
                (double)context->dimension;
            twiddles[(size_t)twiddle * 2] = cos(angle);
            twiddles[(size_t)twiddle * 2 + 1] = sin(angle);
        }
    } else {
        float *twiddles = (float *)context->radix2_twiddles;
        uint32_t twiddle;

        for (twiddle = 0; twiddle < context->dimension / 2; ++twiddle) {
            double angle = -LECORE_INTERNAL_TAU * (double)twiddle /
                (double)context->dimension;
            twiddles[(size_t)twiddle * 2] = (float)cos(angle);
            twiddles[(size_t)twiddle * 2 + 1] = (float)sin(angle);
        }
    }
}
#endif

static int lecore_reserved_is_zero(const lecore_config_v0 *config)
{
    size_t index;

    for (index = 0; index < sizeof(config->reserved) / sizeof(config->reserved[0]);
         ++index) {
        if (config->reserved[index] != 0) {
            return 0;
        }
    }
    return 1;
}

void LECORE_CALL lecore_config_init_v0(lecore_config_v0 *config)
{
    if (config == NULL) {
        return;
    }

    memset(config, 0, sizeof(*config));
    config->struct_size = (uint32_t)sizeof(*config);
    config->abi_version = LECORE_ABI_VERSION;
    config->profile = LECORE_PROFILE_HRR_F64_V1;
    config->backend = LECORE_BACKEND_AUTO;
    config->validation = LECORE_VALIDATION_SHAPE;
    config->allocator.struct_size = (uint32_t)sizeof(config->allocator);
}

lecore_status LECORE_CALL lecore_context_create(
    const lecore_config_v0 *config,
    lecore_context **out_context)
{
    lecore_allocator_v0 allocator;
    lecore_context *context;
    lecore_scalar_type scalar_type;
    lecore_backend backend;
    lecore_status status;
    size_t scalar_bytes;
    size_t scratch_bytes;
    size_t bit_reversal_bytes = 0;
    size_t twiddle_bytes = 0;
    const size_t alignment = _Alignof(max_align_t);
    void *context_memory = NULL;
    void *scratch = NULL;
    void *bit_reversal = NULL;
    void *twiddles = NULL;
    int has_allocate;
    int has_deallocate;

    if (out_context == NULL) {
        return LECORE_EINVAL;
    }
    *out_context = NULL;

    if (config == NULL) {
        return LECORE_EINVAL;
    }
    if (config->struct_size != sizeof(*config) ||
        config->abi_version != LECORE_ABI_VERSION ||
        config->allocator.struct_size != sizeof(config->allocator)) {
        return LECORE_EINVAL;
    }
    if (config->flags != 0 || !lecore_reserved_is_zero(config)) {
        return LECORE_EINVAL;
    }
    if (config->dimension == 0) {
        return LECORE_EDIM;
    }

    switch (config->profile) {
    case LECORE_PROFILE_HRR_F64_V1:
        scalar_type = LECORE_SCALAR_F64;
        scalar_bytes = sizeof(double);
        break;
    case LECORE_PROFILE_HRR_F32_V1:
        scalar_type = LECORE_SCALAR_F32;
        scalar_bytes = sizeof(float);
        break;
    default:
        return LECORE_EPROFILE;
    }

    if (config->validation != LECORE_VALIDATION_SHAPE &&
        config->validation != LECORE_VALIDATION_FINITE) {
        return LECORE_EINVAL;
    }
    if (config->backend == LECORE_BACKEND_AUTO ||
        config->backend == LECORE_BACKEND_DIRECT) {
        /* AUTO remains the correctness oracle until a crossover is earned. */
        backend = LECORE_BACKEND_DIRECT;
    } else if (config->backend == LECORE_BACKEND_RADIX2) {
#if LECORE_ENABLE_RADIX2
        if (!lecore_is_power_of_two((uintmax_t)config->dimension)) {
            return LECORE_EUNSUPPORTED;
        }
        backend = LECORE_BACKEND_RADIX2;
#else
        return LECORE_EUNSUPPORTED;
#endif
    } else {
        return LECORE_EBACKEND;
    }

    has_allocate = config->allocator.allocate != NULL;
    has_deallocate = config->allocator.deallocate != NULL;
    if (has_allocate != has_deallocate) {
        return LECORE_EINVAL;
    }

    allocator = config->allocator;
    if (!has_allocate) {
        allocator.user = NULL;
        allocator.allocate = lecore_default_allocate;
        allocator.deallocate = lecore_default_deallocate;
    }

    if (!lecore_is_power_of_two((uintmax_t)alignment)) {
        return LECORE_EOVERFLOW;
    }
    if (backend == LECORE_BACKEND_RADIX2) {
        const size_t scratch_scalars_per_element = 4;

        if ((uintmax_t)config->dimension >
            (uintmax_t)SIZE_MAX /
                ((uintmax_t)scratch_scalars_per_element * scalar_bytes) ||
            (uintmax_t)config->dimension >
                (uintmax_t)SIZE_MAX / sizeof(uint32_t) ||
            (config->dimension > 1 &&
             (uintmax_t)config->dimension >
                (uintmax_t)SIZE_MAX / scalar_bytes)) {
            return LECORE_EOVERFLOW;
        }
        scratch_bytes = (size_t)config->dimension *
            scratch_scalars_per_element * scalar_bytes;
        bit_reversal_bytes =
            (size_t)config->dimension * sizeof(uint32_t);
        if (config->dimension > 1) {
            twiddle_bytes = (size_t)config->dimension * scalar_bytes;
        }
    } else {
        if ((uintmax_t)config->dimension >
            (uintmax_t)SIZE_MAX / scalar_bytes) {
            return LECORE_EOVERFLOW;
        }
        scratch_bytes = (size_t)config->dimension * scalar_bytes;
    }

    status = lecore_allocate_block(
        &allocator, sizeof(*context), alignment, &context_memory);
    if (status != LECORE_OK) {
        return status;
    }
    context = (lecore_context *)context_memory;
    memset(context, 0, sizeof(*context));

    status = lecore_allocate_block(
        &allocator, scratch_bytes, alignment, &scratch);
    if (status != LECORE_OK) {
        goto fail;
    }
    if (backend == LECORE_BACKEND_RADIX2) {
        status = lecore_allocate_block(
            &allocator,
            bit_reversal_bytes,
            alignment,
            &bit_reversal);
        if (status != LECORE_OK) {
            goto fail;
        }
        if (twiddle_bytes != 0) {
            status = lecore_allocate_block(
                &allocator, twiddle_bytes, alignment, &twiddles);
            if (status != LECORE_OK) {
                goto fail;
            }
        }
    }

    context->dimension = config->dimension;
    context->profile = config->profile;
    context->backend = backend;
    context->validation = config->validation;
    context->scalar_type = scalar_type;
    context->allocator = allocator;
    context->scratch = scratch;
    context->scratch_bytes = scratch_bytes;
    context->radix2_bit_reversal = (uint32_t *)bit_reversal;
    context->radix2_bit_reversal_bytes = bit_reversal_bytes;
    context->radix2_twiddles = twiddles;
    context->radix2_twiddle_bytes = twiddle_bytes;
    context->context_bytes = sizeof(*context);
    context->allocation_alignment = alignment;
#if LECORE_ENABLE_RADIX2
    if (backend == LECORE_BACKEND_RADIX2) {
        lecore_initialize_radix2_plan(context);
    }
#endif
    context->magic = LECORE_INTERNAL_CONTEXT_MAGIC;

    *out_context = context;
    return LECORE_OK;

fail:
    if (twiddles != NULL) {
        allocator.deallocate(
            allocator.user, twiddles, twiddle_bytes, alignment);
    }
    if (bit_reversal != NULL) {
        allocator.deallocate(
            allocator.user,
            bit_reversal,
            bit_reversal_bytes,
            alignment);
    }
    if (scratch != NULL) {
        allocator.deallocate(
            allocator.user, scratch, scratch_bytes, alignment);
    }
    allocator.deallocate(
        allocator.user, context, sizeof(*context), alignment);
    return status;
}

void LECORE_CALL lecore_context_destroy(lecore_context *context)
{
    lecore_allocator_v0 allocator;
    void *scratch;
    size_t scratch_bytes;
    void *bit_reversal;
    size_t bit_reversal_bytes;
    void *twiddles;
    size_t twiddle_bytes;
    size_t context_bytes;
    size_t alignment;

    if (context == NULL || context->magic != LECORE_INTERNAL_CONTEXT_MAGIC) {
        return;
    }

    allocator = context->allocator;
    scratch = context->scratch;
    scratch_bytes = context->scratch_bytes;
    bit_reversal = context->radix2_bit_reversal;
    bit_reversal_bytes = context->radix2_bit_reversal_bytes;
    twiddles = context->radix2_twiddles;
    twiddle_bytes = context->radix2_twiddle_bytes;
    context_bytes = context->context_bytes;
    alignment = context->allocation_alignment;
    context->magic = 0;

    if (twiddles != NULL) {
        allocator.deallocate(
            allocator.user, twiddles, twiddle_bytes, alignment);
    }
    if (bit_reversal != NULL) {
        allocator.deallocate(
            allocator.user,
            bit_reversal,
            bit_reversal_bytes,
            alignment);
    }
    allocator.deallocate(
        allocator.user, scratch, scratch_bytes, alignment);
    allocator.deallocate(allocator.user, context, context_bytes, alignment);
}

uint32_t LECORE_CALL lecore_context_dimension(const lecore_context *context)
{
    return context != NULL && context->magic == LECORE_INTERNAL_CONTEXT_MAGIC
        ? context->dimension
        : UINT32_C(0);
}

lecore_profile LECORE_CALL lecore_context_profile(const lecore_context *context)
{
    return context != NULL && context->magic == LECORE_INTERNAL_CONTEXT_MAGIC
        ? context->profile
        : UINT32_C(0);
}

lecore_backend LECORE_CALL lecore_context_backend(const lecore_context *context)
{
    return context != NULL && context->magic == LECORE_INTERNAL_CONTEXT_MAGIC
        ? context->backend
        : UINT32_C(0);
}

lecore_validation LECORE_CALL lecore_context_validation(
    const lecore_context *context)
{
    return context != NULL && context->magic == LECORE_INTERNAL_CONTEXT_MAGIC
        ? context->validation
        : UINT32_C(0);
}

lecore_scalar_type LECORE_CALL lecore_context_scalar_type(
    const lecore_context *context)
{
    return context != NULL && context->magic == LECORE_INTERNAL_CONTEXT_MAGIC
        ? context->scalar_type
        : UINT32_C(0);
}

size_t LECORE_CALL lecore_context_scratch_bytes(const lecore_context *context)
{
    return context != NULL && context->magic == LECORE_INTERNAL_CONTEXT_MAGIC
        ? context->scratch_bytes
        : 0;
}

uint64_t LECORE_CALL lecore_capabilities(void)
{
    uint64_t capabilities =
        LECORE_CAP_HRR_F64 |
        LECORE_CAP_HRR_F32 |
        LECORE_CAP_DIRECT |
        LECORE_CAP_BATCH |
        LECORE_CAP_MIXED_F64_F32 |
        LECORE_CAP_FINITE_VALIDATION;

#if LECORE_ENABLE_FORMAT
    capabilities |= LECORE_CAP_FORMAT;
#endif
#if LECORE_ENABLE_RADIX2
    capabilities |= LECORE_CAP_RADIX2;
#endif
    return capabilities;
}
