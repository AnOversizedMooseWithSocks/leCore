#ifndef LECORE_INTERNAL_H
#define LECORE_INTERNAL_H

#include <lecore/lecore.h>

#include <float.h>
#include <limits.h>
#include <stdint.h>

#if CHAR_BIT != 8
#  error "liblecore profiles require 8-bit bytes"
#endif

#if FLT_RADIX != 2
#  error "liblecore profiles require radix-2 floating point"
#endif

#if FLT_MANT_DIG != 24 || FLT_MIN_EXP != -125 || FLT_MAX_EXP != 128
#  error "liblecore HRR_F32_V1 requires IEEE-compatible binary32"
#endif

#if DBL_MANT_DIG != 53 || DBL_MIN_EXP != -1021 || DBL_MAX_EXP != 1024
#  error "liblecore HRR_F64_V1 requires IEEE-compatible binary64"
#endif

#if FLT_EVAL_METHOD != 0
#  error "liblecore profiles require operations evaluated in their declared precision"
#endif

#ifndef LECORE_ENABLE_RADIX2
#  define LECORE_ENABLE_RADIX2 1
#endif

#if defined(__cplusplus)
static_assert(sizeof(float) == 4, "liblecore requires 32-bit float");
static_assert(sizeof(double) == 8, "liblecore requires 64-bit double");
#else
_Static_assert(sizeof(float) == 4, "liblecore requires 32-bit float");
_Static_assert(sizeof(double) == 8, "liblecore requires 64-bit double");
#endif

#define LECORE_INTERNAL_CONTEXT_MAGIC UINT64_C(0x6c65636f72653030)

/* Internal cross-translation-unit calls become file-local in the generated
 * single-translation-unit amalgamation, so standalone objects expose only the
 * documented ABI. */
#if defined(LECORE_AMALGAMATION)
#  define LECORE_INTERNAL_API static
#else
#  define LECORE_INTERNAL_API
#endif

struct lecore_context {
    uint64_t magic;
    uint32_t dimension;
    lecore_profile profile;
    lecore_backend backend;
    lecore_validation validation;
    lecore_scalar_type scalar_type;
    lecore_allocator_v0 allocator;
    void *scratch;
    size_t scratch_bytes;
    uint32_t *radix2_bit_reversal;
    size_t radix2_bit_reversal_bytes;
    void *radix2_twiddles;
    size_t radix2_twiddle_bytes;
    size_t context_bytes;
    size_t allocation_alignment;
};

static inline lecore_status lecore_internal_check_context(
    const lecore_context *context,
    lecore_profile expected_profile)
{
    if (context == NULL || context->magic != LECORE_INTERNAL_CONTEXT_MAGIC) {
        return LECORE_EINVAL;
    }
    if (context->profile != expected_profile) {
        return LECORE_EPROFILE;
    }
    if (context->backend != LECORE_BACKEND_DIRECT &&
        context->backend != LECORE_BACKEND_RADIX2) {
        return LECORE_EBACKEND;
    }
    return LECORE_OK;
}

#if LECORE_ENABLE_RADIX2
LECORE_INTERNAL_API void lecore_internal_hrr_radix2_bind_f64(
    lecore_context *context,
    const double *a,
    const double *b,
    double *output);
LECORE_INTERNAL_API void lecore_internal_hrr_radix2_unbind_f64(
    lecore_context *context,
    const double *composite,
    const double *key,
    double *output);
LECORE_INTERNAL_API void lecore_internal_hrr_radix2_bind_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *output);
LECORE_INTERNAL_API void lecore_internal_hrr_radix2_unbind_f32(
    lecore_context *context,
    const float *composite,
    const float *key,
    float *output);
#endif

static inline int lecore_internal_ranges_overlap(
    const void *left,
    size_t left_bytes,
    const void *right,
    size_t right_bytes)
{
    uintptr_t left_address;
    uintptr_t right_address;

    if (left_bytes == 0 || right_bytes == 0) {
        return 0;
    }

    left_address = (uintptr_t)left;
    right_address = (uintptr_t)right;
    if (left_address <= right_address) {
        return (right_address - left_address) < left_bytes;
    }
    return (left_address - right_address) < right_bytes;
}

static inline lecore_status lecore_internal_check_vector_alias(
    const void *input,
    const void *output,
    size_t vector_bytes,
    int allow_exact_alias)
{
    if (!lecore_internal_ranges_overlap(
            input, vector_bytes, output, vector_bytes)) {
        return LECORE_OK;
    }
    if (allow_exact_alias && input == output) {
        return LECORE_OK;
    }
    return LECORE_EINVAL;
}

static inline lecore_status lecore_internal_matrix_span(
    uint32_t dimension,
    size_t row_count,
    size_t row_stride,
    size_t scalar_bytes,
    size_t *out_span_bytes)
{
    size_t last_row;
    size_t span_elements;

    if (out_span_bytes == NULL) {
        return LECORE_EINVAL;
    }
    *out_span_bytes = 0;
    if (row_count == 0) {
        return LECORE_EINVAL;
    }
    if (row_stride < (size_t)dimension) {
        return LECORE_EDIM;
    }
    if (row_count - 1 > SIZE_MAX / row_stride) {
        return LECORE_EOVERFLOW;
    }
    last_row = (row_count - 1) * row_stride;
    if (last_row > SIZE_MAX - (size_t)dimension) {
        return LECORE_EOVERFLOW;
    }
    span_elements = last_row + (size_t)dimension;
    if (span_elements > SIZE_MAX / scalar_bytes) {
        return LECORE_EOVERFLOW;
    }
    *out_span_bytes = span_elements * scalar_bytes;
    return LECORE_OK;
}

#endif /* LECORE_INTERNAL_H */
