#ifndef LECORE_LECORE_H
#define LECORE_LECORE_H

#include <float.h>
#include <limits.h>
#include <stddef.h>
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

#if defined(__cplusplus)
static_assert(sizeof(float) == 4, "liblecore requires 32-bit float");
static_assert(sizeof(double) == 8, "liblecore requires 64-bit double");
#else
_Static_assert(sizeof(float) == 4, "liblecore requires 32-bit float");
_Static_assert(sizeof(double) == 8, "liblecore requires 64-bit double");
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(LECORE_SHARED)
#    if defined(LECORE_BUILDING_LIBRARY)
#      define LECORE_API __declspec(dllexport)
#    else
#      define LECORE_API __declspec(dllimport)
#    endif
#  else
#    define LECORE_API
#  endif
#  if defined(_MSC_VER)
#    define LECORE_CALL __cdecl
#  else
#    define LECORE_CALL
#  endif
#elif defined(__GNUC__) && __GNUC__ >= 4
#  define LECORE_API __attribute__((visibility("default")))
#  define LECORE_CALL
#else
#  define LECORE_API
#  define LECORE_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* This is an intentionally unstable 0.x preview ABI. */
#ifndef LECORE_ABI_VERSION_VALUE
#  define LECORE_ABI_VERSION_VALUE 0
#endif
#define LECORE_ABI_VERSION ((uint32_t)LECORE_ABI_VERSION_VALUE)

#ifndef LECORE_ISA_VERSION_VALUE
#  define LECORE_ISA_VERSION_VALUE 1
#endif
#define LECORE_ISA_VERSION ((uint32_t)LECORE_ISA_VERSION_VALUE)

#ifndef LECORE_VERSION_STRING_VALUE
#  define LECORE_VERSION_STRING_VALUE "0.1.0"
#endif

typedef uint32_t lecore_status;
#define LECORE_OK             UINT32_C(0)
#define LECORE_EINVAL         UINT32_C(1)
#define LECORE_EDIM           UINT32_C(2)
#define LECORE_EPROFILE       UINT32_C(3)
#define LECORE_EBACKEND       UINT32_C(4)
#define LECORE_EOVERFLOW      UINT32_C(5)
#define LECORE_ENOMEM         UINT32_C(6)
#define LECORE_EUNSUPPORTED   UINT32_C(7)
#define LECORE_ENONFINITE     UINT32_C(8)
#define LECORE_EFORMAT        UINT32_C(9)
#define LECORE_ECHECKSUM      UINT32_C(10)

typedef uint32_t lecore_profile;
#define LECORE_PROFILE_HRR_F64_V1 UINT32_C(0x00010001)
#define LECORE_PROFILE_HRR_F32_V1 UINT32_C(0x00010002)

typedef uint32_t lecore_backend;
/* AUTO is conservatively DIRECT in ABI 0; RADIX2 requires its capability bit
 * and a power-of-two dimension. Forced unsupported selection fails loudly. */
#define LECORE_BACKEND_AUTO   UINT32_C(0)
#define LECORE_BACKEND_DIRECT UINT32_C(1)
#define LECORE_BACKEND_RADIX2 UINT32_C(2)

typedef uint32_t lecore_validation;
#define LECORE_VALIDATION_SHAPE  UINT32_C(0)
#define LECORE_VALIDATION_FINITE UINT32_C(1)

typedef uint32_t lecore_scalar_type;
#define LECORE_SCALAR_F64 UINT32_C(1)
#define LECORE_SCALAR_F32 UINT32_C(2)

/* Capability bits returned by lecore_capabilities(). */
#define LECORE_CAP_HRR_F64          (UINT64_C(1) << 0)
#define LECORE_CAP_HRR_F32          (UINT64_C(1) << 1)
#define LECORE_CAP_DIRECT           (UINT64_C(1) << 2)
#define LECORE_CAP_RADIX2           (UINT64_C(1) << 3)
#define LECORE_CAP_BATCH            (UINT64_C(1) << 4)
#define LECORE_CAP_MIXED_F64_F32    (UINT64_C(1) << 5)
#define LECORE_CAP_FINITE_VALIDATION (UINT64_C(1) << 6)
#define LECORE_CAP_FORMAT           (UINT64_C(1) << 7)

/* Supply both callbacks or neither. Allocation requests occur only during
 * context creation, have nonzero byte counts and power-of-two alignment, and
 * must return NULL or a suitably aligned pointer. Every successful request is
 * released exactly once with the identical user/bytes/alignment tuple during
 * failed creation or context destruction. */
typedef struct lecore_allocator_v0 {
    uint32_t struct_size;
    void *user;
    void *(LECORE_CALL *allocate)(void *user, size_t bytes, size_t alignment);
    void (LECORE_CALL *deallocate)(
        void *user,
        void *pointer,
        size_t bytes,
        size_t alignment);
} lecore_allocator_v0;

typedef struct lecore_config_v0 {
    uint32_t struct_size;
    uint32_t abi_version;
    lecore_profile profile;
    lecore_backend backend;
    lecore_validation validation;
    uint32_t flags;
    uint32_t dimension;
    lecore_allocator_v0 allocator;
    uint64_t reserved[4];
} lecore_config_v0;

typedef struct lecore_context lecore_context;

LECORE_API uint32_t LECORE_CALL lecore_abi_version(void);
LECORE_API uint32_t LECORE_CALL lecore_isa_version(void);
LECORE_API const char *LECORE_CALL lecore_version_string(void);
LECORE_API uint64_t LECORE_CALL lecore_capabilities(void);
LECORE_API const char *LECORE_CALL lecore_status_string(lecore_status status);

/* Initializes the complete preview-0 configuration. A NULL argument is ignored. */
LECORE_API void LECORE_CALL lecore_config_init_v0(lecore_config_v0 *config);
LECORE_API lecore_status LECORE_CALL lecore_context_create(
    const lecore_config_v0 *config,
    lecore_context **out_context);
/* Context calls require exclusive single-thread ownership. Destruction is the
 * sole thread-neutral operation: it may run on another thread only after every
 * call using the context has quiesced. A custom deallocate callback must be
 * valid on the destroying thread. Concurrent destroy/use is invalid. */
LECORE_API void LECORE_CALL lecore_context_destroy(lecore_context *context);

LECORE_API uint32_t LECORE_CALL lecore_context_dimension(
    const lecore_context *context);
LECORE_API lecore_profile LECORE_CALL lecore_context_profile(
    const lecore_context *context);
LECORE_API lecore_backend LECORE_CALL lecore_context_backend(
    const lecore_context *context);
LECORE_API lecore_validation LECORE_CALL lecore_context_validation(
    const lecore_context *context);
LECORE_API lecore_scalar_type LECORE_CALL lecore_context_scalar_type(
    const lecore_context *context);
LECORE_API size_t LECORE_CALL lecore_context_scratch_bytes(
    const lecore_context *context);

/* Standalone ingestion-boundary checks. Empty ranges are valid. */
LECORE_API lecore_status LECORE_CALL lecore_validate_f64(
    const double *values,
    size_t count);
LECORE_API lecore_status LECORE_CALL lecore_validate_f32(
    const float *values,
    size_t count);

/*
 * Vector arguments contain context->dimension elements. Exact input/output
 * aliasing is supported by normalize, involution, permute, bind, and unbind.
 * Partial overlap is rejected. Scalar outputs are written only after inputs
 * have been consumed.
 */
LECORE_API lecore_status LECORE_CALL lecore_normalize_f64(
    lecore_context *context,
    const double *input,
    double *output);
LECORE_API lecore_status LECORE_CALL lecore_normalize_f32(
    lecore_context *context,
    const float *input,
    float *output);

LECORE_API lecore_status LECORE_CALL lecore_dot_f64(
    lecore_context *context,
    const double *a,
    const double *b,
    double *out_dot);
LECORE_API lecore_status LECORE_CALL lecore_dot_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *out_dot);

LECORE_API lecore_status LECORE_CALL lecore_dot_many_f64(
    lecore_context *context,
    const double *query,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *out_scores);
LECORE_API lecore_status LECORE_CALL lecore_dot_many_f32(
    lecore_context *context,
    const float *query,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *out_scores);

LECORE_API lecore_status LECORE_CALL lecore_cosine_f64(
    lecore_context *context,
    const double *a,
    const double *b,
    double *out_cosine);
LECORE_API lecore_status LECORE_CALL lecore_cosine_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *out_cosine);

LECORE_API lecore_status LECORE_CALL lecore_cosine_many_f64(
    lecore_context *context,
    const double *query,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *out_scores);
LECORE_API lecore_status LECORE_CALL lecore_cosine_many_f32(
    lecore_context *context,
    const float *query,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *out_scores);

LECORE_API lecore_status LECORE_CALL lecore_hrr_bind_f64(
    lecore_context *context,
    const double *a,
    const double *b,
    double *output);
LECORE_API lecore_status LECORE_CALL lecore_hrr_bind_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *output);

LECORE_API lecore_status LECORE_CALL lecore_hrr_unbind_f64(
    lecore_context *context,
    const double *composite,
    const double *key,
    double *output);
LECORE_API lecore_status LECORE_CALL lecore_hrr_unbind_f32(
    lecore_context *context,
    const float *composite,
    const float *key,
    float *output);

LECORE_API lecore_status LECORE_CALL lecore_involution_f64(
    lecore_context *context,
    const double *input,
    double *output);
LECORE_API lecore_status LECORE_CALL lecore_involution_f32(
    lecore_context *context,
    const float *input,
    float *output);

LECORE_API lecore_status LECORE_CALL lecore_permute_f64(
    lecore_context *context,
    const double *input,
    int64_t shift,
    double *output);
LECORE_API lecore_status LECORE_CALL lecore_permute_f32(
    lecore_context *context,
    const float *input,
    int64_t shift,
    float *output);

/* Matrix rows and strides are measured in scalar elements, not bytes. */
LECORE_API lecore_status LECORE_CALL lecore_bundle_f64(
    lecore_context *context,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *output);
LECORE_API lecore_status LECORE_CALL lecore_bundle_f32(
    lecore_context *context,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *output);

LECORE_API lecore_status LECORE_CALL lecore_cleanup_f64(
    lecore_context *context,
    const double *query,
    const double *candidates,
    size_t candidate_count,
    size_t candidate_stride,
    size_t *out_index,
    double *out_score);
LECORE_API lecore_status LECORE_CALL lecore_cleanup_f32(
    lecore_context *context,
    const float *query,
    const float *candidates,
    size_t candidate_count,
    size_t candidate_stride,
    size_t *out_index,
    float *out_score);

/* Batch input and output matrices must be disjoint in preview ABI 0. */
LECORE_API lecore_status LECORE_CALL lecore_hrr_bind_batch_f64(
    lecore_context *context,
    const double *a_rows,
    size_t a_stride,
    const double *b_rows,
    size_t b_stride,
    size_t row_count,
    double *out_rows,
    size_t out_stride);
LECORE_API lecore_status LECORE_CALL lecore_hrr_bind_batch_f32(
    lecore_context *context,
    const float *a_rows,
    size_t a_stride,
    const float *b_rows,
    size_t b_stride,
    size_t row_count,
    float *out_rows,
    size_t out_stride);

LECORE_API lecore_status LECORE_CALL lecore_hrr_bind_fixed_f64(
    lecore_context *context,
    const double *role,
    const double *rows,
    size_t row_count,
    size_t row_stride,
    double *out_rows,
    size_t out_stride);
LECORE_API lecore_status LECORE_CALL lecore_hrr_bind_fixed_f32(
    lecore_context *context,
    const float *role,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *out_rows,
    size_t out_stride);

LECORE_API lecore_status LECORE_CALL lecore_hrr_unbind_all_f64(
    lecore_context *context,
    const double *trace,
    const double *keys,
    size_t key_count,
    size_t key_stride,
    double *out_rows,
    size_t out_stride);
LECORE_API lecore_status LECORE_CALL lecore_hrr_unbind_all_f32(
    lecore_context *context,
    const float *trace,
    const float *keys,
    size_t key_count,
    size_t key_stride,
    float *out_rows,
    size_t out_stride);

/* NoSQLite utility: f64 query, f32 rows, ascending-order f64 reductions. */
LECORE_API lecore_status LECORE_CALL lecore_cosine_many_f64_f32(
    lecore_context *f32_context,
    const double *query,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    double *out_scores);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LECORE_LECORE_H */
