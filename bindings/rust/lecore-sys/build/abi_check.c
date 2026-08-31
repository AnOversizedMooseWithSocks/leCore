#include <lecore/lecore.h>

#include <stddef.h>
#include <stdint.h>

_Static_assert(LECORE_ABI_VERSION == UINT32_C(0),
               "lecore-sys 0.1 audits only liblecore ABI 0");
_Static_assert(sizeof(lecore_status) == sizeof(uint32_t),
               "lecore_status must remain uint32_t");
_Static_assert(sizeof(lecore_profile) == sizeof(uint32_t),
               "lecore_profile must remain uint32_t");
_Static_assert(sizeof(lecore_backend) == sizeof(uint32_t),
               "lecore_backend must remain uint32_t");
_Static_assert(offsetof(lecore_config_v0, allocator) % _Alignof(void *) == 0,
               "allocator must retain pointer alignment");

void lecore_rust_abi_check(void)
{
    lecore_status (LECORE_CALL *bind_f64)(
        lecore_context *, const double *, const double *, double *) =
        &lecore_hrr_bind_f64;
    lecore_status (LECORE_CALL *bind_f32)(
        lecore_context *, const float *, const float *, float *) =
        &lecore_hrr_bind_f32;
    lecore_status (LECORE_CALL *mixed)(
        lecore_context *, const double *, const float *, size_t, size_t,
        double *) = &lecore_cosine_many_f64_f32;

    (void)bind_f64;
    (void)bind_f32;
    (void)mixed;
}
