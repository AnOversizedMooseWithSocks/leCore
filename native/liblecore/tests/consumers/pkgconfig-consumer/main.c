#include <lecore/lecore.h>

#include <stdint.h>

int main(void)
{
    lecore_config_v0 config;
    lecore_config_init_v0(&config);
    if (!(lecore_abi_version() == UINT32_C(0) &&
        lecore_isa_version() == UINT32_C(1) &&
        config.abi_version == UINT32_C(0))) {
        return 1;
    }
#if LECORE_ENABLE_FORMAT
    if ((lecore_capabilities() & LECORE_CAP_FORMAT) == 0) {
        return 2;
    }
#else
    if ((lecore_capabilities() & LECORE_CAP_FORMAT) != 0) {
        return 3;
    }
#endif
#if LECORE_ENABLE_RADIX2
    return (lecore_capabilities() & LECORE_CAP_RADIX2) != 0 ? 0 : 4;
#else
    return (lecore_capabilities() & LECORE_CAP_RADIX2) == 0 ? 0 : 5;
#endif
}
