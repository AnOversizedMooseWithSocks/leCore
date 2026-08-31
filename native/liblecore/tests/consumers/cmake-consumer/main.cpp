#include <lecore/lecore.h>

#if LECORE_ENABLE_FORMAT
#include <lecore/lecore_format.h>
#endif

#include <cstdint>

int main()
{
    lecore_config_v0 config{};
    lecore_config_init_v0(&config);
    if (lecore_abi_version() != UINT32_C(0) ||
        lecore_isa_version() != UINT32_C(1) ||
        config.abi_version != UINT32_C(0)) {
        return 1;
    }
#if LECORE_ENABLE_FORMAT
    if ((lecore_capabilities() & LECORE_CAP_FORMAT) == 0) {
        return 2;
    }
    lecore_format_descriptor_v1 descriptor{};
    lecore_format_descriptor_init_v1(&descriptor);
    if (descriptor.format_major != LECORE_FORMAT_MAJOR) {
        return 3;
    }
#else
    if ((lecore_capabilities() & LECORE_CAP_FORMAT) != 0) {
        return 4;
    }
#endif
#if LECORE_ENABLE_RADIX2
    if ((lecore_capabilities() & LECORE_CAP_RADIX2) == 0) {
        return 5;
    }
#else
    if ((lecore_capabilities() & LECORE_CAP_RADIX2) != 0) {
        return 6;
    }
#endif
    return 0;
}
