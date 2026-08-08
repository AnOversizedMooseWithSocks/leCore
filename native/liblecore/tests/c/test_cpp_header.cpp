#include <lecore/lecore.h>

#if LECORE_ENABLE_FORMAT
#include <lecore/lecore_format.h>
#endif

#include <cstdint>
#include <type_traits>

static_assert(std::is_standard_layout<lecore_config_v0>::value,
              "the public configuration must remain standard-layout");

int main()
{
    lecore_config_v0 config{};
    lecore_config_init_v0(&config);
    if (lecore_abi_version() != UINT32_C(0) ||
        config.struct_size != sizeof(config)) {
        return 1;
    }
#if LECORE_ENABLE_FORMAT
    lecore_format_descriptor_v1 descriptor{};
    lecore_format_descriptor_init_v1(&descriptor);
    if (descriptor.header_bytes != LECORE_FORMAT_HEADER_BYTES) {
        return 2;
    }
#endif
    return 0;
}
