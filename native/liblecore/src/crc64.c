#include "internal/format_internal.h"

#define LECORE_CRC64_ECMA_POLYNOMIAL UINT64_C(0x42F0E1EBA9EA3693)

LECORE_INTERNAL_API uint64_t lecore_format_crc64_ecma_update(
    uint64_t crc,
    const uint8_t *data,
    size_t data_bytes)
{
    size_t byte_index;

    for (byte_index = 0; byte_index < data_bytes; ++byte_index) {
        unsigned bit_index;

        crc ^= (uint64_t)data[byte_index] << 56;
        for (bit_index = 0; bit_index < 8; ++bit_index) {
            if ((crc & UINT64_C(0x8000000000000000)) != 0) {
                crc = (crc << 1) ^ LECORE_CRC64_ECMA_POLYNOMIAL;
            } else {
                crc <<= 1;
            }
        }
    }

    return crc;
}
