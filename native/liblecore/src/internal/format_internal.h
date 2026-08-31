#ifndef LECORE_INTERNAL_FORMAT_INTERNAL_H
#define LECORE_INTERNAL_FORMAT_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#ifndef LECORE_INTERNAL_API
#  if defined(LECORE_AMALGAMATION)
#    define LECORE_INTERNAL_API static
#  else
#    define LECORE_INTERNAL_API
#  endif
#endif

LECORE_INTERNAL_API uint64_t lecore_format_crc64_ecma_update(
    uint64_t crc,
    const uint8_t *data,
    size_t data_bytes);

#endif /* LECORE_INTERNAL_FORMAT_INTERNAL_H */
