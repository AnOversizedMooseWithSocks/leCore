#include <lecore/lecore_format.h>

#include <limits.h>
#include <string.h>

#include "internal/format_internal.h"

#define LECORE_FORMAT_MAGIC_OFFSET ((size_t)0)
#define LECORE_FORMAT_MAJOR_OFFSET ((size_t)8)
#define LECORE_FORMAT_MINOR_OFFSET ((size_t)10)
#define LECORE_FORMAT_HEADER_BYTES_OFFSET ((size_t)12)
#define LECORE_FORMAT_FLAGS_OFFSET ((size_t)14)
#define LECORE_FORMAT_ARTIFACT_KIND_OFFSET ((size_t)16)
#define LECORE_FORMAT_SEMANTIC_PROFILE_OFFSET ((size_t)20)
#define LECORE_FORMAT_SCALAR_TYPE_OFFSET ((size_t)24)
#define LECORE_FORMAT_DIMENSION_OFFSET ((size_t)28)
#define LECORE_FORMAT_NORMALIZATION_OFFSET ((size_t)32)
#define LECORE_FORMAT_ATOM_SCHEME_OFFSET ((size_t)36)
#define LECORE_FORMAT_SEED_NAMESPACE_OFFSET ((size_t)40)
#define LECORE_FORMAT_VECTOR_COUNT_OFFSET ((size_t)48)
#define LECORE_FORMAT_PAYLOAD_BYTES_OFFSET ((size_t)56)
#define LECORE_FORMAT_APPLICATION_CONTRACT_OFFSET ((size_t)64)
#define LECORE_FORMAT_CHECKSUM_OFFSET ((size_t)80)
#define LECORE_FORMAT_RESERVED_OFFSET ((size_t)88)
#define LECORE_FORMAT_CHECKSUM_BYTES ((size_t)8)
#define LECORE_FORMAT_RESERVED_BYTES ((size_t)8)

static const uint8_t lecore_format_magic[8] = {
    (uint8_t)'L', (uint8_t)'E', (uint8_t)'C', (uint8_t)'O',
    (uint8_t)'R', (uint8_t)'E', (uint8_t)'V', UINT8_C(0)
};

static uint16_t lecore_format_get_u16_le(const uint8_t *bytes)
{
    return (uint16_t)((uint16_t)bytes[0]
        | ((uint16_t)bytes[1] << 8));
}

static uint32_t lecore_format_get_u32_le(const uint8_t *bytes)
{
    return (uint32_t)((uint32_t)bytes[0]
        | ((uint32_t)bytes[1] << 8)
        | ((uint32_t)bytes[2] << 16)
        | ((uint32_t)bytes[3] << 24));
}

static uint64_t lecore_format_get_u64_le(const uint8_t *bytes)
{
    return (uint64_t)bytes[0]
        | ((uint64_t)bytes[1] << 8)
        | ((uint64_t)bytes[2] << 16)
        | ((uint64_t)bytes[3] << 24)
        | ((uint64_t)bytes[4] << 32)
        | ((uint64_t)bytes[5] << 40)
        | ((uint64_t)bytes[6] << 48)
        | ((uint64_t)bytes[7] << 56);
}

static void lecore_format_put_u16_le(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
}

static void lecore_format_put_u32_le(uint8_t *bytes, uint32_t value)
{
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
    bytes[2] = (uint8_t)(value >> 16);
    bytes[3] = (uint8_t)(value >> 24);
}

static void lecore_format_put_u64_le(uint8_t *bytes, uint64_t value)
{
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
    bytes[2] = (uint8_t)(value >> 16);
    bytes[3] = (uint8_t)(value >> 24);
    bytes[4] = (uint8_t)(value >> 32);
    bytes[5] = (uint8_t)(value >> 40);
    bytes[6] = (uint8_t)(value >> 48);
    bytes[7] = (uint8_t)(value >> 56);
}

static int lecore_format_bytes_are_zero(const uint8_t *bytes, size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index) {
        if (bytes[index] != UINT8_C(0)) {
            return 0;
        }
    }
    return 1;
}

static int lecore_format_u64s_are_zero(const uint64_t *values, size_t count)
{
    size_t index;

    for (index = 0; index < count; ++index) {
        if (values[index] != UINT64_C(0)) {
            return 0;
        }
    }
    return 1;
}

static lecore_status lecore_format_validate_descriptor(
    const lecore_format_descriptor_v1 *descriptor,
    int encoding)
{
    uint64_t element_count;
    uint64_t expected_payload_bytes;
    uint64_t scalar_bytes;

    if (descriptor == NULL
        || descriptor->struct_size != (uint32_t)sizeof(*descriptor)) {
        return LECORE_EINVAL;
    }
    if (!lecore_format_u64s_are_zero(
            descriptor->reserved,
            sizeof(descriptor->reserved) / sizeof(descriptor->reserved[0]))) {
        return LECORE_EINVAL;
    }
    if (descriptor->format_major != LECORE_FORMAT_MAJOR) {
        return LECORE_EFORMAT;
    }
    if (descriptor->flags != LECORE_FORMAT_FLAGS_NONE) {
        return LECORE_EFORMAT;
    }
    if (descriptor->header_bytes < LECORE_FORMAT_HEADER_BYTES) {
        return LECORE_EFORMAT;
    }
    if (descriptor->format_minor == LECORE_FORMAT_MINOR
        && descriptor->header_bytes != LECORE_FORMAT_HEADER_BYTES) {
        return LECORE_EFORMAT;
    }
    if (encoding
        && (descriptor->format_minor != LECORE_FORMAT_MINOR
            || descriptor->header_bytes != LECORE_FORMAT_HEADER_BYTES)) {
        return LECORE_EUNSUPPORTED;
    }

    if (descriptor->artifact_kind != LECORE_ARTIFACT_VECTOR
        && descriptor->artifact_kind != LECORE_ARTIFACT_MATRIX
        && descriptor->artifact_kind != LECORE_ARTIFACT_TRACE) {
        return LECORE_EFORMAT;
    }
    if (descriptor->semantic_profile == LECORE_PROFILE_HRR_F64_V1) {
        if (descriptor->scalar_type != LECORE_SCALAR_F64) {
            return LECORE_EPROFILE;
        }
        scalar_bytes = UINT64_C(8);
    } else if (descriptor->semantic_profile == LECORE_PROFILE_HRR_F32_V1) {
        if (descriptor->scalar_type != LECORE_SCALAR_F32) {
            return LECORE_EPROFILE;
        }
        scalar_bytes = UINT64_C(4);
    } else {
        return LECORE_EPROFILE;
    }

    if (descriptor->dimension == UINT32_C(0)) {
        return LECORE_EDIM;
    }
    if (descriptor->normalization_policy != LECORE_NORMALIZATION_RAW
        && descriptor->normalization_policy != LECORE_NORMALIZATION_UNIT
        && descriptor->normalization_policy
            != LECORE_NORMALIZATION_APPLICATION) {
        return LECORE_EFORMAT;
    }
    if (descriptor->atom_scheme != LECORE_ATOM_EXTERNAL) {
        return LECORE_EFORMAT;
    }
    if (descriptor->vector_count == UINT64_C(0)) {
        return LECORE_EFORMAT;
    }

    if (descriptor->vector_count
        > UINT64_MAX / (uint64_t)descriptor->dimension) {
        return LECORE_EOVERFLOW;
    }
    element_count = descriptor->vector_count * (uint64_t)descriptor->dimension;
    if (element_count > UINT64_MAX / scalar_bytes) {
        return LECORE_EOVERFLOW;
    }
    expected_payload_bytes = element_count * scalar_bytes;
    if (descriptor->payload_bytes != expected_payload_bytes) {
        return LECORE_EFORMAT;
    }

    return LECORE_OK;
}

static uint64_t lecore_format_record_crc64(
    const uint8_t *record,
    size_t header_bytes,
    size_t payload_bytes)
{
    static const uint8_t zero_checksum[LECORE_FORMAT_CHECKSUM_BYTES] = {
        UINT8_C(0), UINT8_C(0), UINT8_C(0), UINT8_C(0),
        UINT8_C(0), UINT8_C(0), UINT8_C(0), UINT8_C(0)
    };
    uint64_t crc;

    crc = lecore_format_crc64_ecma_update(
        UINT64_C(0), record, LECORE_FORMAT_CHECKSUM_OFFSET);
    crc = lecore_format_crc64_ecma_update(
        crc, zero_checksum, LECORE_FORMAT_CHECKSUM_BYTES);
    crc = lecore_format_crc64_ecma_update(
        crc,
        record + LECORE_FORMAT_RESERVED_OFFSET,
        header_bytes - LECORE_FORMAT_RESERVED_OFFSET);
    return lecore_format_crc64_ecma_update(
        crc, record + header_bytes, payload_bytes);
}

LECORE_API void LECORE_CALL lecore_format_descriptor_init_v1(
    lecore_format_descriptor_v1 *descriptor)
{
    if (descriptor == NULL) {
        return;
    }

    memset(descriptor, 0, sizeof(*descriptor));
    descriptor->struct_size = (uint32_t)sizeof(*descriptor);
    descriptor->format_major = LECORE_FORMAT_MAJOR;
    descriptor->format_minor = LECORE_FORMAT_MINOR;
    descriptor->header_bytes = LECORE_FORMAT_HEADER_BYTES;
    descriptor->semantic_profile = LECORE_PROFILE_HRR_F64_V1;
    descriptor->scalar_type = LECORE_SCALAR_F64;
    descriptor->normalization_policy = LECORE_NORMALIZATION_RAW;
    descriptor->atom_scheme = LECORE_ATOM_EXTERNAL;
}

LECORE_API lecore_status LECORE_CALL lecore_format_encoded_size_v1(
    const lecore_format_descriptor_v1 *descriptor,
    size_t *out_record_bytes)
{
    lecore_format_descriptor_v1 descriptor_copy;
    lecore_status status;

    if (out_record_bytes == NULL) {
        return LECORE_EINVAL;
    }
    if (descriptor == NULL) {
        *out_record_bytes = (size_t)0;
        return LECORE_EINVAL;
    }
    descriptor_copy = *descriptor;
    descriptor = &descriptor_copy;
    *out_record_bytes = (size_t)0;

    status = lecore_format_validate_descriptor(descriptor, 1);
    if (status != LECORE_OK) {
        return status;
    }
    if (descriptor->payload_bytes
        > (uint64_t)(SIZE_MAX - (size_t)descriptor->header_bytes)) {
        return LECORE_EOVERFLOW;
    }

    *out_record_bytes = (size_t)descriptor->header_bytes
        + (size_t)descriptor->payload_bytes;
    return LECORE_OK;
}

LECORE_API lecore_status LECORE_CALL lecore_format_encode_v1(
    const lecore_format_descriptor_v1 *descriptor,
    const void *payload,
    size_t payload_bytes,
    void *record,
    size_t record_capacity,
    size_t *out_record_bytes)
{
    lecore_format_descriptor_v1 descriptor_copy;
    uint8_t *record_bytes;
    uint64_t crc;
    size_t required_bytes;
    lecore_status status;

    if (out_record_bytes == NULL) {
        return LECORE_EINVAL;
    }
    if (descriptor == NULL) {
        *out_record_bytes = (size_t)0;
        return LECORE_EINVAL;
    }
    descriptor_copy = *descriptor;
    descriptor = &descriptor_copy;
    *out_record_bytes = (size_t)0;

    status = lecore_format_encoded_size_v1(descriptor, &required_bytes);
    if (status != LECORE_OK) {
        return status;
    }
    *out_record_bytes = required_bytes;

    if (payload_bytes != (size_t)descriptor->payload_bytes) {
        return LECORE_EFORMAT;
    }
    if (payload == NULL || record == NULL) {
        return LECORE_EINVAL;
    }
    if (record_capacity < required_bytes) {
        return LECORE_EINVAL;
    }

    record_bytes = (uint8_t *)record;
    memmove(
        record_bytes + descriptor->header_bytes,
        payload,
        payload_bytes);
    memset(record_bytes, 0, descriptor->header_bytes);

    memcpy(
        record_bytes + LECORE_FORMAT_MAGIC_OFFSET,
        lecore_format_magic,
        sizeof(lecore_format_magic));
    lecore_format_put_u16_le(
        record_bytes + LECORE_FORMAT_MAJOR_OFFSET,
        descriptor->format_major);
    lecore_format_put_u16_le(
        record_bytes + LECORE_FORMAT_MINOR_OFFSET,
        descriptor->format_minor);
    lecore_format_put_u16_le(
        record_bytes + LECORE_FORMAT_HEADER_BYTES_OFFSET,
        descriptor->header_bytes);
    lecore_format_put_u16_le(
        record_bytes + LECORE_FORMAT_FLAGS_OFFSET,
        descriptor->flags);
    lecore_format_put_u32_le(
        record_bytes + LECORE_FORMAT_ARTIFACT_KIND_OFFSET,
        descriptor->artifact_kind);
    lecore_format_put_u32_le(
        record_bytes + LECORE_FORMAT_SEMANTIC_PROFILE_OFFSET,
        descriptor->semantic_profile);
    lecore_format_put_u32_le(
        record_bytes + LECORE_FORMAT_SCALAR_TYPE_OFFSET,
        descriptor->scalar_type);
    lecore_format_put_u32_le(
        record_bytes + LECORE_FORMAT_DIMENSION_OFFSET,
        descriptor->dimension);
    lecore_format_put_u32_le(
        record_bytes + LECORE_FORMAT_NORMALIZATION_OFFSET,
        descriptor->normalization_policy);
    lecore_format_put_u32_le(
        record_bytes + LECORE_FORMAT_ATOM_SCHEME_OFFSET,
        descriptor->atom_scheme);
    lecore_format_put_u64_le(
        record_bytes + LECORE_FORMAT_SEED_NAMESPACE_OFFSET,
        descriptor->seed_namespace);
    lecore_format_put_u64_le(
        record_bytes + LECORE_FORMAT_VECTOR_COUNT_OFFSET,
        descriptor->vector_count);
    lecore_format_put_u64_le(
        record_bytes + LECORE_FORMAT_PAYLOAD_BYTES_OFFSET,
        descriptor->payload_bytes);
    memcpy(
        record_bytes + LECORE_FORMAT_APPLICATION_CONTRACT_OFFSET,
        descriptor->application_contract,
        LECORE_FORMAT_APPLICATION_CONTRACT_BYTES);

    crc = lecore_format_record_crc64(
        record_bytes,
        descriptor->header_bytes,
        payload_bytes);
    lecore_format_put_u64_le(
        record_bytes + LECORE_FORMAT_CHECKSUM_OFFSET,
        crc);
    return LECORE_OK;
}

LECORE_API lecore_status LECORE_CALL lecore_format_decode_v1(
    const void *record,
    size_t record_bytes,
    lecore_format_descriptor_v1 *out_descriptor,
    const void **out_payload,
    size_t *out_payload_bytes)
{
    const uint8_t *bytes;
    lecore_format_descriptor_v1 descriptor;
    uint64_t actual_crc;
    size_t payload_bytes;
    size_t total_bytes;
    lecore_status status;

    if (out_descriptor == NULL || out_payload == NULL
        || out_payload_bytes == NULL) {
        return LECORE_EINVAL;
    }
    memset(out_descriptor, 0, sizeof(*out_descriptor));
    out_descriptor->struct_size = (uint32_t)sizeof(*out_descriptor);
    *out_payload = NULL;
    *out_payload_bytes = (size_t)0;

    if (record == NULL) {
        return LECORE_EINVAL;
    }
    if (record_bytes < (size_t)LECORE_FORMAT_HEADER_BYTES) {
        return LECORE_EFORMAT;
    }

    bytes = (const uint8_t *)record;
    if (memcmp(
            bytes + LECORE_FORMAT_MAGIC_OFFSET,
            lecore_format_magic,
            sizeof(lecore_format_magic)) != 0) {
        return LECORE_EFORMAT;
    }

    memset(&descriptor, 0, sizeof(descriptor));
    descriptor.struct_size = (uint32_t)sizeof(descriptor);
    descriptor.format_major = lecore_format_get_u16_le(
        bytes + LECORE_FORMAT_MAJOR_OFFSET);
    descriptor.format_minor = lecore_format_get_u16_le(
        bytes + LECORE_FORMAT_MINOR_OFFSET);
    descriptor.header_bytes = lecore_format_get_u16_le(
        bytes + LECORE_FORMAT_HEADER_BYTES_OFFSET);
    descriptor.flags = lecore_format_get_u16_le(
        bytes + LECORE_FORMAT_FLAGS_OFFSET);
    descriptor.artifact_kind = lecore_format_get_u32_le(
        bytes + LECORE_FORMAT_ARTIFACT_KIND_OFFSET);
    descriptor.semantic_profile = lecore_format_get_u32_le(
        bytes + LECORE_FORMAT_SEMANTIC_PROFILE_OFFSET);
    descriptor.scalar_type = lecore_format_get_u32_le(
        bytes + LECORE_FORMAT_SCALAR_TYPE_OFFSET);
    descriptor.dimension = lecore_format_get_u32_le(
        bytes + LECORE_FORMAT_DIMENSION_OFFSET);
    descriptor.normalization_policy = lecore_format_get_u32_le(
        bytes + LECORE_FORMAT_NORMALIZATION_OFFSET);
    descriptor.atom_scheme = lecore_format_get_u32_le(
        bytes + LECORE_FORMAT_ATOM_SCHEME_OFFSET);
    descriptor.seed_namespace = lecore_format_get_u64_le(
        bytes + LECORE_FORMAT_SEED_NAMESPACE_OFFSET);
    descriptor.vector_count = lecore_format_get_u64_le(
        bytes + LECORE_FORMAT_VECTOR_COUNT_OFFSET);
    descriptor.payload_bytes = lecore_format_get_u64_le(
        bytes + LECORE_FORMAT_PAYLOAD_BYTES_OFFSET);
    memcpy(
        descriptor.application_contract,
        bytes + LECORE_FORMAT_APPLICATION_CONTRACT_OFFSET,
        LECORE_FORMAT_APPLICATION_CONTRACT_BYTES);
    descriptor.content_crc64 = lecore_format_get_u64_le(
        bytes + LECORE_FORMAT_CHECKSUM_OFFSET);

    if (!lecore_format_bytes_are_zero(
            bytes + LECORE_FORMAT_RESERVED_OFFSET,
            LECORE_FORMAT_RESERVED_BYTES)) {
        return LECORE_EFORMAT;
    }
    status = lecore_format_validate_descriptor(&descriptor, 0);
    if (status != LECORE_OK) {
        return status;
    }
    if ((size_t)descriptor.header_bytes > record_bytes) {
        return LECORE_EFORMAT;
    }
    if (descriptor.payload_bytes
        > (uint64_t)(SIZE_MAX - (size_t)descriptor.header_bytes)) {
        return LECORE_EOVERFLOW;
    }
    payload_bytes = (size_t)descriptor.payload_bytes;
    total_bytes = (size_t)descriptor.header_bytes + payload_bytes;
    if (record_bytes != total_bytes) {
        return LECORE_EFORMAT;
    }

    actual_crc = lecore_format_record_crc64(
        bytes, descriptor.header_bytes, payload_bytes);
    if (actual_crc != descriptor.content_crc64) {
        return LECORE_ECHECKSUM;
    }

    *out_descriptor = descriptor;
    *out_payload = bytes + descriptor.header_bytes;
    *out_payload_bytes = payload_bytes;
    return LECORE_OK;
}

LECORE_API lecore_status LECORE_CALL lecore_format_check_v1(
    const void *record,
    size_t record_bytes)
{
    lecore_format_descriptor_v1 descriptor;
    const void *payload;
    size_t payload_bytes;

    return lecore_format_decode_v1(
        record,
        record_bytes,
        &descriptor,
        &payload,
        &payload_bytes);
}

LECORE_API lecore_status LECORE_CALL lecore_format_check_compatibility_v1(
    const lecore_format_descriptor_v1 *expected,
    const lecore_format_descriptor_v1 *actual)
{
    lecore_status status;

    status = lecore_format_validate_descriptor(expected, 0);
    if (status != LECORE_OK) {
        return status;
    }
    status = lecore_format_validate_descriptor(actual, 0);
    if (status != LECORE_OK) {
        return status;
    }

    if (expected->semantic_profile != actual->semantic_profile
        || expected->scalar_type != actual->scalar_type) {
        return LECORE_EPROFILE;
    }
    if (expected->dimension != actual->dimension) {
        return LECORE_EDIM;
    }
    if (expected->format_major != actual->format_major
        || expected->flags != actual->flags
        || expected->artifact_kind != actual->artifact_kind
        || expected->normalization_policy != actual->normalization_policy
        || expected->atom_scheme != actual->atom_scheme
        || expected->seed_namespace != actual->seed_namespace
        || expected->vector_count != actual->vector_count
        || expected->payload_bytes != actual->payload_bytes
        || memcmp(
            expected->application_contract,
            actual->application_contract,
            LECORE_FORMAT_APPLICATION_CONTRACT_BYTES) != 0) {
        return LECORE_EFORMAT;
    }

    return LECORE_OK;
}
