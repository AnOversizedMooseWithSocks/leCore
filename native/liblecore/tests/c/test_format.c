#include "test_common.h"

#include <lecore/lecore_format.h>

#include <stdint.h>
#include <string.h>

static const uint8_t golden_header[LECORE_FORMAT_HEADER_BYTES] = {
    0x4c, 0x45, 0x43, 0x4f, 0x52, 0x45, 0x56, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x60, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x3c, 0x54, 0x2b, 0xeb, 0x98, 0xdc, 0xcc, 0xb6,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};

static const uint8_t golden_payload[16] = {
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf4, 0x3f,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0xc0
};

static uint64_t test_crc64(const uint8_t *bytes, size_t count)
{
    const uint64_t polynomial = UINT64_C(0x42f0e1eba9ea3693);
    uint64_t crc = UINT64_C(0);
    size_t byte_index;

    for (byte_index = 0; byte_index < count; ++byte_index) {
        unsigned bit_index;
        crc ^= (uint64_t)bytes[byte_index] << 56;
        for (bit_index = 0; bit_index < 8; ++bit_index) {
            crc = (crc & UINT64_C(0x8000000000000000)) != 0
                ? (crc << 1) ^ polynomial
                : crc << 1;
        }
    }
    return crc;
}

static void put_u16_le(uint8_t *bytes, uint16_t value)
{
    bytes[0] = (uint8_t)value;
    bytes[1] = (uint8_t)(value >> 8);
}

static void put_u64_le(uint8_t *bytes, uint64_t value)
{
    unsigned index;
    for (index = 0; index < 8; ++index) {
        bytes[index] = (uint8_t)(value >> (index * 8));
    }
}

static void set_record_crc(uint8_t *record, size_t record_bytes)
{
    uint64_t crc;
    memset(record + 80, 0, 8);
    crc = test_crc64(record, record_bytes);
    put_u64_le(record + 80, crc);
}

static void make_descriptor(lecore_format_descriptor_v1 *descriptor)
{
    lecore_format_descriptor_init_v1(descriptor);
    descriptor->artifact_kind = LECORE_ARTIFACT_VECTOR;
    descriptor->dimension = 2;
    descriptor->vector_count = UINT64_C(1);
    descriptor->payload_bytes = UINT64_C(16);
}

static int test_golden_and_roundtrip(void)
{
    union descriptor_record_storage {
        max_align_t alignment;
        uint8_t bytes[256];
    } descriptor_record;
    lecore_format_descriptor_v1 descriptor;
    lecore_format_descriptor_v1 *overlapping_descriptor;
    lecore_format_descriptor_v1 decoded;
    uint8_t record[112];
    uint8_t overlap_record[112];
    const void *decoded_payload = NULL;
    size_t decoded_payload_bytes = 0;
    size_t record_bytes = 0;

    memset(&descriptor, 0xa5, sizeof(descriptor));
    lecore_format_descriptor_init_v1(&descriptor);
    CHECK(descriptor.struct_size == sizeof(descriptor));
    CHECK(descriptor.format_major == LECORE_FORMAT_MAJOR);
    CHECK(descriptor.format_minor == LECORE_FORMAT_MINOR);
    CHECK(descriptor.header_bytes == LECORE_FORMAT_HEADER_BYTES);
    CHECK(descriptor.semantic_profile == LECORE_PROFILE_HRR_F64_V1);
    CHECK(descriptor.scalar_type == LECORE_SCALAR_F64);
    CHECK(descriptor.normalization_policy == LECORE_NORMALIZATION_RAW);
    CHECK(descriptor.atom_scheme == LECORE_ATOM_EXTERNAL);
    lecore_format_descriptor_init_v1(NULL);

    make_descriptor(&descriptor);
    CHECK_STATUS(lecore_format_encoded_size_v1(&descriptor, &record_bytes), LECORE_OK);
    CHECK(record_bytes == sizeof(record));
    CHECK_STATUS(lecore_format_encode_v1(
        &descriptor,
        golden_payload,
        sizeof(golden_payload),
        record,
        sizeof(record),
        &record_bytes), LECORE_OK);
    CHECK(record_bytes == sizeof(record));
    CHECK(memcmp(record, golden_header, sizeof(golden_header)) == 0);
    CHECK(memcmp(record + sizeof(golden_header), golden_payload, sizeof(golden_payload)) == 0);
    CHECK_STATUS(lecore_format_check_v1(record, sizeof(record)), LECORE_OK);

    CHECK_STATUS(lecore_format_decode_v1(
        record,
        sizeof(record),
        &decoded,
        &decoded_payload,
        &decoded_payload_bytes), LECORE_OK);
    CHECK(decoded_payload == record + LECORE_FORMAT_HEADER_BYTES);
    CHECK(decoded_payload_bytes == sizeof(golden_payload));
    CHECK(memcmp(decoded_payload, golden_payload, sizeof(golden_payload)) == 0);
    CHECK(decoded.content_crc64 == UINT64_C(0xb6ccdc98eb2b543c));
    CHECK(decoded.dimension == descriptor.dimension);
    CHECK_STATUS(lecore_format_check_compatibility_v1(&descriptor, &decoded), LECORE_OK);

    memcpy(overlap_record, golden_payload, sizeof(golden_payload));
    CHECK_STATUS(lecore_format_encode_v1(
        &descriptor,
        overlap_record,
        sizeof(golden_payload),
        overlap_record,
        sizeof(overlap_record),
        &record_bytes), LECORE_OK);
    CHECK(memcmp(overlap_record, record, sizeof(record)) == 0);

    overlapping_descriptor =
        (lecore_format_descriptor_v1 *)(void *)descriptor_record.bytes;
    make_descriptor(overlapping_descriptor);
    CHECK_STATUS(lecore_format_encode_v1(
        overlapping_descriptor,
        golden_payload,
        sizeof(golden_payload),
        descriptor_record.bytes,
        sizeof(descriptor_record.bytes),
        &record_bytes), LECORE_OK);
    CHECK(memcmp(descriptor_record.bytes, record, sizeof(record)) == 0);
    return EXIT_SUCCESS;
}

static int test_descriptor_errors(void)
{
    lecore_format_descriptor_v1 descriptor;
    size_t record_bytes = 99;

    make_descriptor(&descriptor);
    CHECK_STATUS(lecore_format_encoded_size_v1(NULL, &record_bytes), LECORE_EINVAL);
    CHECK(record_bytes == 0);
    CHECK_STATUS(lecore_format_encoded_size_v1(&descriptor, NULL), LECORE_EINVAL);

#define CHECK_DESCRIPTOR_ERROR(field, value, expected_status)                   \
    do {                                                                        \
        lecore_format_descriptor_v1 changed_ = descriptor;                      \
        changed_.field = (value);                                               \
        record_bytes = 99;                                                      \
        CHECK_STATUS(lecore_format_encoded_size_v1(                             \
            &changed_, &record_bytes), (expected_status));                      \
        CHECK(record_bytes == 0);                                               \
    } while (0)

    CHECK_DESCRIPTOR_ERROR(struct_size, 0, LECORE_EINVAL);
    CHECK_DESCRIPTOR_ERROR(format_major, 2, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(format_minor, 1, LECORE_EUNSUPPORTED);
    CHECK_DESCRIPTOR_ERROR(header_bytes, 95, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(header_bytes, 97, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(flags, 1, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(artifact_kind, 99, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(semantic_profile, 99, LECORE_EPROFILE);
    CHECK_DESCRIPTOR_ERROR(scalar_type, LECORE_SCALAR_F32, LECORE_EPROFILE);
    CHECK_DESCRIPTOR_ERROR(dimension, 0, LECORE_EDIM);
    CHECK_DESCRIPTOR_ERROR(normalization_policy, 99, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(atom_scheme, 99, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(vector_count, 0, LECORE_EFORMAT);
    CHECK_DESCRIPTOR_ERROR(payload_bytes, 8, LECORE_EFORMAT);

    descriptor.reserved[0] = 1;
    CHECK_STATUS(lecore_format_encoded_size_v1(&descriptor, &record_bytes), LECORE_EINVAL);
    descriptor.reserved[0] = 0;
    descriptor.dimension = UINT32_MAX;
    descriptor.vector_count = UINT64_MAX;
    descriptor.payload_bytes = UINT64_MAX;
    CHECK_STATUS(lecore_format_encoded_size_v1(&descriptor, &record_bytes), LECORE_EOVERFLOW);
#undef CHECK_DESCRIPTOR_ERROR
    return EXIT_SUCCESS;
}

static int test_record_errors_and_forward_minor(void)
{
    lecore_format_descriptor_v1 descriptor;
    lecore_format_descriptor_v1 decoded;
    uint8_t record[112];
    uint8_t changed[113];
    const void *payload = (const void *)(uintptr_t)1;
    size_t payload_bytes = 99;
    size_t record_bytes = 0;

    make_descriptor(&descriptor);
    CHECK_STATUS(lecore_format_encode_v1(
        &descriptor, golden_payload, sizeof(golden_payload),
        record, sizeof(record), &record_bytes), LECORE_OK);

    CHECK_STATUS(lecore_format_encode_v1(
        &descriptor, golden_payload, sizeof(golden_payload) - 1,
        changed, sizeof(changed), &record_bytes), LECORE_EFORMAT);
    CHECK(record_bytes == sizeof(record));
    CHECK_STATUS(lecore_format_encode_v1(
        &descriptor, golden_payload, sizeof(golden_payload),
        changed, sizeof(record) - 1, &record_bytes), LECORE_EINVAL);
    CHECK(record_bytes == sizeof(record));
    CHECK_STATUS(lecore_format_encode_v1(
        &descriptor, NULL, sizeof(golden_payload),
        changed, sizeof(changed), &record_bytes), LECORE_EINVAL);

    memset(&decoded, 0xa5, sizeof(decoded));
    CHECK_STATUS(lecore_format_decode_v1(
        NULL, sizeof(record), &decoded, &payload, &payload_bytes), LECORE_EINVAL);
    CHECK(decoded.struct_size == sizeof(decoded));
    CHECK(payload == NULL && payload_bytes == 0);
    CHECK_STATUS(lecore_format_decode_v1(
        record, 95, &decoded, &payload, &payload_bytes), LECORE_EFORMAT);
    CHECK_STATUS(lecore_format_decode_v1(
        record, sizeof(record) - 1, &decoded, &payload, &payload_bytes), LECORE_EFORMAT);
    memcpy(changed, record, sizeof(record));
    changed[sizeof(record)] = 0;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(changed)), LECORE_EFORMAT);

    memcpy(changed, record, sizeof(record));
    changed[0] ^= 1;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_EFORMAT);
    memcpy(changed, record, sizeof(record));
    changed[8] = 2;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_EFORMAT);
    memcpy(changed, record, sizeof(record));
    changed[14] = 1;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_EFORMAT);
    memcpy(changed, record, sizeof(record));
    changed[24] = LECORE_SCALAR_F32;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_EPROFILE);
    memcpy(changed, record, sizeof(record));
    changed[28] = 0;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_EDIM);
    memcpy(changed, record, sizeof(record));
    changed[88] = 1;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_EFORMAT);
    memcpy(changed, record, sizeof(record));
    changed[100] ^= 1;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_ECHECKSUM);
    memcpy(changed, record, sizeof(record));
    changed[40] ^= 1;
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(record)), LECORE_ECHECKSUM);

    /* A newer minor may extend a sane header when no unknown mandatory flag is set. */
    memcpy(changed, record, LECORE_FORMAT_HEADER_BYTES);
    changed[96] = 0;
    memcpy(changed + 97, golden_payload, sizeof(golden_payload));
    put_u16_le(changed + 10, 1);
    put_u16_le(changed + 12, 97);
    set_record_crc(changed, sizeof(changed));
    CHECK_STATUS(lecore_format_check_v1(changed, sizeof(changed)), LECORE_OK);
    return EXIT_SUCCESS;
}

static int test_compatibility(void)
{
    lecore_format_descriptor_v1 expected;
    lecore_format_descriptor_v1 actual;

    make_descriptor(&expected);
    actual = expected;
    actual.format_minor = 1;
    actual.header_bytes = 97;
    actual.content_crc64 = UINT64_C(123);
    CHECK_STATUS(lecore_format_check_compatibility_v1(&expected, &actual), LECORE_OK);

    actual = expected;
    actual.semantic_profile = LECORE_PROFILE_HRR_F32_V1;
    actual.scalar_type = LECORE_SCALAR_F32;
    actual.payload_bytes = 8;
    CHECK_STATUS(lecore_format_check_compatibility_v1(&expected, &actual), LECORE_EPROFILE);
    actual = expected;
    actual.dimension = 1;
    actual.payload_bytes = 8;
    CHECK_STATUS(lecore_format_check_compatibility_v1(&expected, &actual), LECORE_EDIM);
    actual = expected;
    actual.application_contract[0] = 1;
    CHECK_STATUS(lecore_format_check_compatibility_v1(&expected, &actual), LECORE_EFORMAT);
    actual = expected;
    actual.seed_namespace = 1;
    CHECK_STATUS(lecore_format_check_compatibility_v1(&expected, &actual), LECORE_EFORMAT);
    return EXIT_SUCCESS;
}

int main(void)
{
    CHECK(test_golden_and_roundtrip() == EXIT_SUCCESS);
    CHECK(test_descriptor_errors() == EXIT_SUCCESS);
    CHECK(test_record_errors_and_forward_minor() == EXIT_SUCCESS);
    CHECK(test_compatibility() == EXIT_SUCCESS);
    return EXIT_SUCCESS;
}
