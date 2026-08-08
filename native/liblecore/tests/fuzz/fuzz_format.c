#include <lecore/lecore_format.h>

#include <stddef.h>
#include <stdint.h>

#define FUZZ_RECORD_CAPACITY ((size_t)512)
#define FUZZ_PAYLOAD_CAPACITY ((size_t)256)

static uint8_t fuzz_byte(const uint8_t *data, size_t size, size_t index)
{
    return size == 0 ? UINT8_C(0) : data[index % size];
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    lecore_format_descriptor_v1 descriptor;
    lecore_format_descriptor_v1 decoded;
    uint8_t payload[FUZZ_PAYLOAD_CAPACITY];
    uint8_t record[FUZZ_RECORD_CAPACITY];
    const void *decoded_payload = NULL;
    size_t decoded_payload_bytes = 0;
    size_t record_bytes = 0;
    size_t input_bytes = size < FUZZ_RECORD_CAPACITY
        ? size
        : FUZZ_RECORD_CAPACITY;
    size_t index;

    (void)lecore_format_check_v1(data, size);
    (void)lecore_format_decode_v1(
        data, size, &decoded, &decoded_payload, &decoded_payload_bytes);

    lecore_format_descriptor_init_v1(&descriptor);
    descriptor.artifact_kind = (lecore_artifact_kind)(
        UINT32_C(1) + (uint32_t)(fuzz_byte(data, size, 0) % UINT8_C(3)));
    descriptor.semantic_profile = (fuzz_byte(data, size, 1) & UINT8_C(1)) != 0
        ? LECORE_PROFILE_HRR_F32_V1
        : LECORE_PROFILE_HRR_F64_V1;
    descriptor.scalar_type = descriptor.semantic_profile ==
            LECORE_PROFILE_HRR_F32_V1
        ? LECORE_SCALAR_F32
        : LECORE_SCALAR_F64;
    descriptor.dimension = UINT32_C(1) +
        (uint32_t)(fuzz_byte(data, size, 2) % UINT8_C(8));
    descriptor.vector_count = UINT64_C(1) +
        (uint64_t)(fuzz_byte(data, size, 3) % UINT8_C(4));
    descriptor.payload_bytes = descriptor.vector_count *
        (uint64_t)descriptor.dimension *
        (descriptor.scalar_type == LECORE_SCALAR_F64 ? UINT64_C(8) : UINT64_C(4));

    for (index = 0; index < FUZZ_PAYLOAD_CAPACITY; ++index) {
        payload[index] = fuzz_byte(data, size, index + (size_t)4);
    }
    for (index = 0; index < input_bytes; ++index) {
        record[index] = data[index];
    }
    if (descriptor.payload_bytes <= FUZZ_PAYLOAD_CAPACITY) {
        (void)lecore_format_encoded_size_v1(&descriptor, &record_bytes);
        (void)lecore_format_encode_v1(
            &descriptor,
            payload,
            (size_t)descriptor.payload_bytes,
            record,
            sizeof(record),
            &record_bytes);
        if (record_bytes <= sizeof(record)) {
            (void)lecore_format_check_v1(record, record_bytes);
            if (lecore_format_decode_v1(
                    record,
                    record_bytes,
                    &decoded,
                    &decoded_payload,
                    &decoded_payload_bytes) == LECORE_OK) {
                (void)lecore_format_check_compatibility_v1(
                    &descriptor, &decoded);
            }
        }
    }
    return 0;
}
