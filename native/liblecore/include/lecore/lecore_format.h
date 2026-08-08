#ifndef LECORE_LECORE_FORMAT_H
#define LECORE_LECORE_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#include <lecore/lecore.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LECORE_FORMAT_MAJOR UINT16_C(1)
#define LECORE_FORMAT_MINOR UINT16_C(0)
#define LECORE_FORMAT_HEADER_BYTES UINT16_C(96)
#define LECORE_FORMAT_APPLICATION_CONTRACT_BYTES UINT32_C(16)

typedef uint32_t lecore_artifact_kind;
#define LECORE_ARTIFACT_VECTOR UINT32_C(1)
#define LECORE_ARTIFACT_MATRIX UINT32_C(2)
#define LECORE_ARTIFACT_TRACE UINT32_C(3)

typedef uint32_t lecore_normalization_policy;
#define LECORE_NORMALIZATION_RAW UINT32_C(0)
#define LECORE_NORMALIZATION_UNIT UINT32_C(1)
#define LECORE_NORMALIZATION_APPLICATION UINT32_C(2)

typedef uint32_t lecore_atom_scheme;
#define LECORE_ATOM_EXTERNAL UINT32_C(0)

/* No format flags are defined in version 1. */
#define LECORE_FORMAT_FLAGS_NONE UINT16_C(0)

/*
 * An in-memory view of the portable descriptor. This structure is not the
 * wire representation; the codec emits and consumes fields explicitly in
 * little-endian order.
 *
 * content_crc64 is populated by decode and ignored by encode, which always
 * recomputes it. struct_size must equal sizeof(lecore_format_descriptor_v1),
 * and reserved must remain zero when a descriptor is encoded.
 */
typedef struct lecore_format_descriptor_v1 {
    uint32_t struct_size;
    uint16_t format_major;
    uint16_t format_minor;
    uint16_t header_bytes;
    uint16_t flags;
    lecore_artifact_kind artifact_kind;
    lecore_profile semantic_profile;
    lecore_scalar_type scalar_type;
    uint32_t dimension;
    lecore_normalization_policy normalization_policy;
    lecore_atom_scheme atom_scheme;
    uint64_t seed_namespace;
    uint64_t vector_count;
    uint64_t payload_bytes;
    uint8_t application_contract[LECORE_FORMAT_APPLICATION_CONTRACT_BYTES];
    uint64_t content_crc64;
    uint64_t reserved[4];
} lecore_format_descriptor_v1;

/* Initialize a descriptor for current-format raw, externally supplied data. */
LECORE_API void LECORE_CALL lecore_format_descriptor_init_v1(
    lecore_format_descriptor_v1 *descriptor);

/*
 * Validate descriptor invariants and return header_bytes + payload_bytes.
 * The output is zeroed before validation and remains zero on failure. The
 * output scalar must not overlap the descriptor.
 */
LECORE_API lecore_status LECORE_CALL lecore_format_encoded_size_v1(
    const lecore_format_descriptor_v1 *descriptor,
    size_t *out_record_bytes);

/*
 * Encode one descriptor and payload into caller-owned storage. The payload is
 * copied verbatim and therefore must already contain little-endian IEEE-754
 * bits. The descriptor and payload may overlap the record; the codec snapshots
 * metadata and moves payload bytes before emitting the header. Output scalar
 * arguments must not overlap those buffers. out_record_bytes receives the
 * required size after descriptor validation, including when record_capacity
 * is too small.
 */
LECORE_API lecore_status LECORE_CALL lecore_format_encode_v1(
    const lecore_format_descriptor_v1 *descriptor,
    const void *payload,
    size_t payload_bytes,
    void *record,
    size_t record_capacity,
    size_t *out_record_bytes);

/*
 * Decode and fully validate an exact record, including CRC. On success,
 * out_payload points at the raw little-endian payload within record and
 * remains valid only as long as record. Output values are initialized to
 * empty values before record validation. The record and all three output
 * objects must occupy mutually disjoint storage.
 */
LECORE_API lecore_status LECORE_CALL lecore_format_decode_v1(
    const void *record,
    size_t record_bytes,
    lecore_format_descriptor_v1 *out_descriptor,
    const void **out_payload,
    size_t *out_payload_bytes);

/* Validate an exact record without returning a payload view. */
LECORE_API lecore_status LECORE_CALL lecore_format_check_v1(
    const void *record,
    size_t record_bytes);

/*
 * Check that two individually valid descriptors describe the same artifact
 * contract. Format minor/header length and CRC may differ; artifact shape,
 * profile, policies, namespace, and the 16 contract bytes must match exactly.
 */
LECORE_API lecore_status LECORE_CALL lecore_format_check_compatibility_v1(
    const lecore_format_descriptor_v1 *expected,
    const lecore_format_descriptor_v1 *actual);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* LECORE_LECORE_FORMAT_H */
