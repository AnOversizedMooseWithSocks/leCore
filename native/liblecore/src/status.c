#include <lecore/lecore.h>

uint32_t LECORE_CALL lecore_abi_version(void)
{
    return LECORE_ABI_VERSION;
}

uint32_t LECORE_CALL lecore_isa_version(void)
{
    return LECORE_ISA_VERSION;
}

const char *LECORE_CALL lecore_version_string(void)
{
    return LECORE_VERSION_STRING_VALUE;
}

const char *LECORE_CALL lecore_status_string(lecore_status status)
{
    switch (status) {
    case LECORE_OK:
        return "ok";
    case LECORE_EINVAL:
        return "invalid argument";
    case LECORE_EDIM:
        return "invalid dimension or stride";
    case LECORE_EPROFILE:
        return "profile mismatch or unknown profile";
    case LECORE_EBACKEND:
        return "unknown backend";
    case LECORE_EOVERFLOW:
        return "size arithmetic overflow";
    case LECORE_ENOMEM:
        return "allocation failed";
    case LECORE_EUNSUPPORTED:
        return "unsupported operation or backend";
    case LECORE_ENONFINITE:
        return "non-finite input";
    case LECORE_EFORMAT:
        return "invalid interchange format";
    case LECORE_ECHECKSUM:
        return "checksum mismatch";
    default:
        return "unknown status";
    }
}
