#include "nosqlite_kernel.h"

#include <stdatomic.h>

static atomic_uint_fast64_t NOSQLITE_ID = 1;

uint64_t nosqlite_fnv1a64(const uint8_t *data, size_t len) {
    uint64_t hash = 1469598103934665603ULL;

    for (size_t i = 0; i < len; i++) {
        hash ^= (uint64_t)data[i];
        hash *= 1099511628211ULL;
    }

    return hash;
}

uint64_t nosqlite_next_id(void) {
    return atomic_fetch_add_explicit(&NOSQLITE_ID, 1, memory_order_relaxed);
}
