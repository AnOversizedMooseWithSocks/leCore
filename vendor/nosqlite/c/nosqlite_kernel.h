#ifndef NOSQLITE_KERNEL_H
#define NOSQLITE_KERNEL_H

#include <stdint.h>
#include <stddef.h>

uint64_t nosqlite_fnv1a64(const uint8_t *data, size_t len);
uint64_t nosqlite_next_id(void);

#endif
