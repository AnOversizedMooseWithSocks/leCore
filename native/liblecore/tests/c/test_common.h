#ifndef LECORE_TEST_COMMON_H
#define LECORE_TEST_COMMON_H

#include <lecore/lecore.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(condition)                                                        \
    do {                                                                        \
        if (!(condition)) {                                                     \
            fprintf(stderr, "%s:%d: check failed: %s\n",                     \
                    __FILE__, __LINE__, #condition);                            \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

#define CHECK_STATUS(expression, expected)                                      \
    do {                                                                        \
        const lecore_status actual_status_ = (expression);                      \
        if (actual_status_ != (expected)) {                                     \
            fprintf(stderr,                                                     \
                    "%s:%d: %s returned %u (%s), expected %u (%s)\n",          \
                    __FILE__, __LINE__, #expression,                            \
                    (unsigned)actual_status_,                                   \
                    lecore_status_string(actual_status_),                       \
                    (unsigned)(expected), lecore_status_string(expected));       \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    } while (0)

static inline int lecore_test_close_f64(double actual, double expected, double tolerance)
{
    return fabs(actual - expected) <= tolerance;
}

static inline int lecore_test_close_f32(float actual, float expected, float tolerance)
{
    return fabsf(actual - expected) <= tolerance;
}

static inline lecore_context *lecore_test_context(
    uint32_t dimension,
    lecore_profile profile,
    lecore_validation validation)
{
    lecore_config_v0 config;
    lecore_context *context = NULL;

    lecore_config_init_v0(&config);
    config.dimension = dimension;
    config.profile = profile;
    config.validation = validation;
    if (lecore_context_create(&config, &context) != LECORE_OK) {
        return NULL;
    }
    return context;
}

#endif /* LECORE_TEST_COMMON_H */
