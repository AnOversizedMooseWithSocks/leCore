#include "holo_trace.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#if defined(HOLO_USE_ACCELERATE) && HOLO_USE_ACCELERATE
#define HOLO_BENCH_RUNTIME "c_accelerate_norms"
#else
#define HOLO_BENCH_RUNTIME "c_scalar_norms"
#endif

static double now_seconds(void)
{
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
#endif
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static int bench_one(size_t dim, size_t pairs, size_t actions_n, size_t queries)
{
    holo_engine *engine = NULL;
    holo_trace trace;
    double *states = NULL;
    double *actions = NULL;
    double *action_norms = NULL;
    uint64_t *labels = NULL;
    holo_match match[1];
    double t0;
    double store_seconds;
    double query_seconds;
    size_t i;
    size_t correct = 0;

    engine = holo_engine_create(dim, 1234);
    if (!engine) {
        fprintf(stderr, "failed to create engine dim=%zu\n", dim);
        return 1;
    }
    states = (double *)calloc(pairs * dim, sizeof(*states));
    actions = (double *)calloc(actions_n * dim, sizeof(*actions));
    action_norms = (double *)calloc(actions_n, sizeof(*action_norms));
    labels = (uint64_t *)calloc(actions_n, sizeof(*labels));
    if (!states || !actions || !action_norms || !labels || holo_trace_init(&trace, engine) != HOLO_OK) {
        fprintf(stderr, "allocation/init failed\n");
        free(states);
        free(actions);
        free(action_norms);
        free(labels);
        holo_engine_destroy(engine);
        return 1;
    }

    for (i = 0; i < pairs; ++i) {
        if (holo_keygen_unitary(engine, 10000 + (uint64_t)i, states + i * dim) != HOLO_OK) {
            return 1;
        }
    }
    for (i = 0; i < actions_n; ++i) {
        labels[i] = (uint64_t)i;
        if (holo_keygen(engine, 20000 + (uint64_t)i, actions + i * dim) != HOLO_OK) {
            return 1;
        }
        action_norms[i] = holo_norm(dim, actions + i * dim);
    }

    t0 = now_seconds();
    for (i = 0; i < pairs; ++i) {
        if (holo_trace_store(&trace,
                             states + i * dim,
                             actions + (i % actions_n) * dim,
                             1.0) != HOLO_OK) {
            return 1;
        }
    }
    store_seconds = now_seconds() - t0;

    t0 = now_seconds();
    for (i = 0; i < queries; ++i) {
        const size_t j = i % pairs;
        if (holo_trace_score_actions_with_norms(&trace,
                                                states + j * dim,
                                                actions,
                                                action_norms,
                                                labels,
                                                actions_n,
                                                1,
                                                match) != HOLO_OK) {
            return 1;
        }
        correct += match[0].label == (uint64_t)(j % actions_n);
    }
    query_seconds = now_seconds() - t0;

    printf("{\"runtime\":\"%s\",\"dim\":%zu,\"pairs\":%zu,\"actions\":%zu,"
           "\"queries\":%zu,\"store_seconds\":%.9f,\"query_seconds\":%.9f,"
           "\"stores_per_second\":%.3f,\"queries_per_second\":%.3f,"
           "\"accuracy\":%.6f}\n",
           HOLO_BENCH_RUNTIME,
           dim,
           pairs,
           actions_n,
           queries,
           store_seconds,
           query_seconds,
           store_seconds > 0.0 ? (double)pairs / store_seconds : 0.0,
           query_seconds > 0.0 ? (double)queries / query_seconds : 0.0,
           queries ? (double)correct / (double)queries : 0.0);

    holo_trace_dispose(&trace);
    free(states);
    free(actions);
    free(action_norms);
    free(labels);
    holo_engine_destroy(engine);
    return 0;
}

int main(int argc, char **argv)
{
    const size_t dim = argc > 1 ? (size_t)strtoull(argv[1], NULL, 10) : 512U;
    const size_t pairs = argc > 2 ? (size_t)strtoull(argv[2], NULL, 10) : 16U;
    const size_t actions = argc > 3 ? (size_t)strtoull(argv[3], NULL, 10) : 16U;
    const size_t queries = argc > 4 ? (size_t)strtoull(argv[4], NULL, 10) : 512U;
    return bench_one(dim, pairs, actions, queries);
}
