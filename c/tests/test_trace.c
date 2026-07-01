#include "holo_trace.h"

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 512U
#define ACTIONS 4U
#define TRACE_THREAD_STORES 50U

typedef struct trace_thread_case {
    holo_trace *trace;
    const double *state;
    const double *action;
    int failed;
} trace_thread_case;

static void require(int ok, const char *msg)
{
    if (!ok) {
        fprintf(stderr, "test_trace: %s\n", msg);
        exit(1);
    }
}

static void require_ok(int rc, const char *msg)
{
    if (rc != HOLO_OK) {
        fprintf(stderr, "test_trace: %s: %s\n", msg, holo_strerror(rc));
        exit(1);
    }
}

static void *trace_store_worker(void *opaque)
{
    trace_thread_case *tc = (trace_thread_case *)opaque;
    for (size_t i = 0; i < TRACE_THREAD_STORES && !tc->failed; ++i) {
        if (holo_trace_store(tc->trace, tc->state, tc->action, 1.0) != HOLO_OK) {
            tc->failed = 1;
        }
    }
    return NULL;
}

int main(void)
{
    holo_engine *engine = holo_engine_create(DIM, 99);
    holo_engine *wrong_dim = NULL;
    holo_action_index *action_index = NULL;
    holo_action_index *wrong_index = NULL;
    holo_trace *heap_trace = NULL;
    holo_trace trace = {0};
    holo_trace loaded = {0};
    holo_trace rejected = {0};
    holo_trace threaded = {0};
    double states[ACTIONS * DIM];
    double actions[ACTIONS * DIM];
    double action_norms[ACTIONS];
    double expected_trace[DIM] = {0.0};
    double pair[DIM];
    double copied[DIM];
    uint64_t labels[ACTIONS] = {1, 2, 3, 4};
    holo_match match[1];
    pthread_t threads[ACTIONS];
    trace_thread_case thread_cases[ACTIONS];
    size_t i;

    require(engine != NULL, "engine create");
    require_ok(holo_trace_init(&trace, engine), "trace init");
    heap_trace = holo_trace_create(engine);
    require(heap_trace != NULL, "heap trace create");
    holo_trace_destroy(heap_trace);

    for (i = 0; i < ACTIONS; ++i) {
        require_ok(holo_keygen_unitary(engine, 100 + i, states + i * DIM), "state key");
        require_ok(holo_keygen(engine, 200 + i, actions + i * DIM), "action key");
        action_norms[i] = holo_norm(DIM, actions + i * DIM);
        require_ok(holo_bind(engine, states + i * DIM, actions + i * DIM, pair), "expected pair");
        for (size_t j = 0; j < DIM; ++j) {
            expected_trace[j] += pair[j];
        }
        require_ok(holo_trace_store(&trace, states + i * DIM, actions + i * DIM, 1.0), "trace store");
    }
    action_index = holo_action_index_create(DIM, ACTIONS);
    require(action_index != NULL, "action index create");
    require_ok(holo_action_index_set(action_index, actions, labels), "action index set");
    require(holo_action_index_dim(action_index) == DIM, "action index dim");
    require(holo_action_index_count(action_index) == ACTIONS, "action index count");

    require(trace.stored_count == ACTIONS, "stored count");
    require(fabs(holo_trace_fidelity(&trace) - 0.5) < 1e-12, "fidelity");
    require(trace.spectrum_valid == 1, "store keeps trace spectrum valid");
    require(trace.real_valid == 0, "store invalidates lazy real trace");
    require_ok(holo_trace_copy(&trace, copied), "lazy trace copy");
    require(trace.real_valid == 1, "copy materializes lazy real trace");
    for (i = 0; i < DIM; ++i) {
        require(fabs(copied[i] - expected_trace[i]) < 1e-9,
                "lazy real trace matches accumulated scalar binds");
    }

    for (i = 0; i < ACTIONS; ++i) {
        require_ok(holo_trace_score_actions(&trace,
                                            states + i * DIM,
                                            actions,
                                            labels,
                                            ACTIONS,
                                            1,
                                            match),
                   "score actions");
        require(match[0].label == labels[i], "trace recalls matching action");
        require(match[0].score > 0.35, "trace recall margin");
        require_ok(holo_trace_score_actions_with_norms(&trace,
                                                       states + i * DIM,
                                                       actions,
                                                       action_norms,
                                                       labels,
                                                       ACTIONS,
                                                       1,
                                                       match),
                   "score actions with norms");
        require(match[0].label == labels[i], "trace with norms recalls matching action");
        require_ok(holo_trace_query_index(&trace,
                                          states + i * DIM,
                                          action_index,
                                          1,
                                          match),
                   "trace query action index");
        require(match[0].label == labels[i], "trace action index recalls matching action");
    }
    wrong_index = holo_action_index_create(DIM / 2U, ACTIONS);
    require(wrong_index != NULL, "wrong-dim action index create");
    require(holo_trace_query_index(&trace, states, wrong_index, 1, match) == HOLO_EINVAL,
            "wrong-dim action index rejected");

    require_ok(holo_trace_init(&threaded, engine), "threaded trace init");
    for (i = 0; i < ACTIONS; ++i) {
        thread_cases[i].trace = &threaded;
        thread_cases[i].state = states + i * DIM;
        thread_cases[i].action = actions + i * DIM;
        thread_cases[i].failed = 0;
        require(pthread_create(&threads[i], NULL, trace_store_worker, &thread_cases[i]) == 0,
                "trace thread create");
    }
    for (i = 0; i < ACTIONS; ++i) {
        require(pthread_join(threads[i], NULL) == 0, "trace thread join");
        require(!thread_cases[i].failed, "shared trace concurrent store");
    }
    require(threaded.stored_count == ACTIONS * TRACE_THREAD_STORES,
            "shared trace concurrent stored count");
    require_ok(holo_trace_query_index(&threaded, states, action_index, 1, match),
               "shared trace concurrent query");
    require(match[0].label == labels[0], "shared trace concurrent recall");

    require_ok(holo_trace_save(&trace, "build/test_trace.htr"), "trace save");
    wrong_dim = holo_engine_create(DIM / 2U, 99);
    require(wrong_dim != NULL, "wrong-dim engine create");
    require(holo_trace_load(&rejected, wrong_dim, "build/test_trace.htr") == HOLO_EVERSION,
            "wrong-dim snapshot rejected");
    require(rejected.trace == NULL && rejected.work == NULL &&
                rejected.spectrum_real == NULL && rejected.spectrum_imag == NULL,
            "rejected load leaves no buffers");

    require_ok(holo_trace_load(&loaded, engine, "build/test_trace.htr"), "trace load");
    require(loaded.stored_count == trace.stored_count, "loaded count");
    require(fabs(loaded.total_weight - trace.total_weight) < 1e-12, "loaded weight");
    require_ok(holo_trace_load(&loaded, engine, "build/test_trace.htr"),
               "trace reload disposes previous buffers");
    require(loaded.stored_count == trace.stored_count, "reloaded count");

    for (i = 0; i < ACTIONS; ++i) {
        holo_match m2[1];
        require_ok(holo_trace_score_actions(&loaded,
                                            states + i * DIM,
                                            actions,
                                            labels,
                                            ACTIONS,
                                            1,
                                            m2),
                   "loaded score actions");
        require(m2[0].label == labels[i], "loaded trace recalls matching action");
    }
    require_ok(holo_trace_copy(&loaded, copied), "trace copy");
    require(fabs(holo_cosine(DIM, loaded.trace, copied) - 1.0) < 1e-12, "trace copy parity");
    require_ok(holo_trace_set(&loaded, copied, loaded.stored_count, loaded.total_weight), "trace set");
    require_ok(holo_trace_score_actions_with_norms(&loaded,
                                                   states,
                                                   actions,
                                                   action_norms,
                                                   labels,
                                                   ACTIONS,
                                                   1,
                                                   match),
               "loaded score after trace set");
    require(match[0].label == labels[0], "trace set keeps recall");

    remove("build/test_trace.htr");
    holo_trace_dispose(&threaded);
    holo_action_index_destroy(wrong_index);
    holo_action_index_destroy(action_index);
    holo_trace_dispose(&loaded);
    holo_trace_dispose(&trace);
    holo_engine_destroy(wrong_dim);
    holo_engine_destroy(engine);
    puts("test_trace ok");
    return 0;
}
