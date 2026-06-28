#include "holo_trace.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 512U
#define ACTIONS 4U

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

int main(void)
{
    holo_engine *engine = holo_engine_create(DIM, 99);
    holo_engine *wrong_dim = NULL;
    holo_trace *heap_trace = NULL;
    holo_trace trace;
    holo_trace loaded;
    holo_trace rejected;
    double states[ACTIONS * DIM];
    double actions[ACTIONS * DIM];
    double action_norms[ACTIONS];
    double expected_trace[DIM] = {0.0};
    double pair[DIM];
    double copied[DIM];
    uint64_t labels[ACTIONS] = {1, 2, 3, 4};
    holo_match match[1];
    size_t i;

    require(engine != NULL, "engine create");
    require_ok(holo_trace_init(&trace, engine), "trace init");
    heap_trace = holo_trace_create(engine);
    require(heap_trace != NULL, "heap trace create");
    holo_trace_destroy(heap_trace);
    rejected.engine = NULL;
    rejected.trace = NULL;
    rejected.work = NULL;
    rejected.spectrum_real = NULL;
    rejected.spectrum_imag = NULL;

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
    }

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
    holo_trace_dispose(&loaded);
    holo_trace_dispose(&trace);
    holo_engine_destroy(wrong_dim);
    holo_engine_destroy(engine);
    puts("test_trace ok");
    return 0;
}
