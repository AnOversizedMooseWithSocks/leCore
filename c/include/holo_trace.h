#ifndef HOLO_TRACE_H
#define HOLO_TRACE_H

#include "holo_core.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct holo_trace {
    uint64_t magic;
    holo_engine *engine;
    size_t dim;
    double *trace;
    double *work;
    double *spectrum_real;
    double *spectrum_imag;
    int real_valid;
    int spectrum_valid;
    uint64_t stored_count;
    double total_weight;
} holo_trace;

/* Stack traces should be zero-initialized before holo_trace_load().
   Operations are internally serialized; dispose/destroy still require normal
   ownership, with no concurrent users of the object being destroyed. */
holo_trace *holo_trace_create(holo_engine *engine);
void holo_trace_destroy(holo_trace *trace);

int holo_trace_init(holo_trace *trace, holo_engine *engine);
void holo_trace_dispose(holo_trace *trace);
int holo_trace_clear(holo_trace *trace);
int holo_trace_set(holo_trace *trace,
                   const double *values,
                   uint64_t stored_count,
                   double total_weight);
int holo_trace_copy(holo_trace *trace, double *out);

int holo_trace_store(holo_trace *trace,
                     const double *state,
                     const double *action,
                     double weight);
int holo_trace_recall(holo_trace *trace,
                      const double *query_state,
                      double *out_action_context);
int holo_trace_score_actions(holo_trace *trace,
                             const double *query_state,
                             const double *action_matrix,
                             const uint64_t *labels,
                             size_t action_count,
                             size_t k,
                             holo_match *out);
int holo_trace_score_actions_with_norms(holo_trace *trace,
                                        const double *query_state,
                                        const double *action_matrix,
                                        const double *action_norms,
                                        const uint64_t *labels,
                                        size_t action_count,
                                        size_t k,
                                        holo_match *out);
int holo_trace_query_index(holo_trace *trace,
                           const double *query_state,
                           const holo_action_index *index,
                           size_t k,
                           holo_match *out);

double holo_trace_fidelity(const holo_trace *trace);

int holo_trace_save(holo_trace *trace, const char *path);
int holo_trace_load(holo_trace *trace, holo_engine *engine, const char *path);

#ifdef __cplusplus
}
#endif

#endif
