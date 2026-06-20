#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include "holo_trace.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define HOLO_TRACE_MAGIC "HOLOTRC"
#define HOLO_TRACE_VERSION 1U
#define HOLO_TRACE_ENDIAN UINT32_C(0x01020304)
#define HOLO_ALIGN 64U

typedef struct holo_trace_header {
    char magic[8];
    uint32_t version;
    uint32_t endian;
    uint64_t dim;
    uint64_t stored_count;
    double total_weight;
} holo_trace_header;

static uint64_t checksum_trace(const double *v, size_t n)
{
    uint64_t h = UINT64_C(1469598103934665603);
    size_t i;
    for (i = 0; i < n; ++i) {
        uint64_t bits;
        memcpy(&bits, &v[i], sizeof(bits));
        h ^= bits;
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static void *alloc_zeroed(size_t count, size_t size)
{
    void *ptr = NULL;
    const size_t bytes = count * size;
    if (count != 0 && size > ((size_t)-1) / count) {
        return NULL;
    }
#if defined(_MSC_VER)
    ptr = _aligned_malloc(bytes, HOLO_ALIGN);
    if (!ptr) {
        return NULL;
    }
#elif defined(__APPLE__) || defined(__unix__)
    if (posix_memalign(&ptr, HOLO_ALIGN, bytes) != 0) {
        return NULL;
    }
#else
    ptr = malloc(bytes);
    if (!ptr) {
        return NULL;
    }
#endif
    memset(ptr, 0, bytes);
    return ptr;
}

static void free_aligned(void *ptr)
{
#if defined(_MSC_VER)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

holo_trace *holo_trace_create(holo_engine *engine)
{
    holo_trace *trace;
    int rc;
    if (!engine) {
        return NULL;
    }
    trace = (holo_trace *)calloc(1, sizeof(*trace));
    if (!trace) {
        return NULL;
    }
    rc = holo_trace_init(trace, engine);
    if (rc != HOLO_OK) {
        free(trace);
        return NULL;
    }
    return trace;
}

void holo_trace_destroy(holo_trace *trace)
{
    if (!trace) {
        return;
    }
    holo_trace_dispose(trace);
    free(trace);
}

int holo_trace_init(holo_trace *trace, holo_engine *engine)
{
    size_t dim;
    if (!trace || !engine) {
        return HOLO_EINVAL;
    }
    dim = holo_engine_dim(engine);
    memset(trace, 0, sizeof(*trace));
    trace->trace = (double *)alloc_zeroed(dim, sizeof(*trace->trace));
    trace->work = (double *)alloc_zeroed(dim, sizeof(*trace->work));
    trace->spectrum_real = (double *)alloc_zeroed(dim, sizeof(*trace->spectrum_real));
    trace->spectrum_imag = (double *)alloc_zeroed(dim, sizeof(*trace->spectrum_imag));
    if (!trace->trace || !trace->work || !trace->spectrum_real || !trace->spectrum_imag) {
        holo_trace_dispose(trace);
        return HOLO_ENOMEM;
    }
    trace->engine = engine;
    trace->dim = dim;
    return HOLO_OK;
}

void holo_trace_dispose(holo_trace *trace)
{
    if (!trace) {
        return;
    }
    free_aligned(trace->trace);
    free_aligned(trace->work);
    free_aligned(trace->spectrum_real);
    free_aligned(trace->spectrum_imag);
    memset(trace, 0, sizeof(*trace));
}

int holo_trace_clear(holo_trace *trace)
{
    if (!trace || !trace->trace || !trace->spectrum_real || !trace->spectrum_imag) {
        return HOLO_EINVAL;
    }
    memset(trace->trace, 0, trace->dim * sizeof(trace->trace[0]));
    memset(trace->spectrum_real, 0, trace->dim * sizeof(trace->spectrum_real[0]));
    memset(trace->spectrum_imag, 0, trace->dim * sizeof(trace->spectrum_imag[0]));
    trace->spectrum_valid = 1;
    trace->stored_count = 0;
    trace->total_weight = 0.0;
    return HOLO_OK;
}

int holo_trace_set(holo_trace *trace,
                   const double *values,
                   uint64_t stored_count,
                   double total_weight)
{
    if (!trace || !trace->trace || !trace->spectrum_real || !trace->spectrum_imag || !values) {
        return HOLO_EINVAL;
    }
    memcpy(trace->trace, values, trace->dim * sizeof(trace->trace[0]));
    trace->spectrum_valid = 0;
    trace->stored_count = stored_count;
    trace->total_weight = total_weight;
    return HOLO_OK;
}

int holo_trace_copy(const holo_trace *trace, double *out)
{
    if (!trace || !trace->trace || !out) {
        return HOLO_EINVAL;
    }
    memcpy(out, trace->trace, trace->dim * sizeof(out[0]));
    return HOLO_OK;
}

int holo_trace_store(holo_trace *trace,
                     const double *state,
                     const double *action,
                     double weight)
{
    size_t i;
    int rc;
    if (!trace || !trace->engine || !trace->trace || !trace->work || !state || !action) {
        return HOLO_EINVAL;
    }
    if (weight == 0.0) {
        return HOLO_OK;
    }
    rc = holo_bind(trace->engine, state, action, trace->work);
    if (rc != HOLO_OK) {
        return rc;
    }
    for (i = 0; i < trace->dim; ++i) {
        trace->trace[i] += weight * trace->work[i];
    }
    trace->spectrum_valid = 0;
    trace->stored_count += 1;
    trace->total_weight += weight;
    return HOLO_OK;
}

int holo_trace_recall(const holo_trace *trace,
                      const double *query_state,
                      double *out_action_context)
{
    holo_trace *mutable_trace;
    int rc;
    if (!trace || !trace->engine || !trace->trace || !query_state || !out_action_context) {
        return HOLO_EINVAL;
    }
    if (trace->stored_count == 0) {
        memset(out_action_context, 0, trace->dim * sizeof(out_action_context[0]));
        return HOLO_OK;
    }
    mutable_trace = (holo_trace *)trace;
    if (!mutable_trace->spectrum_valid) {
        rc = holo_spectrum_from_real(mutable_trace->engine,
                                     mutable_trace->trace,
                                     mutable_trace->spectrum_real,
                                     mutable_trace->spectrum_imag);
        if (rc != HOLO_OK) {
            return rc;
        }
        mutable_trace->spectrum_valid = 1;
    }
    return holo_unbind_spectrum(mutable_trace->engine,
                                mutable_trace->spectrum_real,
                                mutable_trace->spectrum_imag,
                                query_state,
                                out_action_context);
}

int holo_trace_score_actions(const holo_trace *trace,
                             const double *query_state,
                             const double *action_matrix,
                             const uint64_t *labels,
                             size_t action_count,
                             size_t k,
                             holo_match *out)
{
    return holo_trace_score_actions_with_norms(trace,
                                               query_state,
                                               action_matrix,
                                               NULL,
                                               labels,
                                               action_count,
                                               k,
                                               out);
}

int holo_trace_score_actions_with_norms(const holo_trace *trace,
                                        const double *query_state,
                                        const double *action_matrix,
                                        const double *action_norms,
                                        const uint64_t *labels,
                                        size_t action_count,
                                        size_t k,
                                        holo_match *out)
{
    int rc;
    if (!trace || !trace->work) {
        return HOLO_EINVAL;
    }
    rc = holo_trace_recall(trace, query_state, trace->work);
    if (rc != HOLO_OK) {
        return rc;
    }
    return holo_cleanup_topk_with_norms(trace->dim,
                                        trace->work,
                                        action_matrix,
                                        action_norms,
                                        labels,
                                        action_count,
                                        k,
                                        out);
}

double holo_trace_fidelity(const holo_trace *trace)
{
    if (!trace || trace->stored_count == 0) {
        return 0.0;
    }
    return 1.0 / sqrt((double)trace->stored_count);
}

int holo_trace_save(const holo_trace *trace, const char *path)
{
    FILE *fp;
    holo_trace_header header;
    uint64_t sum;
    if (!trace || !trace->trace || !path) {
        return HOLO_EINVAL;
    }
    memset(&header, 0, sizeof(header));
    memcpy(header.magic, HOLO_TRACE_MAGIC, sizeof(HOLO_TRACE_MAGIC));
    header.version = HOLO_TRACE_VERSION;
    header.endian = HOLO_TRACE_ENDIAN;
    header.dim = (uint64_t)trace->dim;
    header.stored_count = trace->stored_count;
    header.total_weight = trace->total_weight;

    fp = fopen(path, "wb");
    if (!fp) {
        return HOLO_EIO;
    }
    sum = checksum_trace(trace->trace, trace->dim);
    if (fwrite(&header, sizeof(header), 1, fp) != 1 ||
        fwrite(trace->trace, sizeof(double), trace->dim, fp) != trace->dim ||
        fwrite(&sum, sizeof(sum), 1, fp) != 1) {
        fclose(fp);
        return HOLO_EIO;
    }
    if (fclose(fp) != 0) {
        return HOLO_EIO;
    }
    return HOLO_OK;
}

int holo_trace_load(holo_trace *trace, holo_engine *engine, const char *path)
{
    FILE *fp;
    holo_trace_header header;
    uint64_t expected;
    uint64_t actual;
    int rc;
    if (!trace || !engine || !path) {
        return HOLO_EINVAL;
    }
    fp = fopen(path, "rb");
    if (!fp) {
        return HOLO_EIO;
    }
    if (fread(&header, sizeof(header), 1, fp) != 1) {
        fclose(fp);
        return HOLO_EIO;
    }
    if (memcmp(header.magic, HOLO_TRACE_MAGIC, sizeof(HOLO_TRACE_MAGIC)) != 0 ||
        header.version != HOLO_TRACE_VERSION ||
        header.endian != HOLO_TRACE_ENDIAN ||
        header.dim != (uint64_t)holo_engine_dim(engine)) {
        fclose(fp);
        return HOLO_EVERSION;
    }
    rc = holo_trace_init(trace, engine);
    if (rc != HOLO_OK) {
        fclose(fp);
        return rc;
    }
    trace->stored_count = header.stored_count;
    trace->total_weight = header.total_weight;
    trace->spectrum_valid = 0;
    if (fread(trace->trace, sizeof(double), trace->dim, fp) != trace->dim ||
        fread(&expected, sizeof(expected), 1, fp) != 1) {
        holo_trace_dispose(trace);
        fclose(fp);
        return HOLO_EIO;
    }
    actual = checksum_trace(trace->trace, trace->dim);
    if (actual != expected) {
        holo_trace_dispose(trace);
        fclose(fp);
        return HOLO_EVERSION;
    }
    if (fclose(fp) != 0) {
        holo_trace_dispose(trace);
        return HOLO_EIO;
    }
    return HOLO_OK;
}
