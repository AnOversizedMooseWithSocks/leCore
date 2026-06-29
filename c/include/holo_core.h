#ifndef HOLO_CORE_H
#define HOLO_CORE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct holo_engine holo_engine;
typedef struct holo_action_index holo_action_index;

typedef struct holo_match {
    size_t index;
    uint64_t label;
    double score;
} holo_match;

enum {
    HOLO_OK = 0,
    HOLO_EINVAL = -1,
    HOLO_ENOMEM = -2,
    HOLO_ENOTPOW2 = -3,
    HOLO_EIO = -4,
    HOLO_EVERSION = -5
};

holo_engine *holo_engine_create(size_t dim, uint64_t seed);
void holo_engine_destroy(holo_engine *engine);

size_t holo_engine_dim(const holo_engine *engine);
uint64_t holo_engine_seed(const holo_engine *engine);
int holo_is_power_of_two(size_t n);
const char *holo_strerror(int status);

double holo_dot(size_t dim, const double *a, const double *b);
double holo_norm(size_t dim, const double *v);
double holo_cosine(size_t dim, const double *a, const double *b);
int holo_normalize(size_t dim, double *v);

int holo_keygen(holo_engine *engine, uint64_t id, double *out);
int holo_keygen_unitary(holo_engine *engine, uint64_t id, double *out);

int holo_bind(holo_engine *engine, const double *a, const double *b, double *out);
int holo_bind_spectrum_accumulate(holo_engine *engine,
                                  const double *a,
                                  const double *b,
                                  double weight,
                                  double *freq_real,
                                  double *freq_imag);
int holo_bind_fixed_many(holo_engine *engine,
                         const double *fixed,
                         const double *rows,
                         size_t count,
                         double *out);
int holo_unbind(holo_engine *engine, const double *pair, const double *key, double *out);
int holo_spectrum_from_real(holo_engine *engine,
                            const double *in,
                            double *freq_real,
                            double *freq_imag);
int holo_real_from_spectrum(holo_engine *engine,
                            const double *freq_real,
                            const double *freq_imag,
                            double *out);
int holo_unbind_spectrum(holo_engine *engine,
                         const double *pair_freq_real,
                         const double *pair_freq_imag,
                         const double *key,
                         double *out);
int holo_weighted_sum(size_t dim,
                      const double *vectors,
                      const double *weights,
                      size_t count,
                      double *out);
int holo_bundle(size_t dim,
                const double *vectors,
                const double *weights,
                size_t count,
                double *out);
int holo_permute(size_t dim, const double *in, long shift, double *out);

int holo_cleanup_topk(size_t dim,
                      const double *query,
                      const double *matrix,
                      const uint64_t *labels,
                      size_t count,
                      size_t k,
                      holo_match *out);
int holo_cleanup_topk_with_norms(size_t dim,
                                 const double *query,
                                 const double *matrix,
                                 const double *matrix_norms,
                                 const uint64_t *labels,
                                 size_t count,
                                 size_t k,
                                 holo_match *out);

holo_action_index *holo_action_index_create(size_t dim, size_t count);
void holo_action_index_destroy(holo_action_index *index);
size_t holo_action_index_dim(const holo_action_index *index);
size_t holo_action_index_count(const holo_action_index *index);
int holo_action_index_set(holo_action_index *index,
                          const double *vectors,
                          const uint64_t *labels);
int holo_action_index_search(const holo_action_index *index,
                             const double *query,
                             size_t k,
                             holo_match *out);

#ifdef __cplusplus
}
#endif

#endif
