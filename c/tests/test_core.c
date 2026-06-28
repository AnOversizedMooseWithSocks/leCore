#include "holo_core.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM 256U

static void require(int ok, const char *msg)
{
    if (!ok) {
        fprintf(stderr, "test_core: %s\n", msg);
        exit(1);
    }
}

static void require_ok(int rc, const char *msg)
{
    if (rc != HOLO_OK) {
        fprintf(stderr, "test_core: %s: %s\n", msg, holo_strerror(rc));
        exit(1);
    }
}

int main(void)
{
    holo_engine *engine = holo_engine_create(DIM, 42);
    double a[DIM];
    double a2[DIM];
    double b[DIM];
    double pair[DIM];
    double pair_freq_real[DIM];
    double pair_freq_imag[DIM];
    double accum_freq_real[DIM] = {0.0};
    double accum_freq_imag[DIM] = {0.0};
    double materialized_pair[DIM];
    double recovered[DIM];
    double matrix[4 * DIM];
    double fixed_many[4 * DIM];
    double scalar_pair[DIM];
    uint64_t labels[4] = {10, 11, 12, 13};
    holo_match matches[2];
    double noisy[DIM];
    double weights[2] = {0.25, 0.75};
    double bundle[DIM];
    double norms[4];
    size_t i;

    require(engine != NULL, "engine create");
    require(holo_engine_dim(engine) == DIM, "engine dim");

    require_ok(holo_keygen(engine, 7, a), "keygen a");
    require_ok(holo_keygen(engine, 7, a2), "keygen a2");
    require(fabs(holo_cosine(DIM, a, a2) - 1.0) < 1e-12, "keygen deterministic");
    require(fabs(holo_norm(DIM, a) - 1.0) < 1e-12, "keygen unit norm");

    require_ok(holo_keygen_unitary(engine, 100, a), "unitary key");
    require_ok(holo_keygen(engine, 200, b), "value key");
    require_ok(holo_bind(engine, a, b, pair), "bind");
    require_ok(holo_unbind(engine, pair, a, recovered), "unbind");
    require(holo_cosine(DIM, b, recovered) > 0.999999, "unitary bind/unbind roundtrip");
    require_ok(holo_spectrum_from_real(engine, pair, pair_freq_real, pair_freq_imag), "pair spectrum");
    require_ok(holo_unbind_spectrum(engine, pair_freq_real, pair_freq_imag, a, recovered),
               "unbind spectrum");
    require(holo_cosine(DIM, b, recovered) > 0.999999, "spectrum unbind roundtrip");
    require_ok(holo_bind_spectrum_accumulate(engine,
                                             a,
                                             b,
                                             1.0,
                                             accum_freq_real,
                                             accum_freq_imag),
               "bind spectrum accumulate");
    require_ok(holo_real_from_spectrum(engine,
                                       accum_freq_real,
                                       accum_freq_imag,
                                       materialized_pair),
               "real from spectrum");
    for (i = 0; i < DIM; ++i) {
        require(fabs(materialized_pair[i] - pair[i]) < 1e-10,
                "spectrum-accumulated bind materializes to scalar bind");
    }

    for (i = 0; i < 4; ++i) {
        require_ok(holo_keygen(engine, 1000 + i, matrix + i * DIM), "matrix key");
    }
    require_ok(holo_bind_fixed_many(engine, a, matrix, 4, fixed_many), "bind fixed many");
    for (i = 0; i < 4; ++i) {
        size_t j;
        require_ok(holo_bind(engine, a, matrix + i * DIM, scalar_pair), "scalar bind for fixed many");
        for (j = 0; j < DIM; ++j) {
            require(fabs(fixed_many[i * DIM + j] - scalar_pair[j]) < 1e-10,
                    "bind fixed many matches scalar bind");
        }
    }
    for (i = 0; i < DIM; ++i) {
        noisy[i] = matrix[2 * DIM + i] + 0.05 * a[i];
    }
    require_ok(holo_cleanup_topk(DIM, noisy, matrix, labels, 4, 2, matches), "cleanup");
    require(matches[0].label == 12, "cleanup top label");
    require(matches[0].score > matches[1].score, "cleanup ordering");
    for (i = 0; i < 4; ++i) {
        norms[i] = holo_norm(DIM, matrix + i * DIM);
    }
    require_ok(holo_cleanup_topk_with_norms(DIM, noisy, matrix, norms, labels, 4, 2, matches),
               "cleanup with norms");
    require(matches[0].label == 12, "cleanup with norms top label");

    require_ok(holo_bundle(DIM, matrix, weights, 2, bundle), "weighted bundle");
    require(fabs(holo_norm(DIM, bundle) - 1.0) < 1e-12, "bundle normalized");

    require_ok(holo_permute(DIM, matrix, 3, noisy), "permute");
    require(fabs(noisy[3] - matrix[0]) < 1e-12, "permute shift");

    holo_engine_destroy(engine);
    puts("test_core ok");
    return 0;
}
