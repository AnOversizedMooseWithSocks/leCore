#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include "holo_core.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef HOLO_USE_ACCELERATE
#define HOLO_USE_ACCELERATE 0
#endif

#if HOLO_USE_ACCELERATE
#if !defined(__APPLE__)
#error "HOLO_USE_ACCELERATE requires Apple's Accelerate framework"
#endif
#include <Accelerate/Accelerate.h>
#endif

#define HOLO_PI 3.141592653589793238462643383279502884
#define HOLO_ALIGN 64U

#if !HOLO_USE_ACCELERATE
typedef struct holo_complex {
    double re;
    double im;
} holo_complex;
#endif

struct holo_engine {
    size_t dim;
    size_t log2_dim;
    uint64_t seed;
#if HOLO_USE_ACCELERATE
    FFTSetupD fft_setup;
    double *ar;
    double *ai;
    double *br;
    double *bi;
    DSPDoubleSplitComplex za;
    DSPDoubleSplitComplex zb;
#else
    holo_complex *a;
    holo_complex *b;
#endif
    double *real;
};

struct holo_action_index {
    size_t dim;
    size_t count;
    double *vectors;
    double *norms;
    uint64_t *labels;
};

static uint64_t splitmix64(uint64_t *x)
{
    uint64_t z;
    *x += UINT64_C(0x9e3779b97f4a7c15);
    z = *x;
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static uint64_t mix_id(uint64_t seed, uint64_t id)
{
    uint64_t x = seed ^ UINT64_C(0xd1b54a32d192ed03);
    x += id * UINT64_C(0x9e3779b97f4a7c15);
    (void)splitmix64(&x);
    return x;
}

static double u01(uint64_t *state)
{
    const uint64_t r = splitmix64(state) >> 11;
    return (double)r * (1.0 / 9007199254740992.0);
}

static double normal01(uint64_t *state)
{
    double u1 = u01(state);
    double u2 = u01(state);
    if (u1 < 1e-12) {
        u1 = 1e-12;
    }
    return sqrt(-2.0 * log(u1)) * cos(2.0 * HOLO_PI * u2);
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

int holo_is_power_of_two(size_t n)
{
    return n != 0 && (n & (n - 1)) == 0;
}

static size_t log2_size(size_t n)
{
    size_t log2 = 0;
    while (n > 1) {
        n >>= 1;
        ++log2;
    }
    return log2;
}

holo_engine *holo_engine_create(size_t dim, uint64_t seed)
{
    holo_engine *engine;
    if (!holo_is_power_of_two(dim)) {
        return NULL;
    }
    engine = (holo_engine *)calloc(1, sizeof(*engine));
    if (!engine) {
        return NULL;
    }
#if HOLO_USE_ACCELERATE
    engine->log2_dim = log2_size(dim);
    engine->fft_setup = vDSP_create_fftsetupD((vDSP_Length)engine->log2_dim, FFT_RADIX2);
    engine->ar = (double *)alloc_zeroed(dim, sizeof(*engine->ar));
    engine->ai = (double *)alloc_zeroed(dim, sizeof(*engine->ai));
    engine->br = (double *)alloc_zeroed(dim, sizeof(*engine->br));
    engine->bi = (double *)alloc_zeroed(dim, sizeof(*engine->bi));
    engine->real = (double *)alloc_zeroed(dim, sizeof(*engine->real));
    if (!engine->fft_setup || !engine->ar || !engine->ai || !engine->br || !engine->bi || !engine->real) {
        holo_engine_destroy(engine);
        return NULL;
    }
    engine->za.realp = engine->ar;
    engine->za.imagp = engine->ai;
    engine->zb.realp = engine->br;
    engine->zb.imagp = engine->bi;
#else
    engine->log2_dim = log2_size(dim);
    engine->a = (holo_complex *)alloc_zeroed(dim, sizeof(*engine->a));
    engine->b = (holo_complex *)alloc_zeroed(dim, sizeof(*engine->b));
    engine->real = (double *)alloc_zeroed(dim, sizeof(*engine->real));
    if (!engine->a || !engine->b || !engine->real) {
        holo_engine_destroy(engine);
        return NULL;
    }
#endif
    engine->dim = dim;
    engine->seed = seed;
    return engine;
}

void holo_engine_destroy(holo_engine *engine)
{
    if (!engine) {
        return;
    }
#if HOLO_USE_ACCELERATE
    if (engine->fft_setup) {
        vDSP_destroy_fftsetupD(engine->fft_setup);
    }
    free_aligned(engine->ar);
    free_aligned(engine->ai);
    free_aligned(engine->br);
    free_aligned(engine->bi);
#else
    free_aligned(engine->a);
    free_aligned(engine->b);
#endif
    free_aligned(engine->real);
    free(engine);
}

size_t holo_engine_dim(const holo_engine *engine)
{
    return engine ? engine->dim : 0;
}

uint64_t holo_engine_seed(const holo_engine *engine)
{
    return engine ? engine->seed : 0;
}

const char *holo_strerror(int status)
{
    switch (status) {
    case HOLO_OK:
        return "ok";
    case HOLO_EINVAL:
        return "invalid argument";
    case HOLO_ENOMEM:
        return "out of memory";
    case HOLO_ENOTPOW2:
        return "dimension is not a power of two";
    case HOLO_EIO:
        return "i/o error";
    case HOLO_EVERSION:
        return "unsupported version or incompatible snapshot";
    default:
        return "unknown holo error";
    }
}

double holo_dot(size_t dim, const double *a, const double *b)
{
#if HOLO_USE_ACCELERATE
    double s = 0.0;
    if (!a || !b || dim == 0) {
        return 0.0;
    }
    vDSP_dotprD(a, 1, b, 1, &s, (vDSP_Length)dim);
    return s;
#else
    double s = 0.0;
    size_t i;
    if (!a || !b) {
        return 0.0;
    }
    for (i = 0; i < dim; ++i) {
        s += a[i] * b[i];
    }
    return s;
#endif
}

double holo_norm(size_t dim, const double *v)
{
#if HOLO_USE_ACCELERATE
    double s = 0.0;
    if (!v || dim == 0) {
        return 0.0;
    }
    vDSP_svesqD(v, 1, &s, (vDSP_Length)dim);
    return sqrt(s);
#else
    return sqrt(holo_dot(dim, v, v));
#endif
}

double holo_cosine(size_t dim, const double *a, const double *b)
{
    const double na = holo_norm(dim, a);
    const double nb = holo_norm(dim, b);
    if (na <= 0.0 || nb <= 0.0) {
        return 0.0;
    }
    return holo_dot(dim, a, b) / (na * nb);
}

int holo_normalize(size_t dim, double *v)
{
    double n;
    size_t i;
    if (!v || dim == 0) {
        return HOLO_EINVAL;
    }
    n = holo_norm(dim, v);
    if (n <= 0.0) {
        return HOLO_EINVAL;
    }
    for (i = 0; i < dim; ++i) {
        v[i] /= n;
    }
    return HOLO_OK;
}

#if HOLO_USE_ACCELERATE
static void fft_split(holo_engine *engine, DSPDoubleSplitComplex *z, int inverse)
{
    const vDSP_Length n = (vDSP_Length)engine->dim;
    const FFTDirection direction = inverse ? FFT_INVERSE : FFT_FORWARD;
    vDSP_fft_zipD(engine->fft_setup, z, 1, (vDSP_Length)engine->log2_dim, direction);
    if (inverse) {
        const double scale = 1.0 / (double)engine->dim;
        vDSP_vsmulD(z->realp, 1, &scale, z->realp, 1, n);
        vDSP_vsmulD(z->imagp, 1, &scale, z->imagp, 1, n);
    }
}
#else
static void fft(holo_complex *x, size_t n, int inverse)
{
    size_t i;
    size_t j = 0;
    for (i = 1; i < n; ++i) {
        size_t bit = n >> 1;
        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            const holo_complex tmp = x[i];
            x[i] = x[j];
            x[j] = tmp;
        }
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        const double angle = (inverse ? 2.0 : -2.0) * HOLO_PI / (double)len;
        const double wlen_re = cos(angle);
        const double wlen_im = sin(angle);
        for (i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;
            const size_t half = len >> 1;
            for (j = 0; j < half; ++j) {
                holo_complex u = x[i + j];
                holo_complex v;
                double next_re;
                v.re = x[i + j + half].re * w_re - x[i + j + half].im * w_im;
                v.im = x[i + j + half].re * w_im + x[i + j + half].im * w_re;
                x[i + j].re = u.re + v.re;
                x[i + j].im = u.im + v.im;
                x[i + j + half].re = u.re - v.re;
                x[i + j + half].im = u.im - v.im;
                next_re = w_re * wlen_re - w_im * wlen_im;
                w_im = w_re * wlen_im + w_im * wlen_re;
                w_re = next_re;
            }
        }
    }

    if (inverse) {
        const double inv_n = 1.0 / (double)n;
        for (i = 0; i < n; ++i) {
            x[i].re *= inv_n;
            x[i].im *= inv_n;
        }
    }
}
#endif

int holo_keygen(holo_engine *engine, uint64_t id, double *out)
{
    uint64_t state;
    size_t i;
    if (!engine || !out) {
        return HOLO_EINVAL;
    }
    state = mix_id(engine->seed, id);
    for (i = 0; i < engine->dim; ++i) {
        out[i] = normal01(&state);
    }
    return holo_normalize(engine->dim, out);
}

int holo_keygen_unitary(holo_engine *engine, uint64_t id, double *out)
{
    uint64_t state;
    size_t i;
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !out) {
        return HOLO_EINVAL;
    }
    state = mix_id(engine->seed ^ UINT64_C(0xa0761d6478bd642f), id);
#if HOLO_USE_ACCELERATE
    memset(engine->ar, 0, n * sizeof(engine->ar[0]));
    memset(engine->ai, 0, n * sizeof(engine->ai[0]));
    engine->ar[0] = (splitmix64(&state) & 1U) ? 1.0 : -1.0;
    engine->ar[n / 2] = (splitmix64(&state) & 1U) ? 1.0 : -1.0;
    for (i = 1; i < n / 2; ++i) {
        const double theta = 2.0 * HOLO_PI * u01(&state);
        const double c = cos(theta);
        const double s = sin(theta);
        engine->ar[i] = c;
        engine->ai[i] = s;
        engine->ar[n - i] = c;
        engine->ai[n - i] = -s;
    }
    fft_split(engine, &engine->za, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->ar[i];
    }
#else
    memset(engine->a, 0, n * sizeof(engine->a[0]));
    engine->a[0].re = (splitmix64(&state) & 1U) ? 1.0 : -1.0;
    engine->a[n / 2].re = (splitmix64(&state) & 1U) ? 1.0 : -1.0;
    for (i = 1; i < n / 2; ++i) {
        const double theta = 2.0 * HOLO_PI * u01(&state);
        const double c = cos(theta);
        const double s = sin(theta);
        engine->a[i].re = c;
        engine->a[i].im = s;
        engine->a[n - i].re = c;
        engine->a[n - i].im = -s;
    }
    fft(engine->a, n, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->a[i].re;
    }
#endif
    return holo_normalize(n, out);
}

int holo_bind(holo_engine *engine, const double *a, const double *b, double *out)
{
    size_t i;
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !a || !b || !out) {
        return HOLO_EINVAL;
    }
#if HOLO_USE_ACCELERATE
    memcpy(engine->ar, a, n * sizeof(engine->ar[0]));
    memset(engine->ai, 0, n * sizeof(engine->ai[0]));
    memcpy(engine->br, b, n * sizeof(engine->br[0]));
    memset(engine->bi, 0, n * sizeof(engine->bi[0]));
    fft_split(engine, &engine->za, 0);
    fft_split(engine, &engine->zb, 0);
    vDSP_zvmulD(&engine->za, 1, &engine->zb, 1, &engine->za, 1, (vDSP_Length)n, 1);
    fft_split(engine, &engine->za, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->ar[i];
    }
#else
    for (i = 0; i < n; ++i) {
        engine->a[i].re = a[i];
        engine->a[i].im = 0.0;
        engine->b[i].re = b[i];
        engine->b[i].im = 0.0;
    }
    fft(engine->a, n, 0);
    fft(engine->b, n, 0);
    for (i = 0; i < n; ++i) {
        const double re = engine->a[i].re * engine->b[i].re - engine->a[i].im * engine->b[i].im;
        const double im = engine->a[i].re * engine->b[i].im + engine->a[i].im * engine->b[i].re;
        engine->a[i].re = re;
        engine->a[i].im = im;
    }
    fft(engine->a, n, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->a[i].re;
    }
#endif
    return HOLO_OK;
}

int holo_bind_spectrum_accumulate(holo_engine *engine,
                                  const double *a,
                                  const double *b,
                                  double weight,
                                  double *freq_real,
                                  double *freq_imag)
{
    size_t i;
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !a || !b || !freq_real || !freq_imag) {
        return HOLO_EINVAL;
    }
    if (weight == 0.0) {
        return HOLO_OK;
    }
#if HOLO_USE_ACCELERATE
    memcpy(engine->ar, a, n * sizeof(engine->ar[0]));
    memset(engine->ai, 0, n * sizeof(engine->ai[0]));
    memcpy(engine->br, b, n * sizeof(engine->br[0]));
    memset(engine->bi, 0, n * sizeof(engine->bi[0]));
    fft_split(engine, &engine->za, 0);
    fft_split(engine, &engine->zb, 0);
    vDSP_zvmulD(&engine->za, 1, &engine->zb, 1, &engine->za, 1, (vDSP_Length)n, 1);
    for (i = 0; i < n; ++i) {
        freq_real[i] += weight * engine->ar[i];
        freq_imag[i] += weight * engine->ai[i];
    }
#else
    for (i = 0; i < n; ++i) {
        engine->a[i].re = a[i];
        engine->a[i].im = 0.0;
        engine->b[i].re = b[i];
        engine->b[i].im = 0.0;
    }
    fft(engine->a, n, 0);
    fft(engine->b, n, 0);
    for (i = 0; i < n; ++i) {
        const double re = engine->a[i].re * engine->b[i].re - engine->a[i].im * engine->b[i].im;
        const double im = engine->a[i].re * engine->b[i].im + engine->a[i].im * engine->b[i].re;
        freq_real[i] += weight * re;
        freq_imag[i] += weight * im;
    }
#endif
    return HOLO_OK;
}

int holo_bind_fixed_many(holo_engine *engine,
                         const double *fixed,
                         const double *rows,
                         size_t count,
                         double *out)
{
    size_t row;
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !fixed || (!rows && count > 0) || (!out && count > 0)) {
        return HOLO_EINVAL;
    }
    if (count == 0) {
        return HOLO_OK;
    }
#if HOLO_USE_ACCELERATE
    memcpy(engine->ar, fixed, n * sizeof(engine->ar[0]));
    memset(engine->ai, 0, n * sizeof(engine->ai[0]));
    fft_split(engine, &engine->za, 0);
    for (row = 0; row < count; ++row) {
        const double *src = rows + row * n;
        double *dst = out + row * n;
        memcpy(engine->br, src, n * sizeof(engine->br[0]));
        memset(engine->bi, 0, n * sizeof(engine->bi[0]));
        fft_split(engine, &engine->zb, 0);
        vDSP_zvmulD(&engine->za, 1, &engine->zb, 1, &engine->zb, 1, (vDSP_Length)n, 1);
        fft_split(engine, &engine->zb, 1);
        memcpy(dst, engine->br, n * sizeof(dst[0]));
    }
#else
    size_t i;
    for (i = 0; i < n; ++i) {
        engine->a[i].re = fixed[i];
        engine->a[i].im = 0.0;
    }
    fft(engine->a, n, 0);
    for (row = 0; row < count; ++row) {
        const double *src = rows + row * n;
        double *dst = out + row * n;
        for (i = 0; i < n; ++i) {
            engine->b[i].re = src[i];
            engine->b[i].im = 0.0;
        }
        fft(engine->b, n, 0);
        for (i = 0; i < n; ++i) {
            const double re = engine->a[i].re * engine->b[i].re - engine->a[i].im * engine->b[i].im;
            const double im = engine->a[i].re * engine->b[i].im + engine->a[i].im * engine->b[i].re;
            engine->b[i].re = re;
            engine->b[i].im = im;
        }
        fft(engine->b, n, 1);
        for (i = 0; i < n; ++i) {
            dst[i] = engine->b[i].re;
        }
    }
#endif
    return HOLO_OK;
}

int holo_unbind(holo_engine *engine, const double *pair, const double *key, double *out)
{
    size_t i;
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !pair || !key || !out) {
        return HOLO_EINVAL;
    }
    engine->real[0] = key[0];
    for (i = 1; i < n; ++i) {
        engine->real[i] = key[n - i];
    }
    return holo_bind(engine, pair, engine->real, out);
}

int holo_spectrum_from_real(holo_engine *engine,
                            const double *in,
                            double *freq_real,
                            double *freq_imag)
{
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !in || !freq_real || !freq_imag) {
        return HOLO_EINVAL;
    }
#if HOLO_USE_ACCELERATE
    memcpy(engine->ar, in, n * sizeof(engine->ar[0]));
    memset(engine->ai, 0, n * sizeof(engine->ai[0]));
    fft_split(engine, &engine->za, 0);
    memcpy(freq_real, engine->ar, n * sizeof(freq_real[0]));
    memcpy(freq_imag, engine->ai, n * sizeof(freq_imag[0]));
#else
    size_t i;
    for (i = 0; i < n; ++i) {
        engine->a[i].re = in[i];
        engine->a[i].im = 0.0;
    }
    fft(engine->a, n, 0);
    for (i = 0; i < n; ++i) {
        freq_real[i] = engine->a[i].re;
        freq_imag[i] = engine->a[i].im;
    }
#endif
    return HOLO_OK;
}

int holo_real_from_spectrum(holo_engine *engine,
                            const double *freq_real,
                            const double *freq_imag,
                            double *out)
{
    size_t i;
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !freq_real || !freq_imag || !out) {
        return HOLO_EINVAL;
    }
#if HOLO_USE_ACCELERATE
    memcpy(engine->ar, freq_real, n * sizeof(engine->ar[0]));
    memcpy(engine->ai, freq_imag, n * sizeof(engine->ai[0]));
    fft_split(engine, &engine->za, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->ar[i];
    }
#else
    for (i = 0; i < n; ++i) {
        engine->a[i].re = freq_real[i];
        engine->a[i].im = freq_imag[i];
    }
    fft(engine->a, n, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->a[i].re;
    }
#endif
    return HOLO_OK;
}

int holo_unbind_spectrum(holo_engine *engine,
                         const double *pair_freq_real,
                         const double *pair_freq_imag,
                         const double *key,
                         double *out)
{
    size_t i;
    const size_t n = engine ? engine->dim : 0;
    if (!engine || !pair_freq_real || !pair_freq_imag || !key || !out) {
        return HOLO_EINVAL;
    }
#if HOLO_USE_ACCELERATE
    DSPDoubleSplitComplex pair_freq;
    pair_freq.realp = (double *)pair_freq_real;
    pair_freq.imagp = (double *)pair_freq_imag;
    memcpy(engine->br, key, n * sizeof(engine->br[0]));
    memset(engine->bi, 0, n * sizeof(engine->bi[0]));
    fft_split(engine, &engine->zb, 0);
    vDSP_zvmulD(&engine->zb, 1, &pair_freq, 1, &engine->za, 1, (vDSP_Length)n, -1);
    fft_split(engine, &engine->za, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->ar[i];
    }
#else
    for (i = 0; i < n; ++i) {
        engine->b[i].re = key[i];
        engine->b[i].im = 0.0;
    }
    fft(engine->b, n, 0);
    for (i = 0; i < n; ++i) {
        const double re = pair_freq_real[i] * engine->b[i].re + pair_freq_imag[i] * engine->b[i].im;
        const double im = pair_freq_imag[i] * engine->b[i].re - pair_freq_real[i] * engine->b[i].im;
        engine->a[i].re = re;
        engine->a[i].im = im;
    }
    fft(engine->a, n, 1);
    for (i = 0; i < n; ++i) {
        out[i] = engine->a[i].re;
    }
#endif
    return HOLO_OK;
}

int holo_weighted_sum(size_t dim,
                      const double *vectors,
                      const double *weights,
                      size_t count,
                      double *out)
{
    size_t i;
    size_t j;
    if (!out || dim == 0 || (count > 0 && !vectors)) {
        return HOLO_EINVAL;
    }
    for (j = 0; j < dim; ++j) {
        out[j] = 0.0;
    }
    for (i = 0; i < count; ++i) {
        const double w = weights ? weights[i] : 1.0;
        const double *row = vectors + i * dim;
        for (j = 0; j < dim; ++j) {
            out[j] += w * row[j];
        }
    }
    return HOLO_OK;
}

int holo_bundle(size_t dim,
                const double *vectors,
                const double *weights,
                size_t count,
                double *out)
{
    int rc;
    if (count == 0) {
        return HOLO_EINVAL;
    }
    rc = holo_weighted_sum(dim, vectors, weights, count, out);
    if (rc != HOLO_OK) {
        return rc;
    }
    return holo_normalize(dim, out);
}

int holo_permute(size_t dim, const double *in, long shift, double *out)
{
    size_t i;
    long s;
    if (!in || !out || dim == 0) {
        return HOLO_EINVAL;
    }
    s = shift % (long)dim;
    if (s < 0) {
        s += (long)dim;
    }
    if (out == in && s != 0) {
        return HOLO_EINVAL;
    }
    for (i = 0; i < dim; ++i) {
        out[(i + (size_t)s) % dim] = in[i];
    }
    return HOLO_OK;
}

int holo_cleanup_topk(size_t dim,
                      const double *query,
                      const double *matrix,
                      const uint64_t *labels,
                      size_t count,
                      size_t k,
                      holo_match *out)
{
    return holo_cleanup_topk_with_norms(dim, query, matrix, NULL, labels, count, k, out);
}

int holo_cleanup_topk_with_norms(size_t dim,
                                 const double *query,
                                 const double *matrix,
                                 const double *matrix_norms,
                                 const uint64_t *labels,
                                 size_t count,
                                 size_t k,
                                 holo_match *out)
{
    double qnorm;
    size_t i;
    size_t j;
    if (!query || !matrix || !out || dim == 0 || k == 0) {
        return HOLO_EINVAL;
    }
    if (k > count) {
        k = count;
    }
    for (j = 0; j < k; ++j) {
        out[j].index = (size_t)-1;
        out[j].label = 0;
        out[j].score = -INFINITY;
    }
    qnorm = holo_norm(dim, query);
    if (qnorm <= 0.0) {
        return HOLO_EINVAL;
    }
    for (i = 0; i < count; ++i) {
        double dot = 0.0;
        double rnorm;
        double score;
        const double *row = matrix + i * dim;
        if (matrix_norms) {
            rnorm = matrix_norms[i];
            dot = holo_dot(dim, query, row);
        } else {
            double row_norm_sq = 0.0;
            for (size_t m = 0; m < dim; ++m) {
                const double rv = row[m];
                dot += query[m] * rv;
                row_norm_sq += rv * rv;
            }
            rnorm = sqrt(row_norm_sq);
        }
        score = rnorm > 0.0 ? dot / (qnorm * rnorm) : -INFINITY;
        for (j = 0; j < k; ++j) {
            if (score > out[j].score) {
                size_t m;
                for (m = k - 1; m > j; --m) {
                    out[m] = out[m - 1];
                }
                out[j].index = i;
                out[j].label = labels ? labels[i] : (uint64_t)i;
                out[j].score = score;
                break;
            }
        }
    }
    return HOLO_OK;
}

holo_action_index *holo_action_index_create(size_t dim, size_t count)
{
    holo_action_index *index;
    if (dim == 0 || count == 0) {
        return NULL;
    }
    if (count > ((size_t)-1) / dim) {
        return NULL;
    }
    index = (holo_action_index *)calloc(1, sizeof(*index));
    if (!index) {
        return NULL;
    }
    index->vectors = (double *)alloc_zeroed(dim * count, sizeof(index->vectors[0]));
    index->norms = (double *)alloc_zeroed(count, sizeof(index->norms[0]));
    index->labels = (uint64_t *)alloc_zeroed(count, sizeof(index->labels[0]));
    if (!index->vectors || !index->norms || !index->labels) {
        holo_action_index_destroy(index);
        return NULL;
    }
    index->dim = dim;
    index->count = count;
    return index;
}

void holo_action_index_destroy(holo_action_index *index)
{
    if (!index) {
        return;
    }
    free_aligned(index->vectors);
    free_aligned(index->norms);
    free_aligned(index->labels);
    free(index);
}

size_t holo_action_index_dim(const holo_action_index *index)
{
    return index ? index->dim : 0;
}

size_t holo_action_index_count(const holo_action_index *index)
{
    return index ? index->count : 0;
}

int holo_action_index_set(holo_action_index *index,
                          const double *vectors,
                          const uint64_t *labels)
{
    size_t i;
    if (!index || !index->vectors || !index->norms || !index->labels || !vectors) {
        return HOLO_EINVAL;
    }
    memcpy(index->vectors, vectors, index->dim * index->count * sizeof(index->vectors[0]));
    for (i = 0; i < index->count; ++i) {
        index->norms[i] = holo_norm(index->dim, index->vectors + i * index->dim);
        index->labels[i] = labels ? labels[i] : (uint64_t)i;
    }
    return HOLO_OK;
}

int holo_action_index_search(const holo_action_index *index,
                             const double *query,
                             size_t k,
                             holo_match *out)
{
    if (!index || !index->vectors || !index->norms || !index->labels) {
        return HOLO_EINVAL;
    }
    return holo_cleanup_topk_with_norms(index->dim,
                                        query,
                                        index->vectors,
                                        index->norms,
                                        index->labels,
                                        index->count,
                                        k,
                                        out);
}
