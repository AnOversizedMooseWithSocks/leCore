#include "internal/lecore_internal.h"

#if LECORE_ENABLE_RADIX2

static void lecore_radix2_fft_f32(
    const lecore_context *context,
    float *values,
    int inverse)
{
    const uint32_t dimension = context->dimension;
    const float *twiddles = (const float *)context->radix2_twiddles;
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        uint32_t reversed = context->radix2_bit_reversal[index];
        if (reversed > index) {
            float real = values[(size_t)index * 2];
            float imaginary = values[(size_t)index * 2 + 1];
            values[(size_t)index * 2] = values[(size_t)reversed * 2];
            values[(size_t)index * 2 + 1] =
                values[(size_t)reversed * 2 + 1];
            values[(size_t)reversed * 2] = real;
            values[(size_t)reversed * 2 + 1] = imaginary;
        }
    }

    if (dimension > 1) {
        uint32_t length = 2;

        for (;;) {
            const uint32_t half = length / 2;
            const uint32_t twiddle_step = dimension / length;
            uint32_t block;

            for (block = 0; block < dimension; block += length) {
                uint32_t offset;

                for (offset = 0; offset < half; ++offset) {
                    const uint32_t twiddle_index = offset * twiddle_step;
                    const uint32_t even_index = block + offset;
                    const uint32_t odd_index = even_index + half;
                    const float wr =
                        twiddles[(size_t)twiddle_index * 2];
                    const float wi = inverse
                        ? -twiddles[(size_t)twiddle_index * 2 + 1]
                        : twiddles[(size_t)twiddle_index * 2 + 1];
                    const float odd_real =
                        values[(size_t)odd_index * 2];
                    const float odd_imaginary =
                        values[(size_t)odd_index * 2 + 1];
                    const float transformed_real =
                        odd_real * wr - odd_imaginary * wi;
                    const float transformed_imaginary =
                        odd_real * wi + odd_imaginary * wr;
                    const float even_real =
                        values[(size_t)even_index * 2];
                    const float even_imaginary =
                        values[(size_t)even_index * 2 + 1];

                    values[(size_t)even_index * 2] =
                        even_real + transformed_real;
                    values[(size_t)even_index * 2 + 1] =
                        even_imaginary + transformed_imaginary;
                    values[(size_t)odd_index * 2] =
                        even_real - transformed_real;
                    values[(size_t)odd_index * 2 + 1] =
                        even_imaginary - transformed_imaginary;
                }
            }
            if (length == dimension) {
                break;
            }
            length <<= 1;
        }
    }

    if (inverse) {
        const float scale = (float)dimension;

        for (index = 0; index < dimension; ++index) {
            values[(size_t)index * 2] /= scale;
            values[(size_t)index * 2 + 1] /= scale;
        }
    }
}

static void lecore_radix2_load_f32(
    float *destination,
    const float *source,
    uint32_t dimension,
    int involute)
{
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        uint32_t source_index = involute && index != 0
            ? dimension - index
            : index;
        destination[(size_t)index * 2] = source[source_index];
        destination[(size_t)index * 2 + 1] = 0.0f;
    }
}

static void lecore_radix2_convolve_f32(
    lecore_context *context,
    const float *left_input,
    const float *right_input,
    int involute_right,
    float *output)
{
    float *left = (float *)context->scratch;
    float *right = left + (size_t)context->dimension * 2;
    uint32_t index;

    lecore_radix2_load_f32(
        left, left_input, context->dimension, 0);
    lecore_radix2_load_f32(
        right, right_input, context->dimension, involute_right);
    lecore_radix2_fft_f32(context, left, 0);
    lecore_radix2_fft_f32(context, right, 0);

    for (index = 0; index < context->dimension; ++index) {
        const float left_real = left[(size_t)index * 2];
        const float left_imaginary = left[(size_t)index * 2 + 1];
        const float right_real = right[(size_t)index * 2];
        const float right_imaginary = right[(size_t)index * 2 + 1];

        left[(size_t)index * 2] =
            left_real * right_real - left_imaginary * right_imaginary;
        left[(size_t)index * 2 + 1] =
            left_real * right_imaginary + left_imaginary * right_real;
    }

    lecore_radix2_fft_f32(context, left, 1);
    for (index = 0; index < context->dimension; ++index) {
        output[index] = left[(size_t)index * 2];
    }
}

LECORE_INTERNAL_API void lecore_internal_hrr_radix2_bind_f32(
    lecore_context *context,
    const float *a,
    const float *b,
    float *output)
{
    lecore_radix2_convolve_f32(context, a, b, 0, output);
}

LECORE_INTERNAL_API void lecore_internal_hrr_radix2_unbind_f32(
    lecore_context *context,
    const float *composite,
    const float *key,
    float *output)
{
    lecore_radix2_convolve_f32(context, composite, key, 1, output);
}

#endif /* LECORE_ENABLE_RADIX2 */
