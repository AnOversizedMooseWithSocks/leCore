#include "internal/lecore_internal.h"

#if LECORE_ENABLE_RADIX2

static void lecore_radix2_fft_f32(
    const lecore_context *context,
    float *values,
    int inverse,
    int input_is_bit_reversed)
{
    const uint32_t dimension = context->dimension;
    const float *twiddles = (const float *)context->radix2_twiddles;
    uint32_t index;

    if (!input_is_bit_reversed) {
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
    const lecore_context *context,
    float *destination,
    const float *source,
    int involute)
{
    uint32_t index;

    for (index = 0; index < context->dimension; ++index) {
        const uint32_t reversed = context->radix2_bit_reversal[index];
        uint32_t source_index = involute && index != 0
            ? context->dimension - index
            : index;
        destination[(size_t)reversed * 2] = source[source_index];
        destination[(size_t)reversed * 2 + 1] = 0.0f;
    }
}

static void lecore_radix2_multiply_f32(
    float *destination,
    const float *left,
    const float *right,
    uint32_t dimension)
{
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        const float left_real = left[(size_t)index * 2];
        const float left_imaginary = left[(size_t)index * 2 + 1];
        const float right_real = right[(size_t)index * 2];
        const float right_imaginary = right[(size_t)index * 2 + 1];

        destination[(size_t)index * 2] =
            left_real * right_real - left_imaginary * right_imaginary;
        destination[(size_t)index * 2 + 1] =
            left_real * right_imaginary + left_imaginary * right_real;
    }
}

static void lecore_radix2_store_real_f32(
    float *output,
    const float *values,
    uint32_t dimension)
{
    uint32_t index;

    for (index = 0; index < dimension; ++index) {
        output[index] = values[(size_t)index * 2];
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
    lecore_radix2_load_f32(
        context, left, left_input, 0);
    lecore_radix2_load_f32(
        context, right, right_input, involute_right);
    lecore_radix2_fft_f32(context, left, 0, 1);
    lecore_radix2_fft_f32(context, right, 0, 1);
    lecore_radix2_multiply_f32(
        left, left, right, context->dimension);
    lecore_radix2_fft_f32(context, left, 1, 0);
    lecore_radix2_store_real_f32(output, left, context->dimension);
}

static void lecore_radix2_convolve_fixed_left_f32(
    lecore_context *context,
    const float *fixed_left,
    const float *right_rows,
    size_t row_count,
    size_t right_stride,
    int involute_right,
    float *out_rows,
    size_t out_stride)
{
    float *left = (float *)context->scratch;
    float *right = left + (size_t)context->dimension * 2;
    size_t row;

    lecore_radix2_load_f32(context, left, fixed_left, 0);
    lecore_radix2_fft_f32(context, left, 0, 1);
    for (row = 0; row < row_count; ++row) {
        lecore_radix2_load_f32(
            context,
            right,
            right_rows + row * right_stride,
            involute_right);
        lecore_radix2_fft_f32(context, right, 0, 1);
        lecore_radix2_multiply_f32(
            right, left, right, context->dimension);
        lecore_radix2_fft_f32(context, right, 1, 0);
        lecore_radix2_store_real_f32(
            out_rows + row * out_stride, right, context->dimension);
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

LECORE_INTERNAL_API void lecore_internal_hrr_radix2_bind_fixed_f32(
    lecore_context *context,
    const float *role,
    const float *rows,
    size_t row_count,
    size_t row_stride,
    float *out_rows,
    size_t out_stride)
{
    lecore_radix2_convolve_fixed_left_f32(
        context, role, rows, row_count, row_stride, 0, out_rows, out_stride);
}

LECORE_INTERNAL_API void lecore_internal_hrr_radix2_unbind_all_f32(
    lecore_context *context,
    const float *trace,
    const float *keys,
    size_t key_count,
    size_t key_stride,
    float *out_rows,
    size_t out_stride)
{
    lecore_radix2_convolve_fixed_left_f32(
        context,
        trace,
        keys,
        key_count,
        key_stride,
        1,
        out_rows,
        out_stride);
}

#endif /* LECORE_ENABLE_RADIX2 */
