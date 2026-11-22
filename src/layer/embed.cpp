// Copyright 2017 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "embed.h"

#include <limits.h>
#include <string.h>

namespace ncnn {

Embed::Embed()
{
    one_blob_only = true;
    support_inplace = false;
}

int Embed::load_param(const ParamDict& pd)
{
    num_output = pd.get(0, 0);
    input_dim = pd.get(1, 0);
    bias_term = pd.get(2, 0);
    weight_data_size = pd.get(3, 0);
    int8_scale_term = pd.get(18, 0);

    if (int8_scale_term == 900)
    {
#if NCNN_WEIGHT_QUANT
        if (num_output <= 0 || num_output % 32 != 0 || input_dim <= 0 || num_output > INT_MAX / input_dim || weight_data_size != num_output * input_dim)
        {
            NCNN_LOGE("Embed invalid Q8_0 weight shape");
            return -1;
        }
#else
        NCNN_LOGE("Embed Q8_0 requires NCNN_WEIGHT_QUANT");
        return -1;
#endif // NCNN_WEIGHT_QUANT
    }

    return 0;
}

int Embed::load_model(const ModelBin& mb)
{
    weight_data = mb.load(weight_data_size, int8_scale_term == 900 ? 3 : 0);
    if (weight_data.empty())
        return -100;

    if (bias_term)
    {
        bias_data = mb.load(num_output, 1);
        if (bias_data.empty())
            return -100;
    }

#if NCNN_WEIGHT_QUANT
    if (int8_scale_term == 900)
    {
        const int scale_size = weight_data_size / 32 * 2;
        weight_data_q8_scales = mb.load(scale_size, 3);
        if (weight_data.elemsize != 1u || weight_data.w != weight_data_size || weight_data_q8_scales.empty()
            || weight_data_q8_scales.elemsize != 1u || weight_data_q8_scales.w != scale_size)
            return -100;

        const unsigned char* scales = weight_data_q8_scales;
        for (int i = 1; i < weight_data_q8_scales.w; i += 2)
        {
            if ((scales[i] & 0x7c) == 0x7c)
            {
                NCNN_LOGE("Embed Q8_0 weight scale is not finite");
                return -1;
            }
        }

        return 0;
    }
#endif // NCNN_WEIGHT_QUANT

#if NCNN_INT8
    if (int8_scale_term)
    {
        weight_data_int8_scale = mb.load(1, 1)[0];
    }
#endif // NCNN_INT8

    return 0;
}

static void embed(const Mat& bottom_blob, const Mat& weight_data, const Mat& bias_data, Mat& top_blob, int input_dim, const Option& opt)
{
    const int num_output = top_blob.w;
    const int words = top_blob.h;

    const float* bias_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < words; q++)
    {
        float* outptr = top_blob.row(q);

        int word_index = ((const int*)bottom_blob)[q];

        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;

        const float* em = (const float*)weight_data + num_output * word_index;

        if (bias_ptr)
        {
            for (int p = 0; p < num_output; p++)
            {
                outptr[p] = em[p] + bias_ptr[p];
            }
        }
        else
        {
            memcpy(outptr, em, num_output * sizeof(float));
        }
    }
}

#if NCNN_INT8
static void embed_int8(const Mat& bottom_blob, const Mat& weight_data, float weight_data_int8_scale, const Mat& bias_data, Mat& top_blob, int input_dim, const Option& opt)
{
    const int num_output = top_blob.w;
    const int words = top_blob.h;

    const float* bias_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < words; q++)
    {
        float* outptr = top_blob.row(q);

        int word_index = ((const int*)bottom_blob)[q];

        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;

        const float descale_em = 1.f / weight_data_int8_scale;

        const signed char* em = (const signed char*)weight_data + num_output * word_index;

        if (bias_ptr)
        {
            for (int p = 0; p < num_output; p++)
            {
                outptr[p] = em[p] * descale_em + bias_ptr[p];
            }
        }
        else
        {
            for (int p = 0; p < num_output; p++)
            {
                outptr[p] = em[p] * descale_em;
            }
        }
    }
}
#endif // NCNN_INT8

#if NCNN_WEIGHT_QUANT
static void embed_q8_0(const Mat& bottom_blob, const Mat& weight_data, const Mat& scales, const Mat& bias_data, Mat& top_blob, int input_dim, const Option& opt)
{
    const int num_output = top_blob.w;
    const int words = top_blob.h;

    const float* bias_ptr = bias_data;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < words; q++)
    {
        float* outptr = top_blob.row(q);

        int word_index = ((const int*)bottom_blob)[q];

        if (word_index < 0)
            word_index = 0;
        if (word_index >= input_dim)
            word_index = input_dim - 1;

        const size_t row_offset = (size_t)word_index * num_output;
        const signed char* em = (const signed char*)weight_data + row_offset;
        const unsigned char* sptr = (const unsigned char*)scales + row_offset / 32 * 2;

        if (bias_ptr)
        {
            const float* ptr = bias_ptr;
            for (int p = 0; p < num_output; p += 32)
            {
                const float descale_em = float16_to_float32(sptr[0] | (sptr[1] << 8));
                for (int i = 0; i < 32; i++)
                {
                    outptr[i] = em[i] * descale_em + ptr[i];
                }
                em += 32;
                ptr += 32;
                sptr += 2;
                outptr += 32;
            }
        }
        else
        {
            for (int p = 0; p < num_output; p += 32)
            {
                const float descale_em = float16_to_float32(sptr[0] | (sptr[1] << 8));
                for (int i = 0; i < 32; i++)
                {
                    // As with a zero bias, normalize negative zero.
                    outptr[i] = em[i] * descale_em + 0.f;
                }
                em += 32;
                sptr += 2;
                outptr += 32;
            }
        }
    }
}
#endif // NCNN_WEIGHT_QUANT

int Embed::forward(const Mat& bottom_blob, Mat& top_blob, const Option& opt) const
{
    const int words = bottom_blob.w;

    top_blob.create(num_output, words, 4u, opt.blob_allocator);
    if (top_blob.empty())
        return -100;

#if NCNN_WEIGHT_QUANT
    if (int8_scale_term == 900)
    {
        embed_q8_0(bottom_blob, weight_data, weight_data_q8_scales, bias_data, top_blob, input_dim, opt);
        return 0;
    }
#endif // NCNN_WEIGHT_QUANT

#if NCNN_INT8
    if (int8_scale_term)
    {
        embed_int8(bottom_blob, weight_data, weight_data_int8_scale, bias_data, top_blob, input_dim, opt);
    }
    else
#endif // NCNN_INT8
    {
        embed(bottom_blob, weight_data, bias_data, top_blob, input_dim, opt);
    }

    return 0;
}

} // namespace ncnn
