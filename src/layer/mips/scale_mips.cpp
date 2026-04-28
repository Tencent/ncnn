// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_mips.h"

#if __mips_msa
#include <msa.h>
#include "mips_usability.h"
#endif // __mips_msa

namespace ncnn {

Scale_mips::Scale_mips()
{
#if __mips_msa
    support_packing = true;
#endif // __mips_msa
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Scale_mips::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

#if __mips_msa
    if (elempack == 4)
    {
        if (dims == 1)
        {
            const int w = bottom_top_blob.w;

            const float* scale = scale_blob;
            if (bias_term)
            {
                const float* bias = bias_data;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _s = (v4f32)__msa_ld_w(scale + i * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w(bias + i * 4, 0);
                    _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    __builtin_prefetch(ptr + 16);
                    v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                    v4f32 _s = (v4f32)__msa_ld_w(scale + i * 4, 0);
                    _p = __msa_fmul_w(_p, _s);
                    __msa_st_w((v4i32)_p, ptr, 0);
                }
            }

            return 0;
        }

        if (dims == 2)
        {
            const int w = bottom_top_blob.w;
            const int h = bottom_top_blob.h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __builtin_prefetch(ptr + 16);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __builtin_prefetch(ptr + 16);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }

            return 0;
        }

        if (dims == 3)
        {
            const int size = bottom_top_blob.w * bottom_top_blob.h;
            const int channels = bottom_top_blob.c;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w((const float*)bias_data + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __builtin_prefetch(ptr + 16);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    v4f32 _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __builtin_prefetch(ptr + 16);
                        v4f32 _p = (v4f32)__msa_ld_w(ptr, 0);
                        _p = __msa_fmul_w(_p, _s);
                        __msa_st_w((v4i32)_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }

            return 0;
        }
    }
#endif // __mips_msa

    return Scale::forward_inplace(bottom_top_blobs, opt);
}

#if NCNN_BF16
static void scale_bf16s_msa(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
#if __mips_msa
    v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale, 0) : (v4f32)__msa_fill_w_f32(scale[0]);
    v4f32 _b = (elempack == 4) ? (v4f32)__msa_ld_w(bias, 0) : (v4f32)__msa_fill_w_f32(bias[0]);
#endif
    float s = scale[0];
    float b = bias[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __ncnn_msa_fmadd_w(_b, _p, _s);
        float2bfloat_msa_store(_p, ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s + b);
        ptr++;
    }
}

static void scale_bf16s_no_bias_msa(unsigned short* ptr, const float* scale, int size, int elempack)
{
#if __mips_msa
    v4f32 _s = (elempack == 4) ? (v4f32)__msa_ld_w(scale, 0) : (v4f32)__msa_fill_w_f32(scale[0]);
#endif
    float s = scale[0];

    int i = 0;
#if __mips_msa
    for (; i + 3 < size; i += 4)
    {
        v4f32 _p = bfloat2float_msa(ptr);
        _p = __msa_fmul_w(_p, _s);
        float2bfloat_msa_store(_p, ptr);
        ptr += 4;
    }
#endif // __mips_msa
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s);
        ptr++;
    }
}

static void scale_bf16s_per_element_msa(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
        v4f32 _b = (v4f32)__msa_ld_w(bias + i, 0);
        _p = __ncnn_msa_fmadd_w(_b, _p, _s);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i] + bias[i]);
    }
}

static void scale_bf16s_no_bias_per_element_msa(unsigned short* ptr, const float* scale, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __mips_msa
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        v4f32 _p = bfloat2float_msa(ptr + i);
        v4f32 _s = (v4f32)__msa_ld_w(scale + i, 0);
        _p = __msa_fmul_w(_p, _s);
        float2bfloat_msa_store(_p, ptr + i);
    }
    remain_size_start += nn_size * 4;
#endif // __mips_msa
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i]);
    }
}

int Scale_mips::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    int scale_elembits = scale_blob.elembits();

    int needs_cast_scale = (scale_elembits == 16);

#if __mips_msa
    if (elempack == 4)
    {
        if (dims == 1)
        {
            const int w = bottom_top_blob.w;
            const unsigned short* scale_bf16 = (const unsigned short*)scale_blob;
            const float* scale_fp32 = (const float*)scale_blob;

            unsigned short* ptr = (unsigned short*)bottom_top_blob;
            if (bias_term)
            {
                const float* bias = bias_data;
                for (int i = 0; i < w; i++)
                {
                    v4f32 _p = bfloat2float_msa(ptr);
                    v4f32 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_msa(scale_bf16 + i * 4);
                    else
                        _s = (v4f32)__msa_ld_w(scale_fp32 + i * 4, 0);
                    v4f32 _bias = (v4f32)__msa_ld_w(bias + i * 4, 0);
                    _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                    float2bfloat_msa_store(_p, ptr);
                    ptr += 4;
                }
            }
            else
            {
                for (int i = 0; i < w; i++)
                {
                    v4f32 _p = bfloat2float_msa(ptr);
                    v4f32 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_msa(scale_bf16 + i * 4);
                    else
                        _s = (v4f32)__msa_ld_w(scale_fp32 + i * 4, 0);
                    _p = __msa_fmul_w(_p, _s);
                    float2bfloat_msa_store(_p, ptr);
                    ptr += 4;
                }
            }
            return 0;
        }

        if (dims == 2)
        {
            const int w = bottom_top_blob.w;
            const int h = bottom_top_blob.h;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
                    v4f32 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_msa((const unsigned short*)scale_blob + i * 4);
                    else
                        _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);
                    v4f32 _bias;
                    if (needs_cast_scale)
                        _bias = bfloat2float_msa((const unsigned short*)bias_data + i * 4);
                    else
                        _bias = (v4f32)__msa_ld_w((const float*)bias_data + i * 4, 0);
                    for (int j = 0; j < w; j++)
                    {
                        v4f32 _p = bfloat2float_msa(ptr);
                        _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                        float2bfloat_msa_store(_p, ptr);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
                    v4f32 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_msa((const unsigned short*)scale_blob + i * 4);
                    else
                        _s = (v4f32)__msa_ld_w((const float*)scale_blob + i * 4, 0);
                    for (int j = 0; j < w; j++)
                    {
                        v4f32 _p = bfloat2float_msa(ptr);
                        _p = __msa_fmul_w(_p, _s);
                        float2bfloat_msa_store(_p, ptr);
                        ptr += 4;
                    }
                }
            }
            return 0;
        }

        if (dims == 3)
        {
            const int size = bottom_top_blob.w * bottom_top_blob.h;
            const int channels = bottom_top_blob.c;

            if (bias_term)
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    unsigned short* ptr = bottom_top_blob.channel(q);
                    v4f32 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_msa((const unsigned short*)scale_blob + q * 4);
                    else
                        _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);
                    v4f32 _bias;
                    if (needs_cast_scale)
                        _bias = bfloat2float_msa((const unsigned short*)bias_data + q * 4);
                    else
                        _bias = (v4f32)__msa_ld_w((const float*)bias_data + q * 4, 0);
                    for (int i = 0; i < size; i++)
                    {
                        v4f32 _p = bfloat2float_msa(ptr);
                        _p = __ncnn_msa_fmadd_w(_bias, _p, _s);
                        float2bfloat_msa_store(_p, ptr);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    unsigned short* ptr = bottom_top_blob.channel(q);
                    v4f32 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_msa((const unsigned short*)scale_blob + q * 4);
                    else
                        _s = (v4f32)__msa_ld_w((const float*)scale_blob + q * 4, 0);
                    for (int i = 0; i < size; i++)
                    {
                        v4f32 _p = bfloat2float_msa(ptr);
                        _p = __msa_fmul_w(_p, _s);
                        float2bfloat_msa_store(_p, ptr);
                        ptr += 4;
                    }
                }
            }

            return 0;
        }
    }
#endif // __mips_msa

    // scalar fallback, convert to fp32 and delegate
    Option opt_cast = opt;
    opt_cast.blob_allocator = opt.workspace_allocator;

    Mat bottom_top_blob_fp32;
    cast_bfloat16_to_float32(bottom_top_blob, bottom_top_blob_fp32, opt_cast);
    if (bottom_top_blob_fp32.empty())
        return -100;

    Mat scale_blob_fp32 = scale_blob;
    if (scale_blob.elembits() == 16)
    {
        cast_bfloat16_to_float32(scale_blob, scale_blob_fp32, opt_cast);
        if (scale_blob_fp32.empty())
            return -100;
    }

    std::vector<Mat> bottom_top_blobs_fp32(2);
    bottom_top_blobs_fp32[0] = bottom_top_blob_fp32;
    bottom_top_blobs_fp32[1] = scale_blob_fp32;

    int ret = forward_inplace(bottom_top_blobs_fp32, opt);
    if (ret != 0)
        return ret;

    cast_float32_to_bfloat16(bottom_top_blob_fp32, bottom_top_blob, opt);
    if (bottom_top_blob.empty())
        return -100;

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
