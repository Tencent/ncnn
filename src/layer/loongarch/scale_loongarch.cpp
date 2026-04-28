// Copyright 2026 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#include "scale_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#include "loongarch_usability.h"
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

namespace ncnn {

Scale_loongarch::Scale_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#endif // __loongarch_sx
#if NCNN_BF16
    support_bf16_storage = true;
#endif
}

int Scale_loongarch::forward_inplace(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
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
                    float* ptr = (float*)bottom_top_blob + i * 8;

                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    __m256 _s = (__m256)__lasx_xvld(scale + i * 8, 0);
                    __m256 _bias = (__m256)__lasx_xvld(bias + i * 8, 0);
                    _p = __lasx_xvfmadd_s(_p, _s, _bias);
                    __lasx_xvst(_p, ptr, 0);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 8;

                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    __m256 _s = (__m256)__lasx_xvld(scale + i * 8, 0);
                    _p = __lasx_xvfmul_s(_p, _s);
                    __lasx_xvst(_p, ptr, 0);
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
                    __m256 _s = (__m256)__lasx_xvld((const float*)scale_blob + i * 8, 0);
                    __m256 _bias = (__m256)__lasx_xvld((const float*)bias_data + i * 8, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                        _p = __lasx_xvfmadd_s(_p, _s, _bias);
                        __lasx_xvst(_p, ptr, 0);
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    float* ptr = bottom_top_blob.row(i);
                    __m256 _s = (__m256)__lasx_xvld((const float*)scale_blob + i * 8, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                        _p = __lasx_xvfmul_s(_p, _s);
                        __lasx_xvst(_p, ptr, 0);
                        ptr += 8;
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
                    __m256 _s = (__m256)__lasx_xvld((const float*)scale_blob + q * 8, 0);
                    __m256 _bias = (__m256)__lasx_xvld((const float*)bias_data + q * 8, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                        _p = __lasx_xvfmadd_s(_p, _s, _bias);
                        __lasx_xvst(_p, ptr, 0);
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    float* ptr = bottom_top_blob.channel(q);
                    __m256 _s = (__m256)__lasx_xvld((const float*)scale_blob + q * 8, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                        _p = __lasx_xvfmul_s(_p, _s);
                        __lasx_xvst(_p, ptr, 0);
                        ptr += 8;
                    }
                }
            }

            return 0;
        }
    }
#endif // __loongarch_asx
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

                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    __m128 _s = (__m128)__lsx_vld(scale + i * 4, 0);
                    __m128 _bias = (__m128)__lsx_vld(bias + i * 4, 0);
                    _p = __lsx_vfmadd_s(_p, _s, _bias);
                    __lsx_vst(_p, ptr, 0);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    float* ptr = (float*)bottom_top_blob + i * 4;

                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    __m128 _s = (__m128)__lsx_vld(scale + i * 4, 0);
                    _p = __lsx_vfmul_s(_p, _s);
                    __lsx_vst(_p, ptr, 0);
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
                    __m128 _s = (__m128)__lsx_vld((const float*)scale_blob + i * 4, 0);
                    __m128 _bias = (__m128)__lsx_vld((const float*)bias_data + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = (__m128)__lsx_vld(ptr, 0);
                        _p = __lsx_vfmadd_s(_p, _s, _bias);
                        __lsx_vst(_p, ptr, 0);
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
                    __m128 _s = (__m128)__lsx_vld((const float*)scale_blob + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = (__m128)__lsx_vld(ptr, 0);
                        _p = __lsx_vfmul_s(_p, _s);
                        __lsx_vst(_p, ptr, 0);
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
                    __m128 _s = (__m128)__lsx_vld((const float*)scale_blob + q * 4, 0);
                    __m128 _bias = (__m128)__lsx_vld((const float*)bias_data + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = (__m128)__lsx_vld(ptr, 0);
                        _p = __lsx_vfmadd_s(_p, _s, _bias);
                        __lsx_vst(_p, ptr, 0);
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
                    __m128 _s = (__m128)__lsx_vld((const float*)scale_blob + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = (__m128)__lsx_vld(ptr, 0);
                        _p = __lsx_vfmul_s(_p, _s);
                        __lsx_vst(_p, ptr, 0);
                        ptr += 4;
                    }
                }
            }

            return 0;
        }
    }
#endif // __loongarch_sx

    return Scale::forward_inplace(bottom_top_blobs, opt);
}

#if NCNN_BF16
static void scale_bf16s_lsx(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
{
#if __loongarch_sx
    __m128 _s128 = (elempack == 4) ? (__m128)__lsx_vld(scale, 0) : (__m128)__lsx_vreplfr2vr_s(scale[0]);
    __m128 _b128 = (elempack == 4) ? (__m128)__lsx_vld(bias, 0) : (__m128)__lsx_vreplfr2vr_s(bias[0]);
#if __loongarch_asx
    __m256 _s256 = (elempack == 8) ? (__m256)__lasx_xvld(scale, 0) : combine4x2_ps(_s128, _s128);
    __m256 _b256 = (elempack == 8) ? (__m256)__lasx_xvld(bias, 0) : combine4x2_ps(_b128, _b128);
#endif
#endif
    float s = scale[0];
    float b = bias[0];

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
        _p = __lasx_xvfmadd_s(_p, _s256, _b256);
        __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
        ptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
        _p = __lsx_vfmadd_s(_p, _s128, _b128);
        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr, 0, 0);
        ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s + b);
        ptr++;
    }
}

static void scale_bf16s_no_bias_lsx(unsigned short* ptr, const float* scale, int size, int elempack)
{
#if __loongarch_sx
    __m128 _s128 = (elempack == 4) ? (__m128)__lsx_vld(scale, 0) : (__m128)__lsx_vreplfr2vr_s(scale[0]);
#if __loongarch_asx
    __m256 _s256 = (elempack == 8) ? (__m256)__lasx_xvld(scale, 0) : combine4x2_ps(_s128, _s128);
#endif
#endif
    float s = scale[0];

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
        _p = __lasx_xvfmul_s(_p, _s256);
        __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
        ptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
        _p = __lsx_vfmul_s(_p, _s128);
        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr, 0, 0);
        ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s);
        ptr++;
    }
}

static void scale_bf16s_per_element_lsx(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __loongarch_sx
#if __loongarch_asx
    nn_size = (size - remain_size_start) / 8;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 8;
        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr + i, 0));
        __m256 _s = (__m256)__lasx_xvld(scale + i, 0);
        __m256 _bias = (__m256)__lasx_xvld(bias + i, 0);
        _p = __lasx_xvfmadd_s(_p, _s, _bias);
        __lsx_vst(float2bfloat_lasx(_p), ptr + i, 0);
    }
    remain_size_start += nn_size * 8;
#endif // __loongarch_asx
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr + i, 0));
        __m128 _s = (__m128)__lsx_vld(scale + i, 0);
        __m128 _bias = (__m128)__lsx_vld(bias + i, 0);
        _p = __lsx_vfmadd_s(_p, _s, _bias);
        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr + i, 0, 0);
    }
    remain_size_start += nn_size * 4;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i] + bias[i]);
    }
}

static void scale_bf16s_no_bias_per_element_lsx(unsigned short* ptr, const float* scale, int size, int num_threads)
{
    int nn_size = 0;
    int remain_size_start = 0;
#if __loongarch_sx
#if __loongarch_asx
    nn_size = (size - remain_size_start) / 8;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 8;
        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr + i, 0));
        __m256 _s = (__m256)__lasx_xvld(scale + i, 0);
        _p = __lasx_xvfmul_s(_p, _s);
        __lsx_vst(float2bfloat_lasx(_p), ptr + i, 0);
    }
    remain_size_start += nn_size * 8;
#endif // __loongarch_asx
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr + i, 0));
        __m128 _s = (__m128)__lsx_vld(scale + i, 0);
        _p = __lsx_vfmul_s(_p, _s);
        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr + i, 0, 0);
    }
    remain_size_start += nn_size * 4;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i]);
    }
}

int Scale_loongarch::forward_inplace_bf16s(std::vector<Mat>& bottom_top_blobs, const Option& opt) const
{
    Mat& bottom_top_blob = bottom_top_blobs[0];
    const Mat& scale_blob = bottom_top_blobs[1];

    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;
    int scale_elembits = scale_blob.elembits();

    int needs_cast_scale = (scale_elembits == 16);

#if __loongarch_sx
#if __loongarch_asx
    if (elempack == 8)
    {
        if (dims == 1)
        {
            const int w = bottom_top_blob.w;
            const unsigned short* scale_bf16 = (const unsigned short*)scale_blob;
            const float* scale_fp32 = (const float*)scale_blob;

            if (bias_term)
            {
                const float* bias = bias_data;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 8;

                    __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                    __m256 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lasx((__m128i*)(scale_bf16 + i * 8));
                    else
                        _s = (__m256)__lasx_xvld(scale_fp32 + i * 8, 0);
                    __m256 _bias = (__m256)__lasx_xvld(bias + i * 8, 0);
                    _p = __lasx_xvfmadd_s(_p, _s, _bias);
                    __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 8;

                    __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                    __m256 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lasx((__m128i*)(scale_bf16 + i * 8));
                    else
                        _s = (__m256)__lasx_xvld(scale_fp32 + i * 8, 0);
                    _p = __lasx_xvfmul_s(_p, _s);
                    __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
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
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.row(i);
                    __m256 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lasx((__m128i*)((const unsigned short*)scale_blob + i * 8));
                    else
                        _s = (__m256)__lasx_xvld((const float*)scale_blob + i * 8, 0);
                    __m256 _bias;
                    if (needs_cast_scale)
                        _bias = bfloat2float_lasx((__m128i*)((const unsigned short*)bias_data + i * 8));
                    else
                        _bias = (__m256)__lasx_xvld((const float*)bias_data + i * 8, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                        _p = __lasx_xvfmadd_s(_p, _s, _bias);
                        __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.row(i);
                    __m256 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lasx((__m128i*)((const unsigned short*)scale_blob + i * 8));
                    else
                        _s = (__m256)__lasx_xvld((const float*)scale_blob + i * 8, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                        _p = __lasx_xvfmul_s(_p, _s);
                        __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                        ptr += 8;
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
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);
                    __m256 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lasx((__m128i*)((const unsigned short*)scale_blob + q * 8));
                    else
                        _s = (__m256)__lasx_xvld((const float*)scale_blob + q * 8, 0);
                    __m256 _bias;
                    if (needs_cast_scale)
                        _bias = bfloat2float_lasx((__m128i*)((const unsigned short*)bias_data + q * 8));
                    else
                        _bias = (__m256)__lasx_xvld((const float*)bias_data + q * 8, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                        _p = __lasx_xvfmadd_s(_p, _s, _bias);
                        __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                        ptr += 8;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);
                    __m256 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lasx((__m128i*)((const unsigned short*)scale_blob + q * 8));
                    else
                        _s = (__m256)__lasx_xvld((const float*)scale_blob + q * 8, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m256 _p = bfloat2float_lasx((__m128i*)ptr);
                        _p = __lasx_xvfmul_s(_p, _s);
                        __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
                        ptr += 8;
                    }
                }
            }

            return 0;
        }
    }
#endif // __loongarch_asx
    if (elempack == 4)
    {
        if (dims == 1)
        {
            const int w = bottom_top_blob.w;
            const unsigned short* scale_bf16 = (const unsigned short*)scale_blob;
            const float* scale_fp32 = (const float*)scale_blob;

            if (bias_term)
            {
                const float* bias = bias_data;
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;

                    __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                    __m128 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lsx((__m128i*)(scale_bf16 + i * 4));
                    else
                        _s = (__m128)__lsx_vld(scale_fp32 + i * 4, 0);
                    __m128 _bias = (__m128)__lsx_vld(bias + i * 4, 0);
                    _p = __lsx_vfmadd_s(_p, _s, _bias);
                    __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < w; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob + i * 4;

                    __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                    __m128 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lsx((__m128i*)(scale_bf16 + i * 4));
                    else
                        _s = (__m128)__lsx_vld(scale_fp32 + i * 4, 0);
                    _p = __lsx_vfmul_s(_p, _s);
                    __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
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
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.row(i);
                    __m128 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lsx((__m128i*)((const unsigned short*)scale_blob + i * 4));
                    else
                        _s = (__m128)__lsx_vld((const float*)scale_blob + i * 4, 0);
                    __m128 _bias;
                    if (needs_cast_scale)
                        _bias = bfloat2float_lsx((__m128i*)((const unsigned short*)bias_data + i * 4));
                    else
                        _bias = (__m128)__lsx_vld((const float*)bias_data + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                        _p = __lsx_vfmadd_s(_p, _s, _bias);
                        __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int i = 0; i < h; i++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.row(i);
                    __m128 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lsx((__m128i*)((const unsigned short*)scale_blob + i * 4));
                    else
                        _s = (__m128)__lsx_vld((const float*)scale_blob + i * 4, 0);

                    for (int j = 0; j < w; j++)
                    {
                        __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                        _p = __lsx_vfmul_s(_p, _s);
                        __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
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
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);
                    __m128 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lsx((__m128i*)((const unsigned short*)scale_blob + q * 4));
                    else
                        _s = (__m128)__lsx_vld((const float*)scale_blob + q * 4, 0);
                    __m128 _bias;
                    if (needs_cast_scale)
                        _bias = bfloat2float_lsx((__m128i*)((const unsigned short*)bias_data + q * 4));
                    else
                        _bias = (__m128)__lsx_vld((const float*)bias_data + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                        _p = __lsx_vfmadd_s(_p, _s, _bias);
                        __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                        ptr += 4;
                    }
                }
            }
            else
            {
                #pragma omp parallel for num_threads(opt.num_threads)
                for (int q = 0; q < channels; q++)
                {
                    unsigned short* ptr = (unsigned short*)bottom_top_blob.channel(q);
                    __m128 _s;
                    if (needs_cast_scale)
                        _s = bfloat2float_lsx((__m128i*)((const unsigned short*)scale_blob + q * 4));
                    else
                        _s = (__m128)__lsx_vld((const float*)scale_blob + q * 4, 0);

                    for (int i = 0; i < size; i++)
                    {
                        __m128 _p = bfloat2float_lsx((__m128i*)ptr);
                        _p = __lsx_vfmul_s(_p, _s);
                        __lsx_vstelm_d(float2bfloat_lsx(_p), ptr, 0, 0);
                        ptr += 4;
                    }
                }
            }

            return 0;
        }
    }
#endif // __loongarch_sx

    // scalar fallback
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
