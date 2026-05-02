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

#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blobs, opt);
#endif

#if __loongarch_sx
    const Mat& scale_blob = bottom_top_blobs[1];

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    const float* scale = scale_blob;
    const float* bias = bias_data;

    if (dims == 1)
    {
        float* ptr = bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
            int nn_size = 0;
            int remain_size_start = 0;
#if __loongarch_asx
            nn_size = (size - remain_size_start) / 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 8;
                __m256 _p = (__m256)__lasx_xvld(ptr + i, 0);
                __m256 _s = (__m256)__lasx_xvld(scale + i, 0);
                __m256 _bias = (__m256)__lasx_xvld(bias + i, 0);
                _p = __lasx_xvfmadd_s(_p, _s, _bias);
                __lasx_xvst(_p, ptr + i, 0);
            }
            remain_size_start += nn_size * 8;
#endif // __loongarch_asx
            nn_size = (size - remain_size_start) / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 4;
                __m128 _p = (__m128)__lsx_vld(ptr + i, 0);
                __m128 _s = (__m128)__lsx_vld(scale + i, 0);
                __m128 _bias = (__m128)__lsx_vld(bias + i, 0);
                _p = __lsx_vfmadd_s(_p, _s, _bias);
                __lsx_vst(_p, ptr + i, 0);
            }
            remain_size_start += nn_size * 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_size_start; i < size; i++)
            {
                ptr[i] = ptr[i] * scale[i] + bias[i];
            }
        }
        else
        {
            int nn_size = 0;
            int remain_size_start = 0;
#if __loongarch_asx
            nn_size = (size - remain_size_start) / 8;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 8;
                __m256 _p = (__m256)__lasx_xvld(ptr + i, 0);
                __m256 _s = (__m256)__lasx_xvld(scale + i, 0);
                _p = __lasx_xvfmul_s(_p, _s);
                __lasx_xvst(_p, ptr + i, 0);
            }
            remain_size_start += nn_size * 8;
#endif // __loongarch_asx
            nn_size = (size - remain_size_start) / 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int ii = 0; ii < nn_size; ii++)
            {
                int i = remain_size_start + ii * 4;
                __m128 _p = (__m128)__lsx_vld(ptr + i, 0);
                __m128 _s = (__m128)__lsx_vld(scale + i, 0);
                _p = __lsx_vfmul_s(_p, _s);
                __lsx_vst(_p, ptr + i, 0);
            }
            remain_size_start += nn_size * 4;
            #pragma omp parallel for num_threads(opt.num_threads)
            for (int i = remain_size_start; i < size; i++)
            {
                ptr[i] *= scale[i];
            }
        }

        return 0;
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);

            float s = scale[i];
            __m128 _s128 = (elempack == 4) ? (__m128)__lsx_vld(scale + i * 4, 0) : (__m128)__lsx_vreplfr2vr_s(s);
#if __loongarch_asx
            __m256 _s256 = (elempack == 8) ? (__m256)__lasx_xvld(scale + i * 8, 0) : __lasx_concat_128_s(_s128, _s128);
#endif // __loongarch_asx

            if (bias_term)
            {
                float b = bias[i];
                __m128 _b128 = (elempack == 4) ? (__m128)__lsx_vld(bias + i * 4, 0) : (__m128)__lsx_vreplfr2vr_s(b);
#if __loongarch_asx
                __m256 _b256 = (elempack == 8) ? (__m256)__lasx_xvld(bias + i * 8, 0) : __lasx_concat_128_s(_b128, _b128);
#endif // __loongarch_asx

                int j = 0;
#if __loongarch_asx
                for (; j + 7 < size; j += 8)
                {
                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    _p = __lasx_xvfmadd_s(_p, _s256, _b256);
                    __lasx_xvst(_p, ptr, 0);
                    ptr += 8;
                }
#endif // __loongarch_asx
                for (; j + 3 < size; j += 4)
                {
                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    _p = __lsx_vfmadd_s(_p, _s128, _b128);
                    __lsx_vst(_p, ptr, 0);
                    ptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr = *ptr * s + b;
                    ptr++;
                }
            }
            else
            {
                int j = 0;
#if __loongarch_asx
                for (; j + 7 < size; j += 8)
                {
                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    _p = __lasx_xvfmul_s(_p, _s256);
                    __lasx_xvst(_p, ptr, 0);
                    ptr += 8;
                }
#endif // __loongarch_asx
                for (; j + 3 < size; j += 4)
                {
                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    _p = __lsx_vfmul_s(_p, _s128);
                    __lsx_vst(_p, ptr, 0);
                    ptr += 4;
                }
                for (; j < size; j++)
                {
                    *ptr *= s;
                    ptr++;
                }
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        const int size = w * h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);

            float s = scale[q];
            __m128 _s128 = (elempack == 4) ? (__m128)__lsx_vld(scale + q * 4, 0) : (__m128)__lsx_vreplfr2vr_s(s);
#if __loongarch_asx
            __m256 _s256 = (elempack == 8) ? (__m256)__lasx_xvld(scale + q * 8, 0) : __lasx_concat_128_s(_s128, _s128);
#endif // __loongarch_asx

            if (bias_term)
            {
                float b = bias[q];
                __m128 _b128 = (elempack == 4) ? (__m128)__lsx_vld(bias + q * 4, 0) : (__m128)__lsx_vreplfr2vr_s(b);
#if __loongarch_asx
                __m256 _b256 = (elempack == 8) ? (__m256)__lasx_xvld(bias + q * 8, 0) : __lasx_concat_128_s(_b128, _b128);
#endif // __loongarch_asx

                int i = 0;
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    _p = __lasx_xvfmadd_s(_p, _s256, _b256);
                    __lasx_xvst(_p, ptr, 0);
                    ptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    _p = __lsx_vfmadd_s(_p, _s128, _b128);
                    __lsx_vst(_p, ptr, 0);
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr = *ptr * s + b;
                    ptr++;
                }
            }
            else
            {
                int i = 0;
#if __loongarch_asx
                for (; i + 7 < size; i += 8)
                {
                    __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                    _p = __lasx_xvfmul_s(_p, _s256);
                    __lasx_xvst(_p, ptr, 0);
                    ptr += 8;
                }
#endif // __loongarch_asx
                for (; i + 3 < size; i += 4)
                {
                    __m128 _p = (__m128)__lsx_vld(ptr, 0);
                    _p = __lsx_vfmul_s(_p, _s128);
                    __lsx_vst(_p, ptr, 0);
                    ptr += 4;
                }
                for (; i < size; i++)
                {
                    *ptr *= s;
                    ptr++;
                }
            }
        }

        return 0;
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
    __m256 _s256 = (elempack == 8) ? (__m256)__lasx_xvld(scale, 0) : __lasx_concat_128_s(_s128, _s128);
    __m256 _b256 = (elempack == 8) ? (__m256)__lasx_xvld(bias, 0) : __lasx_concat_128_s(_b128, _b128);
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
    __m256 _s256 = (elempack == 8) ? (__m256)__lasx_xvld(scale, 0) : __lasx_concat_128_s(_s128, _s128);
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

    const int w = bottom_top_blob.w;
    const int h = bottom_top_blob.h;
    const int channels = bottom_top_blob.c;
    const int dims = bottom_top_blob.dims;
    const int elempack = bottom_top_blob.elempack;

    const float* scale = 0;
    Mat scale_fp32;
    if (scale_blob.elembits() == 16)
    {
        const int scale_data_size = scale_blob.w * scale_blob.elempack;
        scale_fp32.create(scale_data_size, 4u, 1, opt.workspace_allocator);
        if (scale_fp32.empty())
            return -100;

        const unsigned short* src = scale_blob;
        float* dst = scale_fp32;
        for (int i = 0; i < scale_data_size; i++)
        {
            dst[i] = bfloat16_to_float32(src[i]);
        }
        scale = scale_fp32;
    }
    else
    {
        scale = scale_blob;
    }
    const float* bias = bias_data;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const int size = w * elempack;

        if (bias_term)
        {
            scale_bf16s_per_element_lsx(ptr, scale, bias, size, opt.num_threads);
        }
        else
        {
            scale_bf16s_no_bias_per_element_lsx(ptr, scale, size, opt.num_threads);
        }

        return 0;
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* sptr = scale + i * elempack;

            if (bias_term)
            {
                const float* bptr = bias + i * elempack;
                scale_bf16s_lsx(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias_lsx(ptr, sptr, size, elempack);
            }
        }

        return 0;
    }

    if (dims == 3)
    {
        const int size = w * h * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* sptr = scale + q * elempack;

            if (bias_term)
            {
                const float* bptr = bias + q * elempack;
                scale_bf16s_lsx(ptr, sptr, bptr, size, elempack);
            }
            else
            {
                scale_bf16s_no_bias_lsx(ptr, sptr, size, elempack);
            }
        }

        return 0;
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
