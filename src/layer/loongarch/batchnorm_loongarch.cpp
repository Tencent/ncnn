// Copyright 2022 yala <zhaojunchao@loongson.cn>;<junchao82@qq.com>
// SPDX-License-Identifier: BSD-3-Clause

#include "batchnorm_loongarch.h"

#if __loongarch_sx
#include <lsxintrin.h>
#if __loongarch_asx
#include <lasxintrin.h>
#endif // __loongarch_asx
#endif // __loongarch_sx

#include "loongarch_usability.h"

namespace ncnn {

#if NCNN_BF16
static void batchnorm_bf16s_lsx(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
{
#if __loongarch_sx
    __m128 _a128 = (elempack == 4) ? (__m128)__lsx_vld(a, 0) : (__m128)__lsx_vreplfr2vr_s(a[0]);
    __m128 _b128 = (elempack == 4) ? (__m128)__lsx_vld(b, 0) : (__m128)__lsx_vreplfr2vr_s(b[0]);
#if __loongarch_asx
    __m256 _a256 = (elempack == 8) ? (__m256)__lasx_xvld(a, 0) : combine4x2_ps(_a128, _a128);
    __m256 _b256 = (elempack == 8) ? (__m256)__lasx_xvld(b, 0) : combine4x2_ps(_b128, _b128);
#endif
#endif
    float sa = a[0];
    float sb = b[0];

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_lasx((__m128i)__lsx_vld(ptr, 0));
        _p = __lasx_xvfmadd_s(_p, _b256, _a256);
        __lsx_vst(float2bfloat_lasx(_p), ptr, 0);
        ptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_lsx((__m128i)__lsx_vld(ptr, 0));
        _p = __lsx_vfmadd_s(_p, _b128, _a128);
        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr, 0, 0);
        ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(sb * bfloat16_to_float32(*ptr) + sa);
        ptr++;
    }
}

static void batchnorm_bf16s_per_element_lsx(unsigned short* ptr, const float* a, const float* b, int size, int num_threads)
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
        __m256 _a0 = (__m256)__lasx_xvld(a + i, 0);
        __m256 _b0 = (__m256)__lasx_xvld(b + i, 0);
        _p = __lasx_xvfmadd_s(_p, _b0, _a0);
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
        __m128 _a0 = (__m128)__lsx_vld(a + i, 0);
        __m128 _b0 = (__m128)__lsx_vld(b + i, 0);
        _p = __lsx_vfmadd_s(_p, _b0, _a0);
        __lsx_vstelm_d(float2bfloat_lsx(_p, _p), ptr + i, 0, 0);
    }
    remain_size_start += nn_size * 4;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(b[i] * bfloat16_to_float32(ptr[i]) + a[i]);
    }
}
#endif

BatchNorm_loongarch::BatchNorm_loongarch()
{
#if __loongarch_sx
    support_packing = true;
#if NCNN_BF16
    support_bf16_storage = true;
#endif
#endif // __loongarch_sx
}

int BatchNorm_loongarch::forward_inplace(Mat& bottom_top_blob, const Option& opt) const
{
#if NCNN_BF16
    if (opt.use_bf16_storage && bottom_top_blob.elembits() == 16)
        return forward_inplace_bf16s(bottom_top_blob, opt);
#endif

    int dims = bottom_top_blob.dims;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        int w = bottom_top_blob.w * elempack;

#if __loongarch_sx
#if __loongarch_asx
        int nn_w8 = w / 8;
        int nn_w = (w - nn_w8 * 8) / 4;
        int remain_w_start = nn_w8 * 8 + nn_w * 4;
#else
        int nn_w = w / 4;
        int remain_w_start = nn_w * 4;
#endif // __loongarch_asx
#else
        int remain_w_start = 0;
#endif // __loongarch_sx

        float* ptr = bottom_top_blob;

#if __loongarch_sx
#if __loongarch_asx
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn_w8; i++)
        {
            float* ptr0 = ptr + i * 8;

            __m256 _p = (__m256)__lasx_xvld(ptr0, 0);
            __m256 _a = (__m256)__lasx_xvld((const float*)a_data + i * 8, 0);
            __m256 _b = (__m256)__lasx_xvld((const float*)b_data + i * 8, 0);
            _p = __lasx_xvfmadd_s(_b, _p, _a);
            __lasx_xvst(_p, ptr0, 0);
        }
#endif // __loongarch_asx
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < nn_w; i++)
        {
#if __loongarch_asx
            float* ptr0 = ptr + nn_w8 * 8 + i * 4;
#else
            float* ptr0 = ptr + i * 4;
#endif // __loongarch_asx

            __m128 _p = (__m128)__lsx_vld(ptr0, 0);
#if __loongarch_asx
            __m128 _a = (__m128)__lsx_vld((const float*)a_data + (nn_w8 * 8 + i * 4), 0);
            __m128 _b = (__m128)__lsx_vld((const float*)b_data + (nn_w8 * 8 + i * 4), 0);
#else
            __m128 _a = (__m128)__lsx_vld((const float*)a_data + i * 4, 0);
            __m128 _b = (__m128)__lsx_vld((const float*)b_data + i * 4, 0);
#endif // __loongarch_asx
            _p = __lsx_vfmadd_s(_b, _p, _a);
            __lsx_vst(_p, ptr0, 0);
        }
#endif // __loongarch_sx

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = remain_w_start; i < w; i++)
        {
            ptr[i] = b_data[i] * ptr[i] + a_data[i];
        }
    }

    if (dims == 2)
    {
        int w = bottom_top_blob.w * elempack;
        int h = bottom_top_blob.h;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            float* ptr = bottom_top_blob.row(i);
            float a = a_data[i];
            float b = b_data[i];

            int j = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _a256 = elempack == 8 ? (__m256)__lasx_xvld((const float*)a_data + i * 8, 0) : elempack == 4 ? combine4x2_ps((__m128)__lsx_vld((const float*)a_data + i * 4, 0), (__m128)__lsx_vld((const float*)a_data + i * 4, 0)) : (__m256)__lasx_xvreplfr2vr_s(a);
            __m256 _b256 = elempack == 8 ? (__m256)__lasx_xvld((const float*)b_data + i * 8, 0) : elempack == 4 ? combine4x2_ps((__m128)__lsx_vld((const float*)b_data + i * 4, 0), (__m128)__lsx_vld((const float*)b_data + i * 4, 0)) : (__m256)__lasx_xvreplfr2vr_s(b);
            for (; j + 7 < w; j += 8)
            {
                __builtin_prefetch(ptr + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                _p = __lasx_xvfmadd_s(_b256, _p, _a256);
                __lasx_xvst(_p, ptr, 0);

                ptr += 8;
            }
#endif // __loongarch_asx
            __m128 _a = elempack == 4 ? (__m128)__lsx_vld((const float*)a_data + i * 4, 0) : (__m128)__lsx_vreplfr2vr_s(a);
            __m128 _b = elempack == 4 ? (__m128)__lsx_vld((const float*)b_data + i * 4, 0) : (__m128)__lsx_vreplfr2vr_s(b);
            for (; j + 3 < w; j += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                _p = __lsx_vfmadd_s(_b, _p, _a);
                __lsx_vst(_p, ptr, 0);

                ptr += 4;
            }
#endif // __loongarch_sx
            for (; j < w; j++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }

    if (dims == 3 || dims == 4)
    {
        int w = bottom_top_blob.w;
        int h = bottom_top_blob.h;
        int d = bottom_top_blob.d;
        int c = bottom_top_blob.c;
        int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            float a = a_data[q];
            float b = b_data[q];

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _a256 = elempack == 8 ? (__m256)__lasx_xvld((const float*)a_data + q * 8, 0) : elempack == 4 ? combine4x2_ps((__m128)__lsx_vld((const float*)a_data + q * 4, 0), (__m128)__lsx_vld((const float*)a_data + q * 4, 0)) : (__m256)__lasx_xvreplfr2vr_s(a);
            __m256 _b256 = elempack == 8 ? (__m256)__lasx_xvld((const float*)b_data + q * 8, 0) : elempack == 4 ? combine4x2_ps((__m128)__lsx_vld((const float*)b_data + q * 4, 0), (__m128)__lsx_vld((const float*)b_data + q * 4, 0)) : (__m256)__lasx_xvreplfr2vr_s(b);
            for (; i + 7 < size; i += 8)
            {
                __builtin_prefetch(ptr + 32);
                __m256 _p = (__m256)__lasx_xvld(ptr, 0);
                _p = __lasx_xvfmadd_s(_b256, _p, _a256);
                __lasx_xvst(_p, ptr, 0);

                ptr += 8;
            }
#endif // __loongarch_asx
            __m128 _a = elempack == 4 ? (__m128)__lsx_vld((const float*)a_data + q * 4, 0) : (__m128)__lsx_vreplfr2vr_s(a);
            __m128 _b = elempack == 4 ? (__m128)__lsx_vld((const float*)b_data + q * 4, 0) : (__m128)__lsx_vreplfr2vr_s(b);
            for (; i + 3 < size; i += 4)
            {
                __builtin_prefetch(ptr + 16);
                __m128 _p = (__m128)__lsx_vld(ptr, 0);
                _p = __lsx_vfmadd_s(_b, _p, _a);
                __lsx_vst(_p, ptr, 0);

                ptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                *ptr = b * *ptr + a;
                ptr++;
            }
        }
    }

    return 0;
}

#if NCNN_BF16
int BatchNorm_loongarch::forward_inplace_bf16s(Mat& bottom_top_blob, const Option& opt) const
{
    int dims = bottom_top_blob.dims;
    int w = bottom_top_blob.w;
    int h = bottom_top_blob.h;
    int d = bottom_top_blob.d;
    int c = bottom_top_blob.c;
    int elempack = bottom_top_blob.elempack;

    if (dims == 1)
    {
        unsigned short* ptr = bottom_top_blob;
        const float* aptr = a_data;
        const float* bptr = b_data;

        const int size = w * elempack;

        batchnorm_bf16s_per_element_lsx(ptr, aptr, bptr, size, opt.num_threads);
    }

    if (dims == 2)
    {
        const int size = w * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int i = 0; i < h; i++)
        {
            unsigned short* ptr = bottom_top_blob.row<unsigned short>(i);
            const float* aptr = (const float*)a_data + i * elempack;
            const float* bptr = (const float*)b_data + i * elempack;

            batchnorm_bf16s_lsx(ptr, aptr, bptr, size, elempack);
        }
    }

    if (dims == 3 || dims == 4)
    {
        const int size = w * h * d * elempack;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < c; q++)
        {
            unsigned short* ptr = bottom_top_blob.channel(q);
            const float* aptr = (const float*)a_data + q * elempack;
            const float* bptr = (const float*)b_data + q * elempack;

            batchnorm_bf16s_lsx(ptr, aptr, bptr, size, elempack);
        }
    }

    return 0;
}
#endif // NCNN_BF16

} // namespace ncnn
