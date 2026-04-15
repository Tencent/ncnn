// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef BATCHNORM_LOONGARCH_BF16S_H
#define BATCHNORM_LOONGARCH_BF16S_H

static void batchnorm_bf16s_sse(unsigned short* ptr, const float* a, const float* b, int size, int elempack)
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

static void batchnorm_bf16s_per_element_sse(unsigned short* ptr, const float* a, const float* b, int size, int num_threads)
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

#endif // BATCHNORM_LOONGARCH_BF16S_H
