// Copyright 2024 Tencent
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SCALE_LOONGARCH_BF16S_H
#define SCALE_LOONGARCH_BF16S_H

static void scale_bf16s_sse(unsigned short* ptr, const float* scale, const float* bias, int size, int elempack)
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
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
        _p = __lasx_xvfmadd_s(_p, _s256, _b256);
        __lsx_vst(float2bfloat_avx(_p), ptr, 0);
        ptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
        _p = __lsx_vfmadd_s(_p, _s128, _b128);
        __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
        ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s + b);
        ptr++;
    }
}

static void scale_bf16s_no_bias_sse(unsigned short* ptr, const float* scale, int size, int elempack)
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
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
        _p = __lasx_xvfmul_s(_p, _s256);
        __lsx_vst(float2bfloat_avx(_p), ptr, 0);
        ptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
        _p = __lsx_vfmul_s(_p, _s128);
        __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
        ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        *ptr = float32_to_bfloat16(bfloat16_to_float32(*ptr) * s);
        ptr++;
    }
}

static void scale_bf16s_per_element_sse(unsigned short* ptr, const float* scale, const float* bias, int size, int num_threads)
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
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr + i, 0));
        __m256 _s = (__m256)__lasx_xvld(scale + i, 0);
        __m256 _bias = (__m256)__lasx_xvld(bias + i, 0);
        _p = __lasx_xvfmadd_s(_p, _s, _bias);
        __lsx_vst(float2bfloat_avx(_p), ptr + i, 0);
    }
    remain_size_start += nn_size * 8;
#endif // __loongarch_asx
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr + i, 0));
        __m128 _s = (__m128)__lsx_vld(scale + i, 0);
        __m128 _bias = (__m128)__lsx_vld(bias + i, 0);
        _p = __lsx_vfmadd_s(_p, _s, _bias);
        __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr + i, 0, 0);
    }
    remain_size_start += nn_size * 4;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i] + bias[i]);
    }
}

static void scale_bf16s_no_bias_per_element_sse(unsigned short* ptr, const float* scale, int size, int num_threads)
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
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr + i, 0));
        __m256 _s = (__m256)__lasx_xvld(scale + i, 0);
        _p = __lasx_xvfmul_s(_p, _s);
        __lsx_vst(float2bfloat_avx(_p), ptr + i, 0);
    }
    remain_size_start += nn_size * 8;
#endif // __loongarch_asx
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr + i, 0));
        __m128 _s = (__m128)__lsx_vld(scale + i, 0);
        _p = __lsx_vfmul_s(_p, _s);
        __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr + i, 0, 0);
    }
    remain_size_start += nn_size * 4;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        ptr[i] = float32_to_bfloat16(bfloat16_to_float32(ptr[i]) * scale[i]);
    }
}

#endif // SCALE_LOONGARCH_BF16S_H
