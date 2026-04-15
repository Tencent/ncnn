// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef PRELU_LOONGARCH_BF16S_H
#define PRELU_LOONGARCH_BF16S_H

static void prelu_bf16s_sse(unsigned short* ptr, const float* slope, int size, int elempack)
{
#if __loongarch_sx
    __m128 _slope128 = (elempack == 4) ? (__m128)__lsx_vld(slope, 0) : (__m128)__lsx_vreplfr2vr_s(slope[0]);
    __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
#if __loongarch_asx
    __m256 _slope256 = (elempack == 8) ? (__m256)__lasx_xvld(slope, 0) : combine4x2_ps(_slope128, _slope128);
    __m256 _zero_lasx = (__m256)__lasx_xvreplgr2vr_w(0);
#endif // __loongarch_asx
#endif // __loongarch_sx
    float s = slope[0];

    int i = 0;
#if __loongarch_sx
#if __loongarch_asx
    for (; i + 7 < size; i += 8)
    {
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
        __m256 _ps = __lasx_xvfmul_s(_p, _slope256);
        __m256i _mask = __lasx_xvfcmp_clt_s(_p, _zero_lasx);
        _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, _mask);
        __lsx_vst(float2bfloat_avx(_p), ptr, 0);
        ptr += 8;
    }
#endif // __loongarch_asx
    for (; i + 3 < size; i += 4)
    {
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
        __m128 _ps = __lsx_vfmul_s(_p, _slope128);
        __m128i _mask = __lsx_vfcmp_clt_s(_p, _zero);
        _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, _mask);
        __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
        ptr += 4;
    }
#endif // __loongarch_sx
    for (; i < size; i++)
    {
        float v = bfloat16_to_float32(*ptr);
        if (v < 0.f)
            v *= s;
        *ptr = float32_to_bfloat16(v);
        ptr++;
    }
}

static void prelu_bf16s_per_element_sse(unsigned short* ptr, const float* slope, int size, int num_threads)
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
        __m256 _zero_lasx = (__m256)__lasx_xvreplgr2vr_w(0);
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr + i, 0));
        __m256 _slope = (__m256)__lasx_xvld(slope + i, 0);
        __m256 _ps = __lasx_xvfmul_s(_p, _slope);
        __m256i _mask = __lasx_xvfcmp_clt_s(_p, _zero_lasx);
        _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, _mask);
        __lsx_vst(float2bfloat_avx(_p), ptr + i, 0);
    }
    remain_size_start += nn_size * 8;
#endif // __loongarch_asx
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr + i, 0));
        __m128 _slope = (__m128)__lsx_vld(slope + i, 0);
        __m128 _ps = __lsx_vfmul_s(_p, _slope);
        __m128i _mask = __lsx_vfcmp_clt_s(_p, _zero);
        _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, _mask);
        __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr + i, 0, 0);
    }
    remain_size_start += nn_size * 4;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]);
        if (v < 0.f)
            v *= slope[i];
        ptr[i] = float32_to_bfloat16(v);
    }
}

static void prelu_bf16s_single_slope_sse(unsigned short* ptr, float slope, int size, int num_threads)
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
        __m256 _zero_lasx = (__m256)__lasx_xvreplgr2vr_w(0);
        __m256 _slope256 = (__m256)__lasx_xvreplfr2vr_s(slope);
        __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr + i, 0));
        __m256 _ps = __lasx_xvfmul_s(_p, _slope256);
        __m256i _mask = __lasx_xvfcmp_clt_s(_p, _zero_lasx);
        _p = (__m256)__lasx_xvbitsel_v((__m256i)_p, (__m256i)_ps, _mask);
        __lsx_vst(float2bfloat_avx(_p), ptr + i, 0);
    }
    remain_size_start += nn_size * 8;
#endif // __loongarch_asx
    nn_size = (size - remain_size_start) / 4;
    #pragma omp parallel for num_threads(num_threads)
    for (int ii = 0; ii < nn_size; ii++)
    {
        int i = remain_size_start + ii * 4;
        __m128 _zero = (__m128)__lsx_vreplgr2vr_w(0);
        __m128 _slope128 = (__m128)__lsx_vreplfr2vr_s(slope);
        __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr + i, 0));
        __m128 _ps = __lsx_vfmul_s(_p, _slope128);
        __m128i _mask = __lsx_vfcmp_clt_s(_p, _zero);
        _p = (__m128)__lsx_vbitsel_v((__m128i)_p, (__m128i)_ps, _mask);
        __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr + i, 0, 0);
    }
    remain_size_start += nn_size * 4;
#endif // __loongarch_sx
    #pragma omp parallel for num_threads(num_threads)
    for (int i = remain_size_start; i < size; i++)
    {
        float v = bfloat16_to_float32(ptr[i]);
        if (v < 0.f)
            v *= slope;
        ptr[i] = float32_to_bfloat16(v);
    }
}

#endif // PRELU_LOONGARCH_BF16S_H
