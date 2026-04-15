// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef GELU_LOONGARCH_BF16S_H
#define GELU_LOONGARCH_BF16S_H

static void gelu_bf16s(Mat& a, int fast_gelu, const Option& opt)
{
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    #pragma omp parallel for num_threads(opt.num_threads)
    for (int q = 0; q < channels; q++)
    {
        unsigned short* ptr = a.channel(q);

        int i = 0;
#if __loongarch_sx
#if __loongarch_asx
        if (fast_gelu)
        {
            __m256 _half_lasx = (__m256)__lasx_xvreplfr2vr_s(0.5f);
            __m256 _one_lasx = (__m256)__lasx_xvreplfr2vr_s(1.f);
            __m256 _fast1c_lasx = (__m256)__lasx_xvreplfr2vr_s(0.79788452f);
            __m256 _fast2c_lasx = (__m256)__lasx_xvreplfr2vr_s(0.044715f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));

                __m256 _cube = __lasx_xvfmul_s(_p, _p);
                _cube = __lasx_xvfmul_s(_p, _cube);

                __m256 _blob = __lasx_xvfmul_s(_fast2c_lasx, _cube);
                _blob = __lasx_xvfadd_s(_p, _blob);
                _blob = __lasx_xvfmul_s(_fast1c_lasx, _blob);
                _blob = tanh256_ps(_blob);
                _blob = __lasx_xvfadd_s(_one_lasx, _blob);

                _blob = __lasx_xvfmul_s(_half_lasx, __lasx_xvfmul_s(_blob, _p));

                __lsx_vst(float2bfloat_avx(_blob), ptr, 0);
                ptr += 8;
            }
        }
        else
        {
            __m256 _half_lasx = (__m256)__lasx_xvreplfr2vr_s(0.5f);
            __m256 _one_lasx = (__m256)__lasx_xvreplfr2vr_s(1.f);
            __m256 _inv_sqrt2_lasx = (__m256)__lasx_xvreplfr2vr_s(0.70710678f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));

                __m256 _erf = erf256_ps(__lasx_xvfmul_s(_p, _inv_sqrt2_lasx));
                __m256 _blob = __lasx_xvfadd_s(_one_lasx, _erf);
                _blob = __lasx_xvfmul_s(_half_lasx, __lasx_xvfmul_s(_blob, _p));

                __lsx_vst(float2bfloat_avx(_blob), ptr, 0);
                ptr += 8;
            }
        }
#endif // __loongarch_asx
        if (fast_gelu)
        {
            __m128 _half = (__m128)__lsx_vreplfr2vr_s(0.5f);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            __m128 _fast1c = (__m128)__lsx_vreplfr2vr_s(0.79788452f);
            __m128 _fast2c = (__m128)__lsx_vreplfr2vr_s(0.044715f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));

                __m128 _cube = __lsx_vfmul_s(_p, _p);
                _cube = __lsx_vfmul_s(_p, _cube);

                __m128 _blob = __lsx_vfmul_s(_fast2c, _cube);
                _blob = __lsx_vfadd_s(_p, _blob);
                _blob = __lsx_vfmul_s(_fast1c, _blob);
                _blob = tanh_ps(_blob);
                _blob = __lsx_vfadd_s(_one, _blob);

                _blob = __lsx_vfmul_s(_half, __lsx_vfmul_s(_blob, _p));

                __lsx_vstelm_d(float2bfloat_sse(_blob, _blob), ptr, 0, 0);
                ptr += 4;
            }
        }
        else
        {
            __m128 _half = (__m128)__lsx_vreplfr2vr_s(0.5f);
            __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
            __m128 _inv_sqrt2 = (__m128)__lsx_vreplfr2vr_s(0.70710678f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));

                __m128 _erf = erf_ps(__lsx_vfmul_s(_p, _inv_sqrt2));
                __m128 _blob = __lsx_vfadd_s(_one, _erf);
                _blob = __lsx_vfmul_s(_half, __lsx_vfmul_s(_blob, _p));

                __lsx_vstelm_d(float2bfloat_sse(_blob, _blob), ptr, 0, 0);
                ptr += 4;
            }
        }
#endif // __loongarch_sx
        if (fast_gelu)
        {
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + tanhf(0.79788452f * (v + 0.044715f * v * v * v)));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
        else
        {
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                v = 0.5f * v * (1.0f + erff(0.70710678f * v));
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
}

#endif // GELU_LOONGARCH_BF16S_H
