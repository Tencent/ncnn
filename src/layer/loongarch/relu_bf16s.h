// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef RELU_LOONGARCH_BF16S_H
#define RELU_LOONGARCH_BF16S_H

static void relu_bf16s(Mat& a, float slope, const Option& opt)
{
    int w = a.w;
    int h = a.h;
    int d = a.d;
    int channels = a.c;
    int elempack = a.elempack;
    int size = w * h * d * elempack;

    if (slope == 0.f)
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = a.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
                _p = __lasx_xvfmax_s(_p, _zero_lasx);
                __lsx_vst(float2bfloat_avx(_p), ptr, 0);
                ptr += 8;
            }
#endif // __loongarch_asx
            __m128 _zero = (__m128)__lsx_vreplfr2vr_s(0.f);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
                _p = __lsx_vfmax_s(_p, _zero);
                __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
                ptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f) v = 0.f;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
    else
    {
        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            unsigned short* ptr = a.channel(q);

            int i = 0;
#if __loongarch_sx
#if __loongarch_asx
            __m256 _zero_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
            __m256 _slope_lasx = (__m256)__lasx_xvreplfr2vr_s(slope);
            for (; i + 7 < size; i += 8)
            {
                __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
                __m256 _pos = __lasx_xvfmax_s(_zero_lasx, _p);
                __m256 _neg = __lasx_xvfmin_s(_zero_lasx, _p);
                _p = __lasx_xvfadd_s(_pos, __lasx_xvfmul_s(_slope_lasx, _neg));
                __lsx_vst(float2bfloat_avx(_p), ptr, 0);
                ptr += 8;
            }
#endif // __loongarch_asx
            __m128 _zero = (__m128)__lsx_vreplfr2vr_s(0.f);
            __m128 _slope = (__m128)__lsx_vreplfr2vr_s(slope);
            for (; i + 3 < size; i += 4)
            {
                __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
                __m128 _pos = __lsx_vfmax_s(_zero, _p);
                __m128 _neg = __lsx_vfmin_s(_zero, _p);
                _p = __lsx_vfadd_s(_pos, __lsx_vfmul_s(_slope, _neg));
                __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
                ptr += 4;
            }
#endif // __loongarch_sx
            for (; i < size; i++)
            {
                float v = bfloat16_to_float32(*ptr);
                if (v < 0.f) v *= slope;
                *ptr = float32_to_bfloat16(v);
                ptr++;
            }
        }
    }
}

#endif // RELU_LOONGARCH_BF16S_H
