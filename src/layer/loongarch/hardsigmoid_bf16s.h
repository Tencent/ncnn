// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef HARDSIGMOID_LOONGARCH_BF16S_H
#define HARDSIGMOID_LOONGARCH_BF16S_H

static void hardsigmoid_bf16s(Mat& a, float alpha, float beta, const Option& opt)
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
        __m256 _zero_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        __m256 _one_lasx = (__m256)__lasx_xvreplfr2vr_s(1.f);
        __m256 _alpha_lasx = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _beta_lasx = (__m256)__lasx_xvreplfr2vr_s(beta);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
            _p = __lasx_xvfmadd_s(_alpha_lasx, _p, _beta_lasx);
            _p = __lasx_xvfmax_s(_p, _zero_lasx);
            _p = __lasx_xvfmin_s(_p, _one_lasx);
            __lsx_vst(float2bfloat_avx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _zero = (__m128)__lsx_vreplfr2vr_s(0.f);
        __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
        __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(alpha);
        __m128 _beta = (__m128)__lsx_vreplfr2vr_s(beta);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
            _p = __lsx_vfmadd_s(_alpha, _p, _beta);
            _p = __lsx_vfmax_s(_p, _zero);
            _p = __lsx_vfmin_s(_p, _one);
            __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = std::min(1.f, std::max(0.f, v * alpha + beta));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

#endif // HARDSIGMOID_LOONGARCH_BF16S_H
