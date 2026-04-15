// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SELU_LOONGARCH_BF16S_H
#define SELU_LOONGARCH_BF16S_H

static void selu_bf16s(Mat& a, float alpha, float lambda, const Option& opt)
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
        __m256 _alpha_lasx = (__m256)__lasx_xvreplfr2vr_s(alpha);
        __m256 _lambda_lasx = (__m256)__lasx_xvreplfr2vr_s(lambda);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
            __m256 _pos = __lasx_xvfmul_s(_lambda_lasx, _p);
            __m256 _neg = __lasx_xvfmul_s(__lasx_xvfsub_s(__lasx_xvfmul_s(exp256_ps(_p), _alpha_lasx), _alpha_lasx), _lambda_lasx);
            __m256i _mask = __lasx_xvfcmp_clt_s(_p, _zero_lasx);
            _p = (__m256)__lasx_xvbitsel_v((__m256i)_pos, (__m256i)_neg, _mask);
            __lsx_vst(float2bfloat_avx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _zero = (__m128)__lsx_vreplfr2vr_s(0.f);
        __m128 _alpha = (__m128)__lsx_vreplfr2vr_s(alpha);
        __m128 _lambda = (__m128)__lsx_vreplfr2vr_s(lambda);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
            __m128 _pos = __lsx_vfmul_s(_lambda, _p);
            __m128 _neg = __lsx_vfmul_s(__lsx_vfsub_s(__lsx_vfmul_s(exp_ps(_p), _alpha), _alpha), _lambda);
            __m128i _mask = __lsx_vfcmp_clt_s(_p, _zero);
            _p = (__m128)__lsx_vbitsel_v((__m128i)_pos, (__m128i)_neg, _mask);
            __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            if (v < 0.f) v = (expf(v) * alpha - alpha) * lambda;
            else v = v * lambda;
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

#endif // SELU_LOONGARCH_BF16S_H
