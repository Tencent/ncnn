// Tencent is pleased to support the open source community by making ncnn available.
//
//                    https://opensource.org/licenses/BSD-3-Clause
//
// Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#ifndef SIGMOID_LOONGARCH_BF16S_H
#define SIGMOID_LOONGARCH_BF16S_H

static void sigmoid_bf16s(Mat& a, const Option& opt)
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
        __m256 _one_lasx = (__m256)__lasx_xvreplfr2vr_s(1.f);
        __m256 _zero_lasx = (__m256)__lasx_xvreplfr2vr_s(0.f);
        for (; i + 7 < size; i += 8)
        {
            __m256 _p = bfloat2float_avx((__m128i)__lsx_vld(ptr, 0));
            _p = __lasx_xvfdiv_s(_one_lasx, __lasx_xvfadd_s(_one_lasx, exp256_ps(__lasx_xvfsub_s(_zero_lasx, _p))));
            __lsx_vst(float2bfloat_avx(_p), ptr, 0);
            ptr += 8;
        }
#endif // __loongarch_asx
        __m128 _one = (__m128)__lsx_vreplfr2vr_s(1.f);
        __m128 _zero = (__m128)__lsx_vreplfr2vr_s(0.f);
        for (; i + 3 < size; i += 4)
        {
            __m128 _p = bfloat2float_sse((__m128i)__lsx_vld(ptr, 0));
            _p = __lsx_vfdiv_s(_one, __lsx_vfadd_s(_one, exp_ps(__lsx_vfsub_s(_zero, _p))));
            __lsx_vstelm_d(float2bfloat_sse(_p, _p), ptr, 0, 0);
            ptr += 4;
        }
#endif // __loongarch_sx
        for (; i < size; i++)
        {
            float v = bfloat16_to_float32(*ptr);
            v = 1.f / (1.f + expf(-v));
            *ptr = float32_to_bfloat16(v);
            ptr++;
        }
    }
}

#endif // SIGMOID_LOONGARCH_BF16S_H
